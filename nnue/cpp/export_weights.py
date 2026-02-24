"""Export PyTorch NNUE weights to C++ binary format.

Binary format v2 (SCReLU + PSQT + Layer Stack Buckets):
  [0-3]   uint32 magic (0x454E4E55)
  [4-7]   uint32 version (2)
  [8-11]  int32  num_features
  [12-15] int32  accumulator_size  (hidden part, excludes PSQT)
  [16-19] int32  hidden1_size
  [20-23] int32  hidden2_size
  [24-27] int32  num_buckets
  [28-31] int32  bucket_divisor
  [32..]  float32 weights:
    acc_weight_T  [num_features * (acc_size+1)]  (transposed, includes PSQT column)
    acc_bias      [acc_size+1]                    (includes PSQT bias)
    --- per-bucket layer stack (fc1, fc2, fc3 are stored with bucket dim folded in) ---
    fc1_weight    [h1*num_buckets * acc_size*2]
    fc1_bias      [h1*num_buckets]
    fc2_weight    [h2*num_buckets * h1]
    fc2_bias      [h2*num_buckets]
    fc3_weight    [1*num_buckets * h2]
    fc3_bias      [1*num_buckets]
"""
import struct
import numpy as np
import torch

from nnue.core.model import NNUE
from nnue.core.features import NUM_FEATURES


NNUE_MAGIC = 0x454E4E55
NNUE_VERSION = 2


def export_weights(model: NNUE, output_path: str):
    """Export NNUE v2 model weights to binary file for C++ engine."""
    sd = model.state_dict()

    # Accumulator: (acc_size+1, 199) weight, (acc_size+1,) bias
    acc_weight = sd['accumulator.weight'].cpu().numpy()
    acc_bias = sd['accumulator.bias'].cpu().numpy()

    acc_size = model.accumulator_size  # hidden part (excludes PSQT)
    h1_size = model.layer_stack.hidden1_size
    h2_size = model.layer_stack.hidden2_size
    num_buckets = model.num_buckets
    bucket_divisor = model.bucket_divisor

    # Transpose accumulator for cache-friendly sparse access: (199, acc_size+1)
    acc_weight_t = acc_weight.T.copy()

    # Layer stack weights (already stored with bucket dim folded)
    fc1_weight = sd['layer_stack.fc1.weight'].cpu().numpy()  # (h1*nb, acc_size*2)
    fc1_bias = sd['layer_stack.fc1.bias'].cpu().numpy()       # (h1*nb,)
    fc2_weight = sd['layer_stack.fc2.weight'].cpu().numpy()   # (h2*nb, h1)
    fc2_bias = sd['layer_stack.fc2.bias'].cpu().numpy()       # (h2*nb,)
    fc3_weight = sd['layer_stack.fc3.weight'].cpu().numpy()   # (1*nb, h2)
    fc3_bias = sd['layer_stack.fc3.bias'].cpu().numpy()       # (1*nb,)

    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<II', NNUE_MAGIC, NNUE_VERSION))
        f.write(struct.pack('<iiiiii', NUM_FEATURES, acc_size,
                            h1_size, h2_size, num_buckets, bucket_divisor))

        # Weights
        f.write(acc_weight_t.astype(np.float32).tobytes())
        f.write(acc_bias.astype(np.float32).tobytes())
        f.write(fc1_weight.astype(np.float32).tobytes())
        f.write(fc1_bias.astype(np.float32).tobytes())
        f.write(fc2_weight.astype(np.float32).tobytes())
        f.write(fc2_bias.astype(np.float32).tobytes())
        f.write(fc3_weight.reshape(-1).astype(np.float32).tobytes())
        f.write(fc3_bias.astype(np.float32).tobytes())

    total_params = (acc_weight_t.size + acc_bias.size +
                    fc1_weight.size + fc1_bias.size +
                    fc2_weight.size + fc2_bias.size +
                    fc3_weight.size + fc3_bias.size)
    print(f"Exported {total_params} parameters ({total_params * 4} bytes) [v2]")
    print(f"  accumulator: ({NUM_FEATURES}, {acc_size}+1 PSQT)")
    print(f"  layer_stack: {num_buckets} buckets")
    print(f"  fc1: ({h1_size}*{num_buckets}, {acc_size * 2})")
    print(f"  fc2: ({h2_size}*{num_buckets}, {h1_size})")
    print(f"  fc3: (1*{num_buckets}, {h2_size})")
    print(f"  bucket_divisor: {bucket_divisor}")
    print(f"  → {output_path}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m nnue.cpp.export_weights <model.pt> [output.nnue]")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else model_path.replace('.pt', '.nnue')

    model = NNUE.load(model_path)
    export_weights(model, output_path)
