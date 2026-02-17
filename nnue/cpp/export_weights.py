"""Export PyTorch NNUE weights to C++ binary format.

Binary format:
  [0-3]   uint32 magic (0x454E4E55)
  [4-7]   uint32 version (1)
  [8-23]  int32 x4: num_features, accumulator_size, hidden1_size, hidden2_size
  [24..]  float32 weights:
    acc_weight_T  [num_features * acc_size]  (transposed for sparse access)
    acc_bias      [acc_size]
    fc1_weight    [h1 * acc_size*2]
    fc1_bias      [h1]
    fc2_weight    [h2 * h1]
    fc2_bias      [h2]
    fc3_weight    [1 * h2]
    fc3_bias      [1]
"""
import struct
import numpy as np
import torch

from nnue.core.model import NNUE
from nnue.core.features import NUM_FEATURES


NNUE_MAGIC = 0x454E4E55
NNUE_VERSION = 1


def export_weights(model: NNUE, output_path: str):
    """Export NNUE model weights to binary file for C++ engine."""
    sd = model.state_dict()

    acc_weight = sd['accumulator.weight'].cpu().numpy()  # (acc_size, 199)
    acc_bias = sd['accumulator.bias'].cpu().numpy()       # (acc_size,)
    fc1_weight = sd['fc1.weight'].cpu().numpy()           # (h1, acc_size*2)
    fc1_bias = sd['fc1.bias'].cpu().numpy()               # (h1,)
    fc2_weight = sd['fc2.weight'].cpu().numpy()           # (h2, h1)
    fc2_bias = sd['fc2.bias'].cpu().numpy()               # (h2,)
    fc3_weight = sd['fc3.weight'].cpu().numpy()           # (1, h2)
    fc3_bias = sd['fc3.bias'].cpu().numpy()               # (1,)

    acc_size = acc_weight.shape[0]
    h1_size = fc1_weight.shape[0]
    h2_size = fc2_weight.shape[0]

    # Transpose accumulator for cache-friendly sparse access: (199, acc_size)
    acc_weight_t = acc_weight.T.copy()

    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<II', NNUE_MAGIC, NNUE_VERSION))
        f.write(struct.pack('<iiii', NUM_FEATURES, acc_size, h1_size, h2_size))

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
    print(f"Exported {total_params} parameters ({total_params * 4} bytes)")
    print(f"  accumulator: ({NUM_FEATURES}, {acc_size})")
    print(f"  fc1: ({h1_size}, {acc_size * 2})")
    print(f"  fc2: ({h2_size}, {h1_size})")
    print(f"  fc3: (1, {h2_size})")
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
