"""Export PyTorch NNUE weights to quantized C++ binary format (v3).

Quantization scheme:
  Accumulator weights/bias: float * QA(255) → int16
  fc1/fc2/fc3 weights:      float * QW(64)  → int8
  fc1 bias: float * QA * QW → int32  (accumulator output scale × weight scale)
  fc2 bias: float * QO * QW → int32  (QO=127, CReLU output scale × weight scale)
  fc3 bias: float * QO * QW → int32

Binary format v3:
  Header: magic(u32) version(u32=3) + 6× int32 sizes
  Data:   acc_weight_T  int16[features*(acc_size+1)]
          acc_bias      int16[acc_size+1]
          fc1_weight    int8[h1*nb * acc_size*2]
          fc1_bias      int32[h1*nb]
          fc2_weight    int8[h2*nb * h1]
          fc2_bias      int32[h2*nb]
          fc3_weight    int8[nb * h2]
          fc3_bias      int32[nb]
"""
import struct
import numpy as np
import torch

from nnue.core.model import NNUE
from nnue.core.features import NUM_FEATURES


NNUE_MAGIC = 0x454E4E55
NNUE_VERSION = 3

# Must match C++ constants in nnue_model.hpp
QA = 255       # accumulator scale
QW_FC = 64     # fc weight scale
QO = 127       # CReLU output max


def quantize_clamp(arr, scale, dtype, lo, hi):
    """Quantize float array: round(arr * scale), clamp to [lo, hi], cast to dtype."""
    q = np.round(arr * scale).astype(np.int64)
    q = np.clip(q, lo, hi)
    return q.astype(dtype)


def export_weights(model: NNUE, output_path: str):
    """Export NNUE model weights to quantized binary file (v3) for C++ engine."""
    sd = model.state_dict()

    acc_weight = sd['accumulator.weight'].cpu().numpy()  # (acc_size+1, 199)
    acc_bias = sd['accumulator.bias'].cpu().numpy()       # (acc_size+1,)

    acc_size = model.accumulator_size
    h1_size = model.layer_stack.hidden1_size
    h2_size = model.layer_stack.hidden2_size
    num_buckets = model.num_buckets
    bucket_divisor = model.bucket_divisor

    # Transpose accumulator: (199, acc_size+1) for cache-friendly sparse access
    acc_weight_t = acc_weight.T.copy()

    fc1_weight = sd['layer_stack.fc1.weight'].cpu().numpy()  # (h1*nb, acc_size*2)
    fc1_bias = sd['layer_stack.fc1.bias'].cpu().numpy()
    fc2_weight = sd['layer_stack.fc2.weight'].cpu().numpy()  # (h2*nb, h1)
    fc2_bias = sd['layer_stack.fc2.bias'].cpu().numpy()
    fc3_weight = sd['layer_stack.fc3.weight'].cpu().numpy()  # (nb, h2)
    fc3_bias = sd['layer_stack.fc3.bias'].cpu().numpy()

    # ── Quantize ──
    # Accumulator: float → int16 (scale QA)
    acc_w_q = quantize_clamp(acc_weight_t, QA, np.int16, -32768, 32767)
    acc_b_q = quantize_clamp(acc_bias, QA, np.int16, -32768, 32767)

    # fc weights: float → int8 (scale QW_FC)
    fc1_w_q = quantize_clamp(fc1_weight, QW_FC, np.int8, -128, 127)
    fc2_w_q = quantize_clamp(fc2_weight, QW_FC, np.int8, -128, 127)
    fc3_w_q = quantize_clamp(fc3_weight.reshape(-1, h2_size), QW_FC, np.int8, -128, 127)

    # fc biases: pre-scaled int32
    # fc1 input is SCReLU output in [0,127] (after int16 acc → square >> shift)
    # fc1 bias scale = QA * QW_FC (accumulator_scale * weight_scale)
    # But SCReLU squares and shifts: effective input scale = QA²/2^SCRELU_SHIFT
    # After SCReLU, input is uint8 [0,127]. Weight is int8 [-128,127] scaled by QW_FC.
    # The dot product gives: sum(input_u8 * weight_i8).
    # To match float semantics: float_output = sum(screlu(acc/QA) * weight/QW_FC)
    # In quantized: q_output = sum(screlu_u8 * weight_i8) + bias_i32
    # We need bias_i32 = float_bias * QO * QW_FC
    # because screlu_u8 max is QO=127, and weight_i8 is float*QW_FC.
    # More precisely: the product screlu_u8 * weight_i8 represents:
    #   (float_screlu * QO) * (float_weight * QW_FC) = float_product * QO * QW_FC
    # So bias should be: float_bias * QO * QW_FC
    fc1_b_q = np.round(fc1_bias * QO * QW_FC).astype(np.int32)
    fc2_b_q = np.round(fc2_bias * QO * QW_FC).astype(np.int32)
    fc3_b_q = np.round(fc3_bias.reshape(-1) * QO * QW_FC).astype(np.int32)

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<II', NNUE_MAGIC, NNUE_VERSION))
        f.write(struct.pack('<iiiiii', NUM_FEATURES, acc_size,
                            h1_size, h2_size, num_buckets, bucket_divisor))

        f.write(acc_w_q.tobytes())
        f.write(acc_b_q.tobytes())
        f.write(fc1_w_q.tobytes())
        f.write(fc1_b_q.tobytes())
        f.write(fc2_w_q.tobytes())
        f.write(fc2_b_q.tobytes())
        f.write(fc3_w_q.tobytes())
        f.write(fc3_b_q.tobytes())

    file_size = (acc_w_q.nbytes + acc_b_q.nbytes +
                 fc1_w_q.nbytes + fc1_b_q.nbytes +
                 fc2_w_q.nbytes + fc2_b_q.nbytes +
                 fc3_w_q.nbytes + fc3_b_q.nbytes + 32)  # +32 for header
    print(f"Exported quantized v3: {file_size} bytes")
    print(f"  QA={QA}, QW_FC={QW_FC}, QO={QO}")
    print(f"  accumulator: int16 ({NUM_FEATURES}, {acc_size}+1)")
    print(f"  fc1: int8 ({h1_size}*{num_buckets}, {acc_size*2}), bias int32")
    print(f"  fc2: int8 ({h2_size}*{num_buckets}, {h1_size}), bias int32")
    print(f"  fc3: int8 ({num_buckets}, {h2_size}), bias int32")
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
