#pragma once
/**
 * Quantized NNUE model — int16 accumulator, int8 hidden layers.
 *
 * Precision per layer (Stockfish-style):
 *   Accumulator weights/bias: int16  (float * QA)
 *   Accumulator state:        int16  (incremental add/sub in int16)
 *   SCReLU output:            uint8  (clamp(0,QA), square, shift → [0,127])
 *   fc1 weights:              int8   (float * QW_FC)
 *   fc1 bias:                 int32  (pre-scaled: float * QA * QW_FC)
 *   fc2/fc3:                  same scheme
 *   PSQT:                     int16  (last accumulator element)
 *
 * Binary format v3:
 *   Header: magic(u32) version(u32=3) num_features(i32) acc_size(i32)
 *           h1(i32) h2(i32) num_buckets(i32) bucket_divisor(i32)
 *   Data:   acc_weight  int16[features*(acc_size+1)]
 *           acc_bias    int16[acc_size+1]
 *           fc1_weight  int8[h1*nb * acc_size*2]
 *           fc1_bias    int32[h1*nb]
 *           fc2_weight  int8[h2*nb * h1]
 *           fc2_bias    int32[h2*nb]
 *           fc3_weight  int8[nb * h2]
 *           fc3_bias    int32[nb]
 */

#include "nnue_features.hpp"
#include <vector>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <string>

#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace nnue {

// Quantization constants
constexpr int QA = 255;     // accumulator scale
constexpr int QW_FC = 64;   // fc weight scale (must be power of 2)
constexpr int QW_FC_SHIFT = 6; // log2(QW_FC)
// SCReLU: int16 in [0, QA] → square → shift right → uint8 in [0, 127]
// screlu_shift = such that QA*QA >> shift fits in [0, 127]
// QA=255: 255*255=65025. 65025 >> 9 = 127. So shift=9.
constexpr int SCRELU_SHIFT = 9;
constexpr int QO = 127;        // CReLU / SCReLU output max (uint8 range)

constexpr uint32_t NNUE_MAGIC = 0x454E4E55;
constexpr uint32_t NNUE_VERSION = 3;

// Padded accumulator size: round (acc_size+1) up to multiple of 16 for int16 AVX2
inline int acc_pad16(int acc_full) { return (acc_full + 15) & ~15; }

#ifdef __AVX2__
inline int32_t hsum_epi32(__m256i v) {
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i lo = _mm256_castsi256_si128(v);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_cvtsi128_si32(lo);
}
#endif

class NNUEModel {
public:
    int accumulator_size = 256;
    int hidden1_size = 32;
    int hidden2_size = 32;
    int num_buckets = 4;
    int bucket_divisor = 20;
    int acc_full_ = 257;     // accumulator_size + 1 (PSQT)
    int acc_padded_ = 272;   // acc_full_ rounded up to 16

    // Quantized weights
    std::vector<int16_t> acc_weight;   // [NUM_FEATURES * acc_padded_]
    std::vector<int16_t> acc_bias;     // [acc_padded_]
    std::vector<int8_t>  fc1_weight;   // [h1*nb * acc_size*2]  (padded to 32)
    std::vector<int32_t> fc1_bias;     // [h1*nb]
    std::vector<int8_t>  fc2_weight;   // [h2*nb * h1]  (padded to 32)
    std::vector<int32_t> fc2_bias;     // [h2*nb]
    std::vector<int8_t>  fc3_weight;   // [nb * h2]  (padded to 32)
    std::vector<int32_t> fc3_bias;     // [nb]

    bool loaded = false;

    NNUEModel() = default;

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open())
            throw std::runtime_error("Cannot open NNUE weights: " + path);

        uint32_t magic, version;
        f.read(reinterpret_cast<char*>(&magic), 4);
        f.read(reinterpret_cast<char*>(&version), 4);
        if (magic != NNUE_MAGIC)
            throw std::runtime_error("Invalid NNUE magic");
        if (version != NNUE_VERSION)
            throw std::runtime_error("NNUE version " + std::to_string(version) +
                                     " (expected " + std::to_string(NNUE_VERSION) + ")");

        int32_t sizes[6];
        f.read(reinterpret_cast<char*>(sizes), 24);
        if (sizes[0] != NUM_FEATURES)
            throw std::runtime_error("Feature count mismatch");

        accumulator_size = sizes[1];
        hidden1_size = sizes[2];
        hidden2_size = sizes[3];
        num_buckets = sizes[4];
        bucket_divisor = sizes[5];
        acc_full_ = accumulator_size + 1;
        acc_padded_ = acc_pad16(acc_full_);

        const int h1 = hidden1_size, h2 = hidden2_size, nb = num_buckets;
        const int fc1_in = accumulator_size * 2;
        // Pad fc input sizes to 32 for int8 AVX2
        const int fc1_in_pad = (fc1_in + 31) & ~31;
        const int h1_pad = (h1 + 31) & ~31;
        const int h2_pad = (h2 + 31) & ~31;

        // Read accumulator (stored dense, we pad to acc_padded_)
        {
            std::vector<int16_t> tmp(NUM_FEATURES * acc_full_);
            f.read(reinterpret_cast<char*>(tmp.data()), tmp.size() * 2);
            acc_weight.assign(NUM_FEATURES * acc_padded_, 0);
            for (int feat = 0; feat < NUM_FEATURES; ++feat)
                std::memcpy(&acc_weight[feat * acc_padded_], &tmp[feat * acc_full_],
                            acc_full_ * sizeof(int16_t));
        }
        {
            std::vector<int16_t> tmp(acc_full_);
            f.read(reinterpret_cast<char*>(tmp.data()), tmp.size() * 2);
            acc_bias.assign(acc_padded_, 0);
            std::memcpy(acc_bias.data(), tmp.data(), acc_full_ * sizeof(int16_t));
        }

        // fc1: [h1*nb, fc1_in] → pad cols to fc1_in_pad
        {
            int rows = h1 * nb;
            std::vector<int8_t> tmp(rows * fc1_in);
            f.read(reinterpret_cast<char*>(tmp.data()), tmp.size());
            fc1_weight.assign(rows * fc1_in_pad, 0);
            for (int r = 0; r < rows; ++r)
                std::memcpy(&fc1_weight[r * fc1_in_pad], &tmp[r * fc1_in], fc1_in);
        }
        fc1_bias.resize(h1 * nb);
        f.read(reinterpret_cast<char*>(fc1_bias.data()), fc1_bias.size() * 4);

        // fc2: [h2*nb, h1] → pad cols to h1_pad
        {
            int rows = h2 * nb;
            std::vector<int8_t> tmp(rows * h1);
            f.read(reinterpret_cast<char*>(tmp.data()), tmp.size());
            fc2_weight.assign(rows * h1_pad, 0);
            for (int r = 0; r < rows; ++r)
                std::memcpy(&fc2_weight[r * h1_pad], &tmp[r * h1], h1);
        }
        fc2_bias.resize(h2 * nb);
        f.read(reinterpret_cast<char*>(fc2_bias.data()), fc2_bias.size() * 4);

        // fc3: [nb, h2] → pad cols to h2_pad
        {
            std::vector<int8_t> tmp(nb * h2);
            f.read(reinterpret_cast<char*>(tmp.data()), tmp.size());
            fc3_weight.assign(nb * h2_pad, 0);
            for (int r = 0; r < nb; ++r)
                std::memcpy(&fc3_weight[r * h2_pad], &tmp[r * h2], h2);
        }
        fc3_bias.resize(nb);
        f.read(reinterpret_cast<char*>(fc3_bias.data()), fc3_bias.size() * 4);

        loaded = true;
    }

    int get_acc_full_size() const { return acc_full_; }
    int get_acc_padded_size() const { return acc_padded_; }

    // ─── int16 accumulator operations ────────────────────────────

    void accumulate_raw_q(const int* feats, int n, int16_t* output) const {
#ifdef __AVX2__
        for (int j = 0; j < acc_padded_; j += 16) {
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output[j]),
                _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&acc_bias[j])));
        }
        for (int fi = 0; fi < n; ++fi) {
            const int16_t* row = &acc_weight[feats[fi] * acc_padded_];
            for (int j = 0; j < acc_padded_; j += 16) {
                __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&output[j]));
                __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&row[j]));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output[j]),
                    _mm256_add_epi16(a, b));
            }
        }
#else
        std::memcpy(output, acc_bias.data(), acc_padded_ * sizeof(int16_t));
        for (int fi = 0; fi < n; ++fi) {
            const int16_t* row = &acc_weight[feats[fi] * acc_padded_];
            for (int j = 0; j < acc_padded_; ++j) output[j] += row[j];
        }
#endif
    }

    void acc_add_feature(int16_t* acc_out, int feat_idx) const {
        const int16_t* row = &acc_weight[feat_idx * acc_padded_];
#ifdef __AVX2__
        for (int j = 0; j < acc_padded_; j += 16) {
            __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&acc_out[j]));
            __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&row[j]));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&acc_out[j]),
                _mm256_add_epi16(a, b));
        }
#else
        for (int j = 0; j < acc_padded_; ++j) acc_out[j] += row[j];
#endif
    }

    void acc_sub_feature(int16_t* acc_out, int feat_idx) const {
        const int16_t* row = &acc_weight[feat_idx * acc_padded_];
#ifdef __AVX2__
        for (int j = 0; j < acc_padded_; j += 16) {
            __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&acc_out[j]));
            __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&row[j]));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&acc_out[j]),
                _mm256_sub_epi16(a, b));
        }
#else
        for (int j = 0; j < acc_padded_; ++j) acc_out[j] -= row[j];
#endif
    }

    // ─── Quantized evaluation ────────────────────────────────────

    /**
     * Evaluate from pre-computed int16 accumulators.
     * stm_acc/nstm_acc are int16 accumulator arrays (acc_padded_ elements).
     */
    float evaluate_from_acc(const int16_t* stm_acc, const int16_t* nstm_acc,
                            int piece_count) const {
        const int acc = accumulator_size;
        const int h1 = hidden1_size, h2 = hidden2_size;
        const int fc1_in = acc * 2;
        const int fc1_in_pad = (fc1_in + 31) & ~31;
        const int h1_pad = (h1 + 31) & ~31;
        const int h2_pad = (h2 + 31) & ~31;

        // PSQT: int16 → float
        float psqt = (static_cast<float>(stm_acc[acc]) -
                       static_cast<float>(nstm_acc[acc])) * (0.5f / QA);

        // SCReLU: int16 → uint8 [0, 127]
        // clamp(x, 0, QA), then (x*x) >> SCRELU_SHIFT
        alignas(32) uint8_t fc1_input[1024];  // [fc1_in_pad]
        screlu_i16_to_u8(stm_acc, fc1_input, acc);
        screlu_i16_to_u8(nstm_acc, fc1_input + acc, acc);
        // Zero-pad remaining
        if (fc1_in_pad > fc1_in)
            std::memset(fc1_input + fc1_in, 0, fc1_in_pad - fc1_in);

        int bucket = std::min(piece_count / bucket_divisor, num_buckets - 1);

        // fc1: uint8 input × int8 weight → int32 → CReLU → uint8
        alignas(32) uint8_t h1_out[64];  // [h1_pad]
        {
            const int8_t* w = &fc1_weight[bucket * h1 * fc1_in_pad];
            const int32_t* b = &fc1_bias[bucket * h1];
            linear_crelu_q(w, b, fc1_input, h1_out, h1, fc1_in_pad);
        }
        if (h1_pad > h1) std::memset(h1_out + h1, 0, h1_pad - h1);

        // fc2: uint8 × int8 → CReLU → uint8
        alignas(32) uint8_t h2_out[64];  // [h2_pad]
        {
            const int8_t* w = &fc2_weight[bucket * h2 * h1_pad];
            const int32_t* b = &fc2_bias[bucket * h2];
            linear_crelu_q(w, b, h1_out, h2_out, h2, h1_pad);
        }
        if (h2_pad > h2) std::memset(h2_out + h2, 0, h2_pad - h2);

        // fc3: uint8 × int8 → int32 → float
        const int8_t* fc3_w = &fc3_weight[bucket * h2_pad];
        int32_t fc3_sum = fc3_bias[bucket];
        fc3_sum += dot_u8_i8(h2_out, fc3_w, h2_pad);

        // Dequantize: raw int32 → float
        // Scale chain: acc*QA, screlu square>>SCRELU_SHIFT, fc_w*QW_FC
        // fc1 output was: sum(input_u8 * weight_i8) + bias_i32
        //   input_u8 range [0,127], weight_i8 range ~[-64,64]
        //   bias is pre-scaled by QA*QW_FC (handled in export)
        // After CReLU clamp to [0,127], fc2 same, fc3 produces int32
        // Total dequant factor: 1.0 / (QA * QW_FC * QW_FC * QW_FC)
        // But each CReLU divides by 127 conceptually. Let's use the
        // simple formula: output = fc3_sum / (QA * QW_FC)
        // because: acc is in QA scale, screlu squares and shifts to [0,127],
        //          fc1 multiplies by QW_FC-scale weights → int32,
        //          CReLU clamps to [0,127],
        //          fc2 same → int32, CReLU → [0,127],
        //          fc3 → int32
        // The actual dequant is: fc3_sum / (QW_FC * QO * QO)
        // where QO=127 is the CReLU output range.
        // But bias was pre-scaled to match, so:
        constexpr float dequant = 1.0f / (QW_FC * QO * QO);
        float ls_out = static_cast<float>(fc3_sum) * dequant;

        return ls_out + psqt;
    }

    /**
     * Evaluate a board (full feature extraction + accumulation).
     * For compatibility / testing only — search uses incremental accumulators.
     */
    float evaluate_board(const uttt::Board& board) const {
        int stm_feats[NUM_FEATURES], nstm_feats[NUM_FEATURES];
        int stm_n, nstm_n;
        extract_features(board, stm_feats, stm_n, nstm_feats, nstm_n);

        alignas(32) int16_t stm_acc[512], nstm_acc[512];
        accumulate_raw_q(stm_feats, stm_n, stm_acc);
        accumulate_raw_q(nstm_feats, nstm_n, nstm_acc);

        int pc = 0;
        for (int i = 0; i < stm_n; ++i) if (stm_feats[i] < 162) ++pc;
        return evaluate_from_acc(stm_acc, nstm_acc, pc);
    }

private:
    // SCReLU: int16 accumulator → uint8 [0, 127]
    // clamp(x, 0, QA), then (x*x) >> SCRELU_SHIFT
    // Optimized: use _mm256_mulhi_epu16 to avoid 32-bit expansion + permutes
    static void screlu_i16_to_u8(const int16_t* input, uint8_t* output, int n) {
#ifdef __AVX2__
        const __m256i zero = _mm256_setzero_si256();
        const __m256i qa = _mm256_set1_epi16(QA);
        // mulhi_epu16(a,a) gives (a*a)>>16. We need >>9.
        // So we pre-shift left by 7: mulhi_epu16(a<<7, a) = (a*a<<7)>>16 = (a*a)>>9
        // But a<<7 might overflow uint16 if a>255 — after clamp a is in [0,255], a<<7 max=32640, fits.
        for (int j = 0; j < n; j += 32) {
            // Process 32 elements (two __m256i of int16) → 32 uint8
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input[j]));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input[j + 16]));
            // clamp to [0, QA]
            v0 = _mm256_max_epi16(v0, zero);
            v0 = _mm256_min_epi16(v0, qa);
            v1 = _mm256_max_epi16(v1, zero);
            v1 = _mm256_min_epi16(v1, qa);
            // square >> SCRELU_SHIFT via mulhi trick
            __m256i shifted0 = _mm256_slli_epi16(v0, 7);  // a << 7
            __m256i shifted1 = _mm256_slli_epi16(v1, 7);
            // mulhi_epu16(a<<7, a) = (a * a << 7) >> 16 = (a*a) >> 9
            __m256i sq0 = _mm256_mulhi_epu16(shifted0, v0);  // 16x uint16 in [0, 127]
            __m256i sq1 = _mm256_mulhi_epu16(shifted1, v1);
            // Pack 2x 16x uint16 → 32x uint8
            __m256i packed = _mm256_packus_epi16(sq0, sq1);
            // packus interleaves 128-bit lanes: fix with permute
            packed = _mm256_permute4x64_epi64(packed, 0xD8);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output[j]), packed);
        }
        // Handle remainder if n is not multiple of 32 (for our arch, n=256 so no remainder)
#else
        for (int j = 0; j < n; ++j) {
            int v = std::max(0, std::min((int)input[j], QA));
            output[j] = static_cast<uint8_t>((v * v) >> SCRELU_SHIFT);
        }
#endif
    }

    // Linear + CReLU (quantized): uint8 input × int8 weight → clamp [0,127] → uint8
    // Tiled: process 4 output neurons simultaneously to share input loads
    static void linear_crelu_q(const int8_t* weight, const int32_t* bias,
                               const uint8_t* input, uint8_t* output,
                               int out_size, int in_size_padded) {
#ifdef __AVX2__
        const __m256i ones16 = _mm256_set1_epi16(1);
        int o = 0;
        // Process 4 output neurons at a time
        for (; o + 3 < out_size; o += 4) {
            const int8_t* r0 = &weight[(o + 0) * in_size_padded];
            const int8_t* r1 = &weight[(o + 1) * in_size_padded];
            const int8_t* r2 = &weight[(o + 2) * in_size_padded];
            const int8_t* r3 = &weight[(o + 3) * in_size_padded];
            __m256i s0 = _mm256_setzero_si256();
            __m256i s1 = _mm256_setzero_si256();
            __m256i s2 = _mm256_setzero_si256();
            __m256i s3 = _mm256_setzero_si256();
            for (int i = 0; i < in_size_padded; i += 32) {
                __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input[i]));
                __m256i p0 = _mm256_maddubs_epi16(va, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&r0[i])));
                __m256i p1 = _mm256_maddubs_epi16(va, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&r1[i])));
                __m256i p2 = _mm256_maddubs_epi16(va, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&r2[i])));
                __m256i p3 = _mm256_maddubs_epi16(va, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&r3[i])));
                s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(p0, ones16));
                s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(p1, ones16));
                s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(p2, ones16));
                s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(p3, ones16));
            }
            int32_t sums[4] = {
                hsum_epi32(s0) + bias[o + 0],
                hsum_epi32(s1) + bias[o + 1],
                hsum_epi32(s2) + bias[o + 2],
                hsum_epi32(s3) + bias[o + 3]
            };
            for (int k = 0; k < 4; ++k) {
                int32_t a = sums[k] >> QW_FC_SHIFT;
                output[o + k] = static_cast<uint8_t>(a < 0 ? 0 : (a > QO ? QO : a));
            }
        }
        // Handle remaining neurons (< 4)
        for (; o < out_size; ++o) {
            const int8_t* row = &weight[o * in_size_padded];
            __m256i sum = _mm256_setzero_si256();
            for (int i = 0; i < in_size_padded; i += 32) {
                __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input[i]));
                __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&row[i]));
                __m256i prod = _mm256_maddubs_epi16(va, vb);
                sum = _mm256_add_epi32(sum, _mm256_madd_epi16(prod, ones16));
            }
            int32_t s = hsum_epi32(sum) + bias[o];
            int32_t a = s >> QW_FC_SHIFT;
            output[o] = static_cast<uint8_t>(a < 0 ? 0 : (a > QO ? QO : a));
        }
#else
        for (int o = 0; o < out_size; ++o) {
            const int8_t* row = &weight[o * in_size_padded];
            int32_t sum = bias[o];
            for (int i = 0; i < in_size_padded; ++i)
                sum += static_cast<int32_t>(input[i]) * static_cast<int32_t>(row[i]);
            int32_t a = sum >> QW_FC_SHIFT;
            output[o] = static_cast<uint8_t>(a < 0 ? 0 : (a > QO ? QO : a));
        }
#endif
    }

    // Dot product: uint8[n] × int8[n] → int32
    static int32_t dot_u8_i8(const uint8_t* a, const int8_t* b, int n) {
#ifdef __AVX2__
        __m256i sum = _mm256_setzero_si256();
        const __m256i ones16 = _mm256_set1_epi16(1);
        for (int i = 0; i < n; i += 32) {
            __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
            __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
            __m256i prod = _mm256_maddubs_epi16(va, vb);
            sum = _mm256_add_epi32(sum, _mm256_madd_epi16(prod, ones16));
        }
        return hsum_epi32(sum);
#else
        int32_t s = 0;
        for (int i = 0; i < n; ++i) s += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
        return s;
#endif
    }
};

} // namespace nnue
