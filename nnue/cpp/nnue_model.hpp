#pragma once

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

#ifdef __AVX2__
// Horizontal sum of 8 floats in __m256
inline float hsum256(__m256 v) {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    lo = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, lo);
    lo = _mm_add_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}
#endif

constexpr uint32_t NNUE_MAGIC = 0x454E4E55;  // "UNNE"
constexpr uint32_t NNUE_VERSION = 2;

/**
 * NNUE model v2 for inference.
 *
 * Architecture (Stockfish SFNNv5+ style):
 *   Input (199 sparse) → Accumulator (199 → acc_size+1, shared)
 *     ├── acc_size hidden → SCReLU (clamp(x,0,1)²)
 *     └── 1 PSQT output (direct to final output)
 *   Concat STM+NSTM (acc_size*2) → LayerStack[bucket] (→h1→CReLU→h2→CReLU→1)
 *   output = layer_stack_result + (psqt_stm - psqt_nstm) / 2
 *
 * Weight file format v2 (binary):
 *   [0-3]   uint32 magic (0x454E4E55)
 *   [4-7]   uint32 version (2)
 *   [8-11]  int32  num_features
 *   [12-15] int32  accumulator_size (hidden, excludes PSQT)
 *   [16-19] int32  hidden1_size
 *   [20-23] int32  hidden2_size
 *   [24-27] int32  num_buckets
 *   [28-31] int32  bucket_divisor
 *   [32..]  float32 weights (see export_weights.py for layout)
 */
class NNUEModel {
public:
    int accumulator_size = 256;
    int hidden1_size = 32;
    int hidden2_size = 32;
    int num_buckets = 4;
    int bucket_divisor = 20;

    // acc_weight: transposed layout (NUM_FEATURES, accumulator_size+1) — last col is PSQT
    std::vector<float> acc_weight;
    std::vector<float> acc_bias;  // (accumulator_size+1,)
    // fc layers: bucket-folded (out*num_buckets, in) row-major
    std::vector<float> fc1_weight;
    std::vector<float> fc1_bias;
    std::vector<float> fc2_weight;
    std::vector<float> fc2_bias;
    std::vector<float> fc3_weight;
    std::vector<float> fc3_bias;

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
            throw std::runtime_error("Invalid NNUE magic number");
        if (version != NNUE_VERSION)
            throw std::runtime_error("Unsupported NNUE version: " + std::to_string(version));

        int32_t sizes[6];
        f.read(reinterpret_cast<char*>(sizes), 24);
        if (sizes[0] != NUM_FEATURES)
            throw std::runtime_error("Feature count mismatch");

        accumulator_size = sizes[1];
        hidden1_size = sizes[2];
        hidden2_size = sizes[3];
        num_buckets = sizes[4];
        bucket_divisor = sizes[5];

        const int acc_full = accumulator_size + 1;  // hidden + PSQT

        auto read_vec = [&](std::vector<float>& v, int n) {
            v.resize(n);
            f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
        };

        read_vec(acc_weight, NUM_FEATURES * acc_full);
        read_vec(acc_bias, acc_full);
        read_vec(fc1_weight, hidden1_size * num_buckets * accumulator_size * 2);
        read_vec(fc1_bias, hidden1_size * num_buckets);
        read_vec(fc2_weight, hidden2_size * num_buckets * hidden1_size);
        read_vec(fc2_bias, hidden2_size * num_buckets);
        read_vec(fc3_weight, num_buckets * hidden2_size);
        read_vec(fc3_bias, num_buckets);

        loaded = true;
    }

    /**
     * Count pieces from sparse feature arrays for bucket selection.
     * Features [0-80] = my pieces, [81-161] = opponent pieces.
     */
    int count_pieces(const int* stm_feats, int stm_n) const {
        int count = 0;
        for (int i = 0; i < stm_n; ++i) {
            if (stm_feats[i] < 162) ++count;  // cell features only
        }
        return count;
    }

    /**
     * Evaluate from sparse feature index arrays.
     * Returns raw eval (unbounded). Use sigmoid(val/scaling) for win probability.
     */
    float evaluate(const int* stm_feats, int stm_n,
                   const int* nstm_feats, int nstm_n) const {
        const int acc = accumulator_size;
        const int h1s = hidden1_size;
        const int h2s = hidden2_size;
        const int fc1_in = acc * 2;

        alignas(32) float raw_stm[512];
        alignas(32) float raw_nstm[512];
        alignas(32) float concat[1024];

        // Accumulate (acc_size+1 outputs, no activation yet)
        accumulate_raw(stm_feats, stm_n, raw_stm);
        accumulate_raw(nstm_feats, nstm_n, raw_nstm);

        // Extract PSQT (last element)
        float psqt = (raw_stm[acc] - raw_nstm[acc]) * 0.5f;

        // SCReLU on hidden part: clamp(x, 0, 1)²
        screlu(raw_stm, concat, acc);
        screlu(raw_nstm, concat + acc, acc);

        // Bucket selection
        int piece_count = count_pieces(stm_feats, stm_n);
        int bucket = std::min(piece_count / bucket_divisor, num_buckets - 1);

        // fc1: offset into bucket-folded weights
        const float* fc1_w = &fc1_weight[bucket * h1s * fc1_in];
        const float* fc1_b = &fc1_bias[bucket * h1s];
        linear_crelu(fc1_w, fc1_b, concat, h1_buf, h1s, fc1_in);

        // fc2
        const float* fc2_w = &fc2_weight[bucket * h2s * h1s];
        const float* fc2_b = &fc2_bias[bucket * h2s];
        linear_crelu(fc2_w, fc2_b, h1_buf, h2_buf, h2s, h1s);

        // fc3
        const float* fc3_w = &fc3_weight[bucket * h2s];
        float ls_out = dot_product(fc3_w, h2_buf, h2s) + fc3_bias[bucket];

        return ls_out + psqt;
    }

    /**
     * Evaluate a board position (extracts features internally).
     */
    float evaluate_board(const uttt::Board& board) const {
        int stm_feats[NUM_FEATURES], nstm_feats[NUM_FEATURES];
        int stm_n, nstm_n;
        extract_features(board, stm_feats, stm_n, nstm_feats, nstm_n);
        return evaluate(stm_feats, stm_n, nstm_feats, nstm_n);
    }

private:
    // Mutable scratch buffers (small, stack-like)
    alignas(32) mutable float h1_buf[128];
    alignas(32) mutable float h2_buf[128];

    /**
     * Raw accumulation (no activation). Output has acc_size+1 elements.
     */
    void accumulate_raw(const int* feats, int n, float* output) const {
        const int acc_full = accumulator_size + 1;
        const float* bias = acc_bias.data();
#ifdef __AVX2__
        for (int j = 0; j < acc_full; j += 8) {
            _mm256_store_ps(output + j, _mm256_loadu_ps(bias + j));
        }
        for (int f = 0; f < n; ++f) {
            const float* row = &acc_weight[feats[f] * acc_full];
            for (int j = 0; j < acc_full; j += 8) {
                __m256 a = _mm256_load_ps(output + j);
                __m256 b = _mm256_loadu_ps(row + j);
                _mm256_store_ps(output + j, _mm256_add_ps(a, b));
            }
        }
#else
        std::memcpy(output, bias, acc_full * sizeof(float));
        for (int f = 0; f < n; ++f) {
            const float* row = &acc_weight[feats[f] * acc_full];
            for (int j = 0; j < acc_full; ++j) output[j] += row[j];
        }
#endif
    }

    /**
     * SCReLU: clamp(x, 0, 1)² — applied to accumulator hidden outputs.
     */
    static void screlu(const float* input, float* output, int n) {
#ifdef __AVX2__
        __m256 zero = _mm256_setzero_ps();
        __m256 one = _mm256_set1_ps(1.0f);
        for (int j = 0; j < n; j += 8) {
            __m256 v = _mm256_loadu_ps(input + j);
            v = _mm256_max_ps(v, zero);
            v = _mm256_min_ps(v, one);
            v = _mm256_mul_ps(v, v);  // square
            _mm256_storeu_ps(output + j, v);
        }
#else
        for (int j = 0; j < n; ++j) {
            float v = std::min(std::max(input[j], 0.0f), 1.0f);
            output[j] = v * v;
        }
#endif
    }

    static void linear_crelu(const float* weight, const float* bias,
                             const float* input, float* output,
                             int out_size, int in_size) {
#ifdef __AVX2__
        for (int o = 0; o < out_size; ++o) {
            const float* row = &weight[o * in_size];
            __m256 sum = _mm256_setzero_ps();
            for (int i = 0; i < in_size; i += 8) {
                __m256 w = _mm256_loadu_ps(row + i);
                __m256 x = _mm256_loadu_ps(input + i);
                sum = _mm256_fmadd_ps(w, x, sum);
            }
            float s = hsum256(sum) + bias[o];
            output[o] = std::min(std::max(s, 0.0f), 1.0f);
        }
#else
        for (int o = 0; o < out_size; ++o) {
            float sum = bias[o];
            const float* row = &weight[o * in_size];
            for (int i = 0; i < in_size; ++i) sum += row[i] * input[i];
            output[o] = std::min(std::max(sum, 0.0f), 1.0f);
        }
#endif
    }

    static float dot_product(const float* a, const float* b, int n) {
#ifdef __AVX2__
        __m256 sum = _mm256_setzero_ps();
        for (int i = 0; i < n; i += 8) {
            sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum);
        }
        return hsum256(sum);
#else
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += a[i] * b[i];
        return s;
#endif
    }
};

} // namespace nnue
