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
constexpr uint32_t NNUE_VERSION = 1;

/**
 * NNUE model for inference.
 *
 * Architecture:
 *   Input (199 sparse) → Accumulator (199→acc_size, shared) → ClippedReLU
 *   Concat STM+NSTM → fc1 → CReLU → fc2 → CReLU → fc3 → tanh
 *
 * Weight file format (binary):
 *   [0-3]   uint32 magic (0x454E4E55)
 *   [4-7]   uint32 version (1)
 *   [8-11]  int32  num_features
 *   [12-15] int32  accumulator_size
 *   [16-19] int32  hidden1_size
 *   [20-23] int32  hidden2_size
 *   [24..]  float32 weights:
 *     acc_weight_T  [num_features * acc_size]  (transposed for cache-friendly sparse access)
 *     acc_bias      [acc_size]
 *     fc1_weight    [h1 * acc_size*2]
 *     fc1_bias      [h1]
 *     fc2_weight    [h2 * h1]
 *     fc2_bias      [h2]
 *     fc3_weight    [1 * h2]
 *     fc3_bias      [1]
 */
class NNUEModel {
public:
    int accumulator_size = 256;
    int hidden1_size = 32;
    int hidden2_size = 32;

    // acc_weight: transposed layout (NUM_FEATURES, accumulator_size)
    std::vector<float> acc_weight;
    std::vector<float> acc_bias;
    // fc layers: standard (out, in) row-major
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
            throw std::runtime_error("Unsupported NNUE version");

        int32_t sizes[4];
        f.read(reinterpret_cast<char*>(sizes), 16);
        if (sizes[0] != NUM_FEATURES)
            throw std::runtime_error("Feature count mismatch");

        accumulator_size = sizes[1];
        hidden1_size = sizes[2];
        hidden2_size = sizes[3];

        auto read_vec = [&](std::vector<float>& v, int n) {
            v.resize(n);
            f.read(reinterpret_cast<char*>(v.data()), n * sizeof(float));
        };

        read_vec(acc_weight, NUM_FEATURES * accumulator_size);
        read_vec(acc_bias, accumulator_size);
        read_vec(fc1_weight, hidden1_size * accumulator_size * 2);
        read_vec(fc1_bias, hidden1_size);
        read_vec(fc2_weight, hidden2_size * hidden1_size);
        read_vec(fc2_bias, hidden2_size);
        read_vec(fc3_weight, hidden2_size);  // 1 * hidden2_size
        read_vec(fc3_bias, 1);

        loaded = true;
    }

    /**
     * Evaluate from sparse feature index arrays.
     * Uses AVX2 SIMD when available, aligned stack buffers.
     */
    float evaluate(const int* stm_feats, int stm_n,
                   const int* nstm_feats, int nstm_n) const {
        alignas(32) float acc_stm[512];
        alignas(32) float acc_nstm[512];
        alignas(32) float concat[1024];
        alignas(32) float h1[128];
        alignas(32) float h2[128];

        accumulate(stm_feats, stm_n, acc_stm);
        accumulate(nstm_feats, nstm_n, acc_nstm);

        std::memcpy(concat, acc_stm, accumulator_size * sizeof(float));
        std::memcpy(concat + accumulator_size, acc_nstm, accumulator_size * sizeof(float));

        linear_crelu(fc1_weight.data(), fc1_bias.data(),
                     concat, h1, hidden1_size, accumulator_size * 2);
        linear_crelu(fc2_weight.data(), fc2_bias.data(),
                     h1, h2, hidden2_size, hidden1_size);

        float out = dot_product(fc3_weight.data(), h2, hidden2_size) + fc3_bias[0];
        return std::tanh(out);
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
    void accumulate(const int* feats, int n, float* output) const {
        const int acc = accumulator_size;
        const float* bias = acc_bias.data();
#ifdef __AVX2__
        // Copy bias with AVX2
        for (int j = 0; j < acc; j += 8) {
            _mm256_store_ps(output + j, _mm256_loadu_ps(bias + j));
        }
        // Add weight rows for active features
        for (int f = 0; f < n; ++f) {
            const float* row = &acc_weight[feats[f] * acc];
            for (int j = 0; j < acc; j += 8) {
                __m256 a = _mm256_load_ps(output + j);
                __m256 b = _mm256_loadu_ps(row + j);
                _mm256_store_ps(output + j, _mm256_add_ps(a, b));
            }
        }
        // ClippedReLU
        __m256 zero = _mm256_setzero_ps();
        __m256 one = _mm256_set1_ps(1.0f);
        for (int j = 0; j < acc; j += 8) {
            __m256 v = _mm256_load_ps(output + j);
            v = _mm256_max_ps(v, zero);
            v = _mm256_min_ps(v, one);
            _mm256_store_ps(output + j, v);
        }
#else
        std::memcpy(output, bias, acc * sizeof(float));
        for (int f = 0; f < n; ++f) {
            const float* row = &acc_weight[feats[f] * acc];
            for (int j = 0; j < acc; ++j) output[j] += row[j];
        }
        for (int j = 0; j < acc; ++j)
            output[j] = std::min(std::max(output[j], 0.0f), 1.0f);
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
