#pragma once

#include "game/cpp/board.hpp"

namespace nnue {

constexpr int NUM_FEATURES = 199;

// Feature offsets
constexpr int MY_PIECE = 0;       // +global_idx (0-80)
constexpr int OPP_PIECE = 81;     // +global_idx (0-80)
constexpr int MY_SUB_WON = 162;   // +sub_idx (0-8)
constexpr int OPP_SUB_WON = 171;  // +sub_idx (0-8)
constexpr int SUB_DRAW = 180;     // +sub_idx (0-8)
constexpr int ACTIVE_SUB = 189;   // +sub_idx (0-8)
constexpr int ACTIVE_ANY = 198;

// Precomputed: global_idx[sub_idx][cell_idx] = row * 9 + col
constexpr int GLOBAL_IDX[9][9] = {
    { 0,  1,  2,  9, 10, 11, 18, 19, 20},  // sub 0
    { 3,  4,  5, 12, 13, 14, 21, 22, 23},  // sub 1
    { 6,  7,  8, 15, 16, 17, 24, 25, 26},  // sub 2
    {27, 28, 29, 36, 37, 38, 45, 46, 47},  // sub 3
    {30, 31, 32, 39, 40, 41, 48, 49, 50},  // sub 4
    {33, 34, 35, 42, 43, 44, 51, 52, 53},  // sub 5
    {54, 55, 56, 63, 64, 65, 72, 73, 74},  // sub 6
    {57, 58, 59, 66, 67, 68, 75, 76, 77},  // sub 7
    {60, 61, 62, 69, 70, 71, 78, 79, 80},  // sub 8
};

/**
 * Extract perspective-based sparse features from board.
 * Uses bit iteration for speed (only visits occupied cells).
 *
 * @param board     Board to extract features from
 * @param stm_out   Output array for side-to-move features (must hold >=199 ints)
 * @param stm_n     Output: number of STM features
 * @param nstm_out  Output array for not-side-to-move features
 * @param nstm_n    Output: number of NSTM features
 */
inline void extract_features(const uttt::Board& board,
                             int* stm_out, int& stm_n,
                             int* nstm_out, int& nstm_n) {
    stm_n = 0;
    nstm_n = 0;
    int stm = board.current_player;

    // Cell features (iterate set bits only)
    for (int sub = 0; sub < 9; ++sub) {
        uint16_t x = board.x_masks[sub];
        uint16_t o = board.o_masks[sub];

        while (x) {
            int cell = __builtin_ctz(x);
            x &= x - 1;
            int gi = GLOBAL_IDX[sub][cell];
            if (stm == 1) {
                stm_out[stm_n++] = MY_PIECE + gi;
                nstm_out[nstm_n++] = OPP_PIECE + gi;
            } else {
                stm_out[stm_n++] = OPP_PIECE + gi;
                nstm_out[nstm_n++] = MY_PIECE + gi;
            }
        }

        while (o) {
            int cell = __builtin_ctz(o);
            o &= o - 1;
            int gi = GLOBAL_IDX[sub][cell];
            if (stm == 2) {
                stm_out[stm_n++] = MY_PIECE + gi;
                nstm_out[nstm_n++] = OPP_PIECE + gi;
            } else {
                stm_out[stm_n++] = OPP_PIECE + gi;
                nstm_out[nstm_n++] = MY_PIECE + gi;
            }
        }
    }

    // Sub-board status features
    for (int sub = 0; sub < 9; ++sub) {
        int status = board.completed_boards[sub];
        if (status == 0) continue;
        if (status == 3) {
            stm_out[stm_n++] = SUB_DRAW + sub;
            nstm_out[nstm_n++] = SUB_DRAW + sub;
        } else if (status == stm) {
            stm_out[stm_n++] = MY_SUB_WON + sub;
            nstm_out[nstm_n++] = OPP_SUB_WON + sub;
        } else {
            stm_out[stm_n++] = OPP_SUB_WON + sub;
            nstm_out[nstm_n++] = MY_SUB_WON + sub;
        }
    }

    // Active sub-board constraint
    if (!board.has_last_move) {
        stm_out[stm_n++] = ACTIVE_ANY;
        nstm_out[nstm_n++] = ACTIVE_ANY;
    } else {
        int target_sub = (board.last_move_r % 3) * 3 + (board.last_move_c % 3);
        if (board.completed_boards[target_sub] != 0) {
            stm_out[stm_n++] = ACTIVE_ANY;
            nstm_out[nstm_n++] = ACTIVE_ANY;
        } else {
            stm_out[stm_n++] = ACTIVE_SUB + target_sub;
            nstm_out[nstm_n++] = ACTIVE_SUB + target_sub;
        }
    }
}

} // namespace nnue
