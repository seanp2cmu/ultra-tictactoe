#pragma once

#include "nnue_model.hpp"
#include "nnue_features.hpp"
#include "game/cpp/board.hpp"
#include <vector>
#include <tuple>
#include <cstdint>
#include <utility>
#include <cstring>

namespace nnue {

// ─── D4 symmetry for 3×3 grid (applies to both sub-board and cell indices) ───
// 8 transformations: identity, rot90, rot180, rot270, flipH, flipV, diagMain, diagAnti
// Each maps index [0..8] (row-major 3×3) to another index.
constexpr int NUM_SYMMETRIES = 8;
constexpr int SYM_TABLE[8][9] = {
    {0,1,2, 3,4,5, 6,7,8},  // identity
    {6,3,0, 7,4,1, 8,5,2},  // rot90:  (r,c)→(c, 2-r)
    {8,7,6, 5,4,3, 2,1,0},  // rot180: (r,c)→(2-r, 2-c)
    {2,5,8, 1,4,7, 0,3,6},  // rot270: (r,c)→(2-c, r)
    {2,1,0, 5,4,3, 8,7,6},  // flipH:  (r,c)→(r, 2-c)
    {6,7,8, 3,4,5, 0,1,2},  // flipV:  (r,c)→(2-r, c)
    {0,3,6, 1,4,7, 2,5,8},  // diagMain: (r,c)→(c, r)
    {8,5,2, 7,4,1, 6,3,0},  // diagAnti: (r,c)→(2-c, 2-r)
};
// Inverse: INV_SYM[s][SYM_TABLE[s][i]] == i
constexpr int INV_SYM[8] = {0, 3, 2, 1, 4, 5, 6, 7};
// identity→identity, rot90→rot270, rot180→rot180, rot270→rot90,
// flipH→flipH, flipV→flipV, diagMain→diagMain, diagAnti→diagAnti

// Transform a 9×9 board coordinate (r,c) using symmetry s
// Both sub-board index and cell index get the same 3×3 transform
inline void sym_transform_rc(int r, int c, int s, int& out_r, int& out_c) {
    int sub_r = r / 3, sub_c = c / 3;
    int cell_r = r % 3, cell_c = c % 3;
    int new_sub = SYM_TABLE[s][sub_r * 3 + sub_c];
    int new_cell = SYM_TABLE[s][cell_r * 3 + cell_c];
    out_r = (new_sub / 3) * 3 + (new_cell / 3);
    out_c = (new_sub % 3) * 3 + (new_cell % 3);
}

constexpr int MAX_DEPTH = 64;
constexpr int MAX_QS_DEPTH = 4;    // Max quiescence plies beyond depth 0
constexpr float QS_DELTA = 0.5f;   // Delta pruning margin (raw eval scale)
constexpr float SCORE_INF = 1e9f;

// Transposition table entry types
constexpr uint8_t TT_EXACT = 0;
constexpr uint8_t TT_LOWER = 1;  // beta cutoff
constexpr uint8_t TT_UPPER = 2;  // fail low

struct TTEntry {
    uint64_t key;      // Hash for verification
    float score;       // Evaluation score
    int8_t depth;      // Search depth
    uint8_t flag;      // TT_EXACT / TT_LOWER / TT_UPPER
    int8_t best_r;     // Best move row (-1 if none)
    int8_t best_c;     // Best move col
};

// 3x3 win patterns (9-bit masks)
constexpr uint16_t WIN_MASKS[8] = {
    0b111000000,  // row 0
    0b000111000,  // row 1
    0b000000111,  // row 2
    0b100100100,  // col 0
    0b010010010,  // col 1
    0b001001001,  // col 2
    0b100010001,  // diag
    0b001010100,  // anti-diag
};

struct SearchResult {
    int best_r = -1;
    int best_c = -1;
    float score = 0.0f;
    int depth = 0;
    int nodes = 0;
    int tt_hits = 0;
    double time_ms = 0.0;
};

struct ScoredMove {
    int r, c;
    int score;         // total ordering score
    int tactical_val;  // tactical component (>0 → skip LMR)
};

class NNUESearchEngine {
public:
    NNUESearchEngine(NNUEModel* model, int tt_size_mb = 16);

    /**
     * Iterative deepening search.
     * @param board      Board to search
     * @param max_depth  Maximum depth
     * @param time_limit_ms  Time limit in ms (0 = no limit)
     */
    SearchResult search(uttt::Board& board, int max_depth = 8, int time_limit_ms = 0);

    void clear();
    // QSearch modes: 0=off, 1=on, 2=auto (mid-game only)
    void set_qsearch(int mode) { qsearch_mode_ = mode; }
    int get_qsearch() const { return qsearch_mode_; }

private:
    NNUEModel* model_;

    // Transposition table
    std::vector<TTEntry> tt_;
    int tt_mask_;

    // Statistics
    int nodes_;
    int tt_hits_;
    int sym_tt_hits_;  // TT hits via symmetry (subset of tt_hits_)

    // Killer moves: 2 per ply
    std::pair<int,int> killers_[MAX_DEPTH][2];

    // History heuristic
    int history_[81];

    // Root best move (set by negamax_root)
    int root_best_r_;
    int root_best_c_;

    // ─── Incremental accumulator stack ────────────────────────
    // Two accumulators per ply: white (X/player1 perspective) and black (O/player2)
    // Max plies: MAX_DEPTH + MAX_QS_DEPTH + margin
    static constexpr int MAX_PLY = MAX_DEPTH + MAX_QS_DEPTH + 2;
    static constexpr int ACC_I16_SIZE = 288;  // >= acc_padded_ (272), rounded up
    alignas(32) int16_t white_acc_[MAX_PLY][ACC_I16_SIZE];  // player 1 perspective
    alignas(32) int16_t black_acc_[MAX_PLY][ACC_I16_SIZE];  // player 2 perspective
    int piece_count_;  // total pieces on board (incremental)

    // Initialize accumulators at ply 0 from board state
    void init_accumulators(const uttt::Board& board);

    // Update accumulators for a move: copy parent ply → child ply, then add/sub changed features
    // Must be called BEFORE make_move. Returns new_completed_status for caller to use.
    void update_accumulators_before_move(int parent_ply, int child_ply,
                                         const uttt::Board& board,
                                         int move_r, int move_c);

    // Evaluate using pre-computed accumulators at given ply
    float evaluate_incremental(const uttt::Board& board, int ply) const;

    // Quiescence search mode: 0=off, 1=always, 2=auto
    int qsearch_mode_ = 0;

    // Quick mid-game complexity check for auto qsearch
    bool should_qsearch(const uttt::Board& board) const;

    // Time control
    double deadline_;  // in seconds (0 = no deadline)

    // Root search (with aspiration window support)
    float negamax_root(uttt::Board& board, int depth, float alpha, float beta);

    // Recursive negamax
    float negamax(uttt::Board& board, int depth, float alpha, float beta, int ply);

    // Quiescence search: only tactical moves beyond depth 0
    float quiescence(uttt::Board& board, float alpha, float beta, int ply, int qs_depth);

    // Check if a move is tactical (wins/blocks sub-board)
    bool is_tactical_move(int r, int c, const uttt::Board& board) const;

    // Check if a move wins a sub-board (for qsearch: more selective)
    bool is_winning_move(int r, int c, const uttt::Board& board) const;

    // Move ordering (stack-based: takes raw r/c arrays)
    int order_moves(const int* move_r, const int* move_c, int n_moves,
                    ScoredMove* out,
                    int tt_r, int tt_c, bool has_tt_move,
                    int ply, const uttt::Board& board);

    // Tactical heuristic for move ordering
    int tactical_score(int r, int c, const uttt::Board& board) const;

    // Killer update
    void update_killers(int ply, int r, int c);

    // Symmetry-aware TT: compute canonical hash (min of 8 symmetry hashes)
    // Returns the symmetry index that produced the canonical hash
    int canonical_sym(const uttt::Board& board, uint64_t& canon_key) const;

    // Time check
    bool time_up() const;
    double now_seconds() const;
};

} // namespace nnue
