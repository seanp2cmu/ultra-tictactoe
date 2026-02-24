#pragma once

#include "nnue_model.hpp"
#include "game/cpp/board.hpp"
#include <vector>
#include <tuple>
#include <cstdint>
#include <utility>

namespace nnue {

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

    // Killer moves: 2 per ply
    std::pair<int,int> killers_[MAX_DEPTH][2];

    // History heuristic
    int history_[81];

    // Root best move (set by negamax_root)
    int root_best_r_;
    int root_best_c_;

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

    // Move ordering
    int order_moves(const std::vector<std::tuple<int,int>>& moves,
                    ScoredMove* out,
                    int tt_r, int tt_c, bool has_tt_move,
                    int ply, const uttt::Board& board);

    // Tactical heuristic for move ordering
    int tactical_score(int r, int c, const uttt::Board& board) const;

    // Killer update
    void update_killers(int ply, int r, int c);

    // Board hash (FNV-1a style)
    uint64_t board_hash(const uttt::Board& board) const;

    // Time check
    bool time_up() const;
    double now_seconds() const;
};

} // namespace nnue
