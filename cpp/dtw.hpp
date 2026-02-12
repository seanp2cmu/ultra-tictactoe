#pragma once

#include "board.hpp"
#include <unordered_map>

namespace uttt {

struct DTWResult {
    int result;      // 1=win, -1=loss, 0=draw, -2=aborted
    int dtw;         // Distance to win/loss
    int best_move_r; // -1 if none
    int best_move_c;
    
    DTWResult() : result(-2), dtw(1000000), best_move_r(-1), best_move_c(-1) {}
    DTWResult(int r, int d, int mr, int mc) : result(r), dtw(d), best_move_r(mr), best_move_c(mc) {}
};

class DTWCalculator {
public:
    int endgame_threshold;
    int max_nodes;
    bool use_cache;
    
    // Statistics
    int total_searches;
    int total_nodes;
    int aborted_searches;
    int node_count;
    
    // Simple cache (hash -> result)
    std::unordered_map<uint64_t, DTWResult> cache;
    
    DTWCalculator(bool use_cache = true, int endgame_threshold = 15, int max_nodes = 10000000);
    
    bool is_endgame(const Board& board) const;
    DTWResult calculate_dtw(Board& board);
    void reset_stats();
    
private:
    DTWResult alpha_beta_search(Board& board, int depth, int alpha, int beta);
    uint64_t hash_board(const Board& board) const;
    static int get_move_priority(int r, int c);
};

} // namespace uttt
