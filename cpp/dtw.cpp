#include "dtw.hpp"
#include <algorithm>

namespace uttt {

DTWCalculator::DTWCalculator(bool use_cache_, int endgame_threshold_, int max_nodes_)
    : endgame_threshold(endgame_threshold_),
      max_nodes(max_nodes_),
      use_cache(use_cache_),
      total_searches(0),
      total_nodes(0),
      aborted_searches(0),
      node_count(0) {}

bool DTWCalculator::is_endgame(const Board& board) const {
    return board.count_playable_empty_cells() <= endgame_threshold;
}

uint64_t DTWCalculator::hash_board(const Board& board) const {
    uint64_t hash = 0;
    for (int i = 0; i < 9; i++) {
        hash ^= static_cast<uint64_t>(board.x_masks[i]) << (i * 4);
        hash ^= static_cast<uint64_t>(board.o_masks[i]) << (i * 4 + 36);
    }
    hash ^= static_cast<uint64_t>(board.current_player) << 63;
    return hash;
}

int DTWCalculator::get_move_priority(int r, int c) {
    int local_r = r % 3;
    int local_c = c % 3;
    if (local_r == 1 && local_c == 1) return 0;  // Center
    if ((local_r == 0 || local_r == 2) && (local_c == 0 || local_c == 2)) return 1;  // Corner
    return 2;  // Edge
}

DTWResult DTWCalculator::alpha_beta_search(Board& board, int depth, int alpha, int beta) {
    node_count++;
    if (node_count > max_nodes) {
        return DTWResult(-2, 1000000, -1, -1);  // Aborted
    }
    
    // Check terminal state
    if (board.winner != -1) {
        if (board.winner == board.current_player) {
            return DTWResult(1, 0, -1, -1);  // Win
        } else if (board.winner == 3) {
            return DTWResult(0, 0, -1, -1);  // Draw
        } else {
            return DTWResult(-1, 0, -1, -1);  // Loss
        }
    }
    
    // Get legal moves
    auto legal_moves = board.get_legal_moves();
    
    if (legal_moves.empty()) {
        return DTWResult(0, 0, -1, -1);  // Draw
    }
    
    // Sort by move priority
    std::sort(legal_moves.begin(), legal_moves.end(), 
        [](const auto& a, const auto& b) {
            return get_move_priority(std::get<0>(a), std::get<1>(a)) < 
                   get_move_priority(std::get<0>(b), std::get<1>(b));
        });
    
    int best_result = -2;
    int best_dtw = 1000000;
    int best_r = -1, best_c = -1;
    
    for (const auto& move : legal_moves) {
        int r = std::get<0>(move);
        int c = std::get<1>(move);
        
        // Clone and make move (pure C++, no Python!)
        Board next = board.clone();
        next.make_move(r, c, false);
        
        // Check cache
        DTWResult opponent;
        if (use_cache) {
            uint64_t h = hash_board(next);
            auto it = cache.find(h);
            if (it != cache.end()) {
                opponent = it->second;
            } else {
                opponent = alpha_beta_search(next, depth + 1, -beta, -alpha);
                if (opponent.result == -2) {
                    return DTWResult(-2, 1000000, -1, -1);
                }
                cache[h] = opponent;
            }
        } else {
            opponent = alpha_beta_search(next, depth + 1, -beta, -alpha);
            if (opponent.result == -2) {
                return DTWResult(-2, 1000000, -1, -1);
            }
        }
        
        int my_result = -opponent.result;
        int my_dtw = (opponent.dtw < 1000000) ? opponent.dtw + 1 : 1000000;
        
        // Update best
        if (my_result > best_result) {
            best_result = my_result;
            best_dtw = my_dtw;
            best_r = r;
            best_c = c;
            alpha = std::max(alpha, my_result);
        } else if (my_result == best_result) {
            if (my_result > 0 && my_dtw < best_dtw) {
                best_dtw = my_dtw;
                best_r = r;
                best_c = c;
            } else if (my_result < 0 && my_dtw > best_dtw) {
                best_dtw = my_dtw;
                best_r = r;
                best_c = c;
            } else if (my_result == 0 && my_dtw < best_dtw) {
                best_dtw = my_dtw;
                best_r = r;
                best_c = c;
            }
        }
        
        // Pruning
        if (alpha >= beta) {
            break;
        }
    }
    
    return DTWResult(best_result, best_dtw, best_r, best_c);
}

DTWResult DTWCalculator::calculate_dtw(Board& board) {
    // Check cache first
    if (use_cache) {
        uint64_t h = hash_board(board);
        auto it = cache.find(h);
        if (it != cache.end()) {
            return it->second;
        }
    }
    
    int empty = board.count_playable_empty_cells();
    if (empty > endgame_threshold) {
        return DTWResult();  // Not endgame
    }
    
    node_count = 0;
    DTWResult result = alpha_beta_search(board, 0, -2, 2);
    
    total_searches++;
    total_nodes += node_count;
    
    if (result.result == -2) {
        aborted_searches++;
        return DTWResult();
    }
    
    if (use_cache) {
        uint64_t h = hash_board(board);
        cache[h] = result;
    }
    
    return result;
}

void DTWCalculator::reset_stats() {
    total_searches = 0;
    total_nodes = 0;
    aborted_searches = 0;
}

} // namespace uttt
