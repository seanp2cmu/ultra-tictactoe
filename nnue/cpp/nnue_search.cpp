#include "nnue_search.hpp"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <cmath>

namespace nnue {

NNUESearchEngine::NNUESearchEngine(NNUEModel* model, int tt_size_mb)
    : model_(model), nodes_(0), tt_hits_(0), deadline_(0.0) {
    // Compute TT size: power of 2 entries
    int entry_size = sizeof(TTEntry);
    int num_entries = (tt_size_mb * 1024 * 1024) / entry_size;
    // Round down to power of 2
    int bits = 0;
    while ((1 << (bits + 1)) <= num_entries) ++bits;
    num_entries = 1 << bits;
    tt_mask_ = num_entries - 1;
    tt_.resize(num_entries);
    clear();
}

void NNUESearchEngine::clear() {
    std::memset(tt_.data(), 0, tt_.size() * sizeof(TTEntry));
    std::memset(history_, 0, sizeof(history_));
    for (int d = 0; d < MAX_DEPTH; ++d) {
        killers_[d][0] = {-1, -1};
        killers_[d][1] = {-1, -1};
    }
}

SearchResult NNUESearchEngine::search(uttt::Board& board, int max_depth, int time_limit_ms) {
    nodes_ = 0;
    tt_hits_ = 0;
    // Reset killers per search; keep history across depths
    for (int d = 0; d < MAX_DEPTH; ++d) {
        killers_[d][0] = {-1, -1};
        killers_[d][1] = {-1, -1};
    }

    double start = now_seconds();
    deadline_ = (time_limit_ms > 0) ? start + time_limit_ms / 1000.0 : 0.0;

    root_best_r_ = -1;
    root_best_c_ = -1;

    SearchResult result;
    result.best_r = -1;
    result.best_c = -1;
    result.score = 0.0f;

    float prev_score = 0.0f;
    constexpr float ASP_DELTA = 0.05f;

    for (int depth = 1; depth <= max_depth; ++depth) {
        float alpha, beta;

        // Aspiration windows: use narrow window from depth 4+
        if (depth >= 4) {
            alpha = prev_score - ASP_DELTA;
            beta  = prev_score + ASP_DELTA;
        } else {
            alpha = -SCORE_INF;
            beta  = SCORE_INF;
        }

        float score = negamax_root(board, depth, alpha, beta);

        // If aspiration window failed, re-search with full window
        if (depth >= 4 && (score <= alpha || score >= beta)) {
            score = negamax_root(board, depth, -SCORE_INF, SCORE_INF);
        }

        if (deadline_ > 0.0 && now_seconds() > deadline_)
            break;

        prev_score = score;
        result.depth = depth;
        result.score = score;
        result.best_r = root_best_r_;
        result.best_c = root_best_c_;
    }

    result.nodes = nodes_;
    result.tt_hits = tt_hits_;
    result.time_ms = (now_seconds() - start) * 1000.0;
    return result;
}

// ─── Root search ────────────────────────────────────────────────

float NNUESearchEngine::negamax_root(uttt::Board& board, int depth,
                                     float alpha, float beta) {
    auto moves = board.get_legal_moves();
    if (moves.empty()) return 0.0f;

    ScoredMove ordered[81];
    int n = order_moves(moves, ordered, -1, -1, false, 0, board);

    float best_score = -SCORE_INF;
    int best_r = ordered[0].r;
    int best_c = ordered[0].c;

    for (int i = 0; i < n; ++i) {
        int r = ordered[i].r, c = ordered[i].c;
        int sub_idx = (r / 3) * 3 + (c / 3);
        int prev_completed = board.completed_boards[sub_idx];
        int prev_winner = board.winner;
        int prev_last_r = board.last_move_r;
        int prev_last_c = board.last_move_c;
        bool prev_has_last = board.has_last_move;

        board.make_move(r, c);

        float score;
        if (i == 0) {
            // PVS: first move with full window
            score = -negamax(board, depth - 1, -beta, -alpha, 1);
        } else {
            // PVS: zero-window search
            score = -negamax(board, depth - 1, -(alpha + 0.0001f), -alpha, 1);
            if (score > alpha && score < beta) {
                // Re-search with full window
                score = -negamax(board, depth - 1, -beta, -alpha, 1);
            }
        }

        board.undo_move(r, c, prev_completed, prev_winner,
                        prev_last_r, prev_last_c, prev_has_last);

        if (score > best_score) {
            best_score = score;
            best_r = r;
            best_c = c;
        }
        alpha = std::max(alpha, score);
        if (alpha >= beta) break;
    }

    root_best_r_ = best_r;
    root_best_c_ = best_c;
    return best_score;
}

// ─── Recursive negamax ──────────────────────────────────────────

float NNUESearchEngine::negamax(uttt::Board& board, int depth,
                                float alpha, float beta, int ply) {
    ++nodes_;

    // Time check every 4096 nodes
    if (deadline_ > 0.0 && (nodes_ & 4095) == 0 && time_up())
        return 0.0f;

    // Terminal check
    if (board.winner != -1) {
        if (board.winner == 3) return 0.0f;
        if (board.winner == board.current_player)
            return 100.0f + depth * 0.01f;
        return -(100.0f + depth * 0.01f);
    }

    // Leaf evaluation: qsearch based on mode (0=off, 1=on, 2=auto)
    if (depth <= 0) {
        if (qsearch_mode_ == 1 || (qsearch_mode_ == 2 && should_qsearch(board)))
            return quiescence(board, alpha, beta, ply, 0);
        return model_->evaluate_board(board);
    }

    // TT probe
    uint64_t key = board_hash(board);
    int tt_idx = static_cast<int>(key & tt_mask_);
    TTEntry& tte = tt_[tt_idx];
    int tt_r = -1, tt_c = -1;
    bool has_tt_move = false;

    if (tte.key == key) {
        tt_r = tte.best_r;
        tt_c = tte.best_c;
        has_tt_move = (tt_r >= 0);
        if (tte.depth >= depth) {
            ++tt_hits_;
            if (tte.flag == TT_EXACT) return tte.score;
            if (tte.flag == TT_LOWER && tte.score >= beta) return tte.score;
            if (tte.flag == TT_UPPER && tte.score <= alpha) return tte.score;
        }
    }

    // Generate and order moves
    auto moves = board.get_legal_moves();
    if (moves.empty()) return 0.0f;

    ScoredMove ordered[81];
    int n = order_moves(moves, ordered, tt_r, tt_c, has_tt_move, ply, board);

    // Search with LMR + PVS
    float best_score = -SCORE_INF;
    int best_r = ordered[0].r, best_c = ordered[0].c;
    float orig_alpha = alpha;

    for (int i = 0; i < n; ++i) {
        int r = ordered[i].r, c = ordered[i].c;
        int sub_idx = (r / 3) * 3 + (c / 3);
        int prev_completed = board.completed_boards[sub_idx];
        int prev_winner = board.winner;
        int prev_last_r = board.last_move_r;
        int prev_last_c = board.last_move_c;
        bool prev_has_last = board.has_last_move;

        board.make_move(r, c);

        float score;
        int new_depth = depth - 1;

        if (i == 0) {
            // First move: full-depth full-window
            score = -negamax(board, new_depth, -beta, -alpha, ply + 1);
        } else {
            // LMR: reduce late non-tactical moves
            int R = 0;
            if (depth >= 3 && i >= 3 && ordered[i].tactical_val == 0) {
                R = 1;
                if (depth >= 5 && i >= 6) R = 2;
                // Don't reduce into negative depth
                if (new_depth - R < 0) R = new_depth;
            }

            // PVS zero-window + LMR reduced depth
            score = -negamax(board, new_depth - R, -(alpha + 0.0001f), -alpha, ply + 1);

            // LMR re-search at full depth if reduced search raised alpha
            if (R > 0 && score > alpha) {
                score = -negamax(board, new_depth, -(alpha + 0.0001f), -alpha, ply + 1);
            }

            // PVS re-search with full window if zero-window failed high
            if (score > alpha && score < beta) {
                score = -negamax(board, new_depth, -beta, -alpha, ply + 1);
            }
        }

        board.undo_move(r, c, prev_completed, prev_winner,
                        prev_last_r, prev_last_c, prev_has_last);

        if (score > best_score) {
            best_score = score;
            best_r = r;
            best_c = c;
        }

        alpha = std::max(alpha, score);
        if (alpha >= beta) {
            update_killers(ply, r, c);
            history_[r * 9 + c] += depth * depth;
            break;
        }
    }

    // TT store
    uint8_t flag;
    if (best_score <= orig_alpha)
        flag = TT_UPPER;
    else if (best_score >= beta)
        flag = TT_LOWER;
    else
        flag = TT_EXACT;

    tte.key = key;
    tte.score = best_score;
    tte.depth = static_cast<int8_t>(depth);
    tte.flag = flag;
    tte.best_r = static_cast<int8_t>(best_r);
    tte.best_c = static_cast<int8_t>(best_c);

    return best_score;
}

// ─── Quiescence search ──────────────────────────────────────────

float NNUESearchEngine::quiescence(uttt::Board& board, float alpha, float beta,
                                    int ply, int qs_depth) {
    ++nodes_;

    // Terminal check
    if (board.winner != -1) {
        if (board.winner == 3) return 0.0f;
        if (board.winner == board.current_player)
            return 100.0f;
        return -100.0f;
    }

    // Stand pat: static eval as a lower bound
    float stand_pat = model_->evaluate_board(board);

    // Beta cutoff: position is already too good
    if (stand_pat >= beta)
        return stand_pat;

    // Delta pruning: if stand_pat + margin can't reach alpha, give up
    if (stand_pat + QS_DELTA < alpha)
        return stand_pat;

    if (stand_pat > alpha)
        alpha = stand_pat;

    // Max qsearch depth reached
    if (qs_depth >= MAX_QS_DEPTH)
        return stand_pat;

    // Time check
    if (deadline_ > 0.0 && (nodes_ & 4095) == 0 && time_up())
        return stand_pat;

    // Only search sub-board WINNING moves (not blocks)
    // Blocks are defensive and don't swing eval enough to justify qsearch cost
    auto moves = board.get_legal_moves();

    float best_score = stand_pat;

    for (auto& [r, c] : moves) {
        if (!is_winning_move(r, c, board))
            continue;

        int sub_idx = (r / 3) * 3 + (c / 3);
        int prev_completed = board.completed_boards[sub_idx];
        int prev_winner = board.winner;
        int prev_last_r = board.last_move_r;
        int prev_last_c = board.last_move_c;
        bool prev_has_last = board.has_last_move;

        board.make_move(r, c);
        float score = -quiescence(board, -beta, -alpha, ply + 1, qs_depth + 1);
        board.undo_move(r, c, prev_completed, prev_winner,
                        prev_last_r, prev_last_c, prev_has_last);

        if (score > best_score)
            best_score = score;

        if (score > alpha)
            alpha = score;

        if (alpha >= beta)
            break;
    }

    return best_score;
}

bool NNUESearchEngine::is_tactical_move(int r, int c, const uttt::Board& board) const {
    int sub_idx = (r / 3) * 3 + (c / 3);
    int local_cell = (r % 3) * 3 + (c % 3);
    int cell_bit = 1 << local_cell;

    int player = board.current_player;
    uint16_t my_mask = (player == 1) ? board.x_masks[sub_idx] : board.o_masks[sub_idx];
    uint16_t opp_mask = (player == 1) ? board.o_masks[sub_idx] : board.x_masks[sub_idx];

    // Wins a sub-board?
    uint16_t new_my = my_mask | cell_bit;
    for (int w = 0; w < 8; ++w) {
        if ((new_my & WIN_MASKS[w]) == WIN_MASKS[w])
            return true;
    }

    // Blocks opponent from winning a sub-board?
    for (int w = 0; w < 8; ++w) {
        if ((opp_mask & WIN_MASKS[w]) == (WIN_MASKS[w] & ~cell_bit) &&
            (WIN_MASKS[w] & cell_bit))
            return true;
    }

    return false;
}

bool NNUESearchEngine::should_qsearch(const uttt::Board& board) const {
    // Count completed sub-boards as a quick game-stage proxy
    // Mid-game (2-6 completed) = complex, qsearch helps
    // Early (0-1) or late (7+) = simple, qsearch is overhead
    int completed = 0;
    for (int i = 0; i < 9; ++i)
        if (board.completed_boards[i] != 0) ++completed;
    return completed >= 2 && completed <= 6;
}

bool NNUESearchEngine::is_winning_move(int r, int c, const uttt::Board& board) const {
    int sub_idx = (r / 3) * 3 + (c / 3);
    int local_cell = (r % 3) * 3 + (c % 3);
    int cell_bit = 1 << local_cell;

    int player = board.current_player;
    uint16_t my_mask = (player == 1) ? board.x_masks[sub_idx] : board.o_masks[sub_idx];

    uint16_t new_my = my_mask | cell_bit;
    for (int w = 0; w < 8; ++w) {
        if ((new_my & WIN_MASKS[w]) == WIN_MASKS[w])
            return true;
    }
    return false;
}

// ─── Move ordering ──────────────────────────────────────────────

int NNUESearchEngine::order_moves(
    const std::vector<std::tuple<int,int>>& moves,
    ScoredMove* out,
    int tt_r, int tt_c, bool has_tt_move,
    int ply, const uttt::Board& board)
{
    auto k0 = (ply < MAX_DEPTH) ? killers_[ply][0] : std::pair<int,int>{-1,-1};
    auto k1 = (ply < MAX_DEPTH) ? killers_[ply][1] : std::pair<int,int>{-1,-1};

    int n = 0;

    for (auto& [r, c] : moves) {
        int s = history_[r * 9 + c];

        if (has_tt_move && r == tt_r && c == tt_c) {
            s += 1000000;  // TT move first
        } else if (r == k0.first && c == k0.second) {
            s += 500000;   // Killer 0
        } else if (r == k1.first && c == k1.second) {
            s += 400000;   // Killer 1
        }

        int tv = tactical_score(r, c, board);
        s += tv;
        out[n++] = {r, c, s, tv};
    }

    // Sort by score descending
    std::sort(out, out + n, [](const ScoredMove& a, const ScoredMove& b) {
        return a.score > b.score;
    });

    return n;
}

int NNUESearchEngine::tactical_score(int r, int c, const uttt::Board& board) const {
    int sub_idx = (r / 3) * 3 + (c / 3);
    int local_cell = (r % 3) * 3 + (c % 3);
    int cell_bit = 1 << local_cell;
    int score = 0;

    int player = board.current_player;
    uint16_t my_mask = (player == 1) ? board.x_masks[sub_idx] : board.o_masks[sub_idx];
    uint16_t opp_mask = (player == 1) ? board.o_masks[sub_idx] : board.x_masks[sub_idx];

    // Check if this move wins the sub-board
    uint16_t new_my = my_mask | cell_bit;
    bool wins_sub = false;
    for (int w = 0; w < 8; ++w) {
        if ((new_my & WIN_MASKS[w]) == WIN_MASKS[w]) {
            wins_sub = true;
            break;
        }
    }

    if (wins_sub) {
        score += 30;
        // Check meta-board threat
        uint16_t my_meta = 0;
        for (int i = 0; i < 9; ++i) {
            if (board.completed_boards[i] == player)
                my_meta |= (1 << i);
        }
        uint16_t new_meta = my_meta | (1 << sub_idx);
        for (int w = 0; w < 8; ++w) {
            uint16_t overlap = new_meta & WIN_MASKS[w];
            if (__builtin_popcount(overlap) == 2 && overlap != (my_meta & WIN_MASKS[w])) {
                score += 20;
                break;
            }
        }
    }

    // Check if this move blocks opponent from winning the sub-board
    for (int w = 0; w < 8; ++w) {
        if ((opp_mask & WIN_MASKS[w]) == (WIN_MASKS[w] & ~cell_bit) &&
            (WIN_MASKS[w] & cell_bit)) {
            score += 15;
            break;
        }
    }

    return score;
}

// ─── Killer moves ───────────────────────────────────────────────

void NNUESearchEngine::update_killers(int ply, int r, int c) {
    if (ply >= MAX_DEPTH) return;
    if (killers_[ply][0].first != r || killers_[ply][0].second != c) {
        killers_[ply][1] = killers_[ply][0];
        killers_[ply][0] = {r, c};
    }
}

// ─── Board hash (FNV-1a) ───────────────────────────────────────

uint64_t NNUESearchEngine::board_hash(const uttt::Board& board) const {
    uint64_t h = 14695981039346656037ULL;
    constexpr uint64_t prime = 1099511628211ULL;

    for (int i = 0; i < 9; ++i) {
        h ^= board.x_masks[i];
        h *= prime;
        h ^= board.o_masks[i];
        h *= prime;
    }
    h ^= board.current_player;
    h *= prime;
    if (board.has_last_move) {
        h ^= static_cast<uint64_t>(board.last_move_r * 9 + board.last_move_c + 1);
    }
    h *= prime;
    return h;
}

// ─── Time ───────────────────────────────────────────────────────

bool NNUESearchEngine::time_up() const {
    return now_seconds() > deadline_;
}

double NNUESearchEngine::now_seconds() const {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

} // namespace nnue
