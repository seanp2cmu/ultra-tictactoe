#include "nnue_search.hpp"
#include <algorithm>
#include <cstring>
#include <chrono>
#include <cmath>

namespace nnue {

NNUESearchEngine::NNUESearchEngine(NNUEModel* model, int tt_size_mb)
    : model_(model), nodes_(0), tt_hits_(0), sym_tt_hits_(0), deadline_(0.0) {
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

// ─── Incremental accumulator ────────────────────────────────────

void NNUESearchEngine::init_accumulators(const uttt::Board& board) {
    // Extract features from both absolute perspectives
    // White (player 1) perspective: X = MY, O = OPP
    // Black (player 2) perspective: O = MY, X = OPP
    int w_feats[NUM_FEATURES], b_feats[NUM_FEATURES];
    int w_n = 0, b_n = 0;

    // Piece features
    for (int sub = 0; sub < 9; ++sub) {
        uint16_t x = board.x_masks[sub];
        uint16_t o = board.o_masks[sub];
        while (x) {
            int cell = __builtin_ctz(x);
            x &= x - 1;
            int gi = GLOBAL_IDX[sub][cell];
            w_feats[w_n++] = MY_PIECE + gi;     // X is "my" for white
            b_feats[b_n++] = OPP_PIECE + gi;    // X is "opp" for black
        }
        while (o) {
            int cell = __builtin_ctz(o);
            o &= o - 1;
            int gi = GLOBAL_IDX[sub][cell];
            w_feats[w_n++] = OPP_PIECE + gi;    // O is "opp" for white
            b_feats[b_n++] = MY_PIECE + gi;     // O is "my" for black
        }
    }

    // Sub-board status features
    for (int sub = 0; sub < 9; ++sub) {
        int status = board.completed_boards[sub];
        if (status == 0) continue;
        if (status == 3) {
            w_feats[w_n++] = SUB_DRAW + sub;
            b_feats[b_n++] = SUB_DRAW + sub;
        } else if (status == 1) {  // X won
            w_feats[w_n++] = MY_SUB_WON + sub;
            b_feats[b_n++] = OPP_SUB_WON + sub;
        } else {  // O won
            w_feats[w_n++] = OPP_SUB_WON + sub;
            b_feats[b_n++] = MY_SUB_WON + sub;
        }
    }

    // Active constraint features
    if (!board.has_last_move) {
        w_feats[w_n++] = ACTIVE_ANY;
        b_feats[b_n++] = ACTIVE_ANY;
    } else {
        int target_sub = (board.last_move_r % 3) * 3 + (board.last_move_c % 3);
        if (board.completed_boards[target_sub] != 0) {
            w_feats[w_n++] = ACTIVE_ANY;
            b_feats[b_n++] = ACTIVE_ANY;
        } else {
            w_feats[w_n++] = ACTIVE_SUB + target_sub;
            b_feats[b_n++] = ACTIVE_SUB + target_sub;
        }
    }

    // Accumulate from scratch at ply 0 (int16)
    model_->accumulate_raw_q(w_feats, w_n, white_acc_[0]);
    model_->accumulate_raw_q(b_feats, b_n, black_acc_[0]);

    // Count pieces
    piece_count_ = 0;
    for (int sub = 0; sub < 9; ++sub) {
        piece_count_ += __builtin_popcount(board.x_masks[sub]);
        piece_count_ += __builtin_popcount(board.o_masks[sub]);
    }
}

void NNUESearchEngine::update_accumulators_before_move(int parent_ply, int child_ply,
                                                        const uttt::Board& board,
                                                        int move_r, int move_c) {
    // Called BEFORE make_move: board is still in pre-move state
    const int acc_bytes = model_->get_acc_padded_size() * sizeof(int16_t);
    std::memcpy(white_acc_[child_ply], white_acc_[parent_ply], acc_bytes);
    std::memcpy(black_acc_[child_ply], black_acc_[parent_ply], acc_bytes);

    int sub_idx = (move_r / 3) * 3 + (move_c / 3);
    int cell_idx = (move_r % 3) * 3 + (move_c % 3);
    int gi = GLOBAL_IDX[sub_idx][cell_idx];
    int player = board.current_player;

    // 1. Add piece feature
    if (player == 1) {
        model_->acc_add_feature(white_acc_[child_ply], MY_PIECE + gi);
        model_->acc_add_feature(black_acc_[child_ply], OPP_PIECE + gi);
    } else {
        model_->acc_add_feature(white_acc_[child_ply], OPP_PIECE + gi);
        model_->acc_add_feature(black_acc_[child_ply], MY_PIECE + gi);
    }

    // 2. Check if this move completes the sub-board
    uint16_t my_mask = (player == 1) ? board.x_masks[sub_idx] : board.o_masks[sub_idx];
    uint16_t new_my = my_mask | (1 << cell_idx);
    uint16_t filled = board.x_masks[sub_idx] | board.o_masks[sub_idx] | (1 << cell_idx);

    int new_completed = 0;  // 0=no change
    if (board.completed_boards[sub_idx] == 0) {
        for (int w = 0; w < 8; ++w) {
            if ((new_my & WIN_MASKS[w]) == WIN_MASKS[w]) {
                new_completed = player;
                break;
            }
        }
        if (new_completed == 0 && filled == 0x1FF) {
            new_completed = 3;  // draw
        }
    }

    if (new_completed != 0) {
        if (new_completed == 3) {
            model_->acc_add_feature(white_acc_[child_ply], SUB_DRAW + sub_idx);
            model_->acc_add_feature(black_acc_[child_ply], SUB_DRAW + sub_idx);
        } else if (new_completed == 1) {
            model_->acc_add_feature(white_acc_[child_ply], MY_SUB_WON + sub_idx);
            model_->acc_add_feature(black_acc_[child_ply], OPP_SUB_WON + sub_idx);
        } else {
            model_->acc_add_feature(white_acc_[child_ply], OPP_SUB_WON + sub_idx);
            model_->acc_add_feature(black_acc_[child_ply], MY_SUB_WON + sub_idx);
        }
    }

    // 3. Constraint change: remove old, add new
    // Old constraint
    int old_ci = board.get_constraint_index();
    int old_feat = (old_ci == 9) ? ACTIVE_ANY : (ACTIVE_SUB + old_ci);

    // New constraint: target sub = cell within moved sub
    int new_target = cell_idx;  // same as (move_r%3)*3+(move_c%3) which maps to sub index
    // But new_target is a cell index within the sub, we need the sub index it points to
    new_target = (move_r % 3) * 3 + (move_c % 3);
    int new_target_status;
    if (new_target == sub_idx) {
        new_target_status = new_completed;  // may have just completed
    } else {
        new_target_status = board.completed_boards[new_target];
    }
    int new_ci = (new_target_status != 0) ? 9 : new_target;
    int new_feat = (new_ci == 9) ? ACTIVE_ANY : (ACTIVE_SUB + new_ci);

    if (old_feat != new_feat) {
        model_->acc_sub_feature(white_acc_[child_ply], old_feat);
        model_->acc_sub_feature(black_acc_[child_ply], old_feat);
        model_->acc_add_feature(white_acc_[child_ply], new_feat);
        model_->acc_add_feature(black_acc_[child_ply], new_feat);
    }

    ++piece_count_;
}

float NNUESearchEngine::evaluate_incremental(const uttt::Board& board, int ply) const {
    // Pick STM/NSTM based on current player
    const int16_t* stm_acc;
    const int16_t* nstm_acc;
    if (board.current_player == 1) {
        stm_acc = white_acc_[ply];
        nstm_acc = black_acc_[ply];
    } else {
        stm_acc = black_acc_[ply];
        nstm_acc = white_acc_[ply];
    }
    return model_->evaluate_from_acc(stm_acc, nstm_acc, piece_count_);
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

    // Initialize incremental accumulators from root position
    init_accumulators(board);

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
    int move_r[81], move_c[81];
    int n_moves = board.get_legal_moves_fast(move_r, move_c);
    if (n_moves == 0) return 0.0f;

    ScoredMove ordered[81];
    int n = order_moves(move_r, move_c, n_moves, ordered, -1, -1, false, 0, board);

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
        int saved_piece_count = piece_count_;

        update_accumulators_before_move(0, 1, board, r, c);
        board.make_move(r, c, false);

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
        piece_count_ = saved_piece_count;

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
        return evaluate_incremental(board, ply);
    }

    // TT probe (symmetry-aware: only when no constraint, i.e. opening/free positions)
    uint64_t canon_key;
    int sym, inv_sym;
    if (board.get_constraint_index() == 9) {  // ACTIVE_ANY → symmetry possible
        sym = canonical_sym(board, canon_key);
        inv_sym = INV_SYM[sym];
    } else {
        canon_key = board.zobrist_hash;
        sym = 0;
        inv_sym = 0;
    }

    int tt_idx = static_cast<int>(canon_key & tt_mask_);
    TTEntry& tte = tt_[tt_idx];
    int tt_r = -1, tt_c = -1;
    bool has_tt_move = false;

    if (tte.key == canon_key) {
        // TT move is stored in canonical space → inverse-transform to current space
        if (tte.best_r >= 0) {
            sym_transform_rc(tte.best_r, tte.best_c, inv_sym, tt_r, tt_c);
            has_tt_move = true;
        }
        if (tte.depth >= depth) {
            ++tt_hits_;
            if (sym > 0) ++sym_tt_hits_;
            if (tte.flag == TT_EXACT) return tte.score;
            if (tte.flag == TT_LOWER && tte.score >= beta) return tte.score;
            if (tte.flag == TT_UPPER && tte.score <= alpha) return tte.score;
        }
    }

    // Generate and order moves
    int move_r[81], move_c[81];
    int n_moves = board.get_legal_moves_fast(move_r, move_c);
    if (n_moves == 0) return 0.0f;

    ScoredMove ordered[81];
    int n = order_moves(move_r, move_c, n_moves, ordered, tt_r, tt_c, has_tt_move, ply, board);

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
        int saved_piece_count = piece_count_;

        update_accumulators_before_move(ply, ply + 1, board, r, c);
        board.make_move(r, c, false);

        float score;
        int new_depth = depth - 1;

        if (i == 0) {
            score = -negamax(board, new_depth, -beta, -alpha, ply + 1);
        } else {
            int R = 0;
            if (depth >= 3 && i >= 3 && ordered[i].tactical_val == 0) {
                R = 1;
                if (depth >= 5 && i >= 6) R = 2;
                if (new_depth - R < 0) R = new_depth;
            }

            score = -negamax(board, new_depth - R, -(alpha + 0.0001f), -alpha, ply + 1);

            if (R > 0 && score > alpha) {
                score = -negamax(board, new_depth, -(alpha + 0.0001f), -alpha, ply + 1);
            }

            if (score > alpha && score < beta) {
                score = -negamax(board, new_depth, -beta, -alpha, ply + 1);
            }
        }

        board.undo_move(r, c, prev_completed, prev_winner,
                        prev_last_r, prev_last_c, prev_has_last);
        piece_count_ = saved_piece_count;

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

    // TT store (canonical space: forward-transform best move)
    uint8_t flag;
    if (best_score <= orig_alpha)
        flag = TT_UPPER;
    else if (best_score >= beta)
        flag = TT_LOWER;
    else
        flag = TT_EXACT;

    int store_r = best_r, store_c = best_c;
    if (sym > 0 && best_r >= 0) {
        sym_transform_rc(best_r, best_c, sym, store_r, store_c);
    }

    tte.key = canon_key;
    tte.score = best_score;
    tte.depth = static_cast<int8_t>(depth);
    tte.flag = flag;
    tte.best_r = static_cast<int8_t>(store_r);
    tte.best_c = static_cast<int8_t>(store_c);

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
    float stand_pat = evaluate_incremental(board, ply);

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
    int move_r[81], move_c[81];
    int n_moves = board.get_legal_moves_fast(move_r, move_c);

    float best_score = stand_pat;

    for (int mi = 0; mi < n_moves; ++mi) {
        int r = move_r[mi], c = move_c[mi];
        if (!is_winning_move(r, c, board))
            continue;

        int sub_idx = (r / 3) * 3 + (c / 3);
        int prev_completed = board.completed_boards[sub_idx];
        int prev_winner = board.winner;
        int prev_last_r = board.last_move_r;
        int prev_last_c = board.last_move_c;
        bool prev_has_last = board.has_last_move;
        int saved_piece_count = piece_count_;

        update_accumulators_before_move(ply, ply + 1, board, r, c);
        board.make_move(r, c, false);
        float score = -quiescence(board, -beta, -alpha, ply + 1, qs_depth + 1);
        board.undo_move(r, c, prev_completed, prev_winner,
                        prev_last_r, prev_last_c, prev_has_last);
        piece_count_ = saved_piece_count;

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
    const int* move_r, const int* move_c, int n_moves,
    ScoredMove* out,
    int tt_r, int tt_c, bool has_tt_move,
    int ply, const uttt::Board& board)
{
    auto k0 = (ply < MAX_DEPTH) ? killers_[ply][0] : std::pair<int,int>{-1,-1};
    auto k1 = (ply < MAX_DEPTH) ? killers_[ply][1] : std::pair<int,int>{-1,-1};

    int n = 0;

    for (int i = 0; i < n_moves; ++i) {
        int r = move_r[i], c = move_c[i];
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

    // Insertion sort (faster than std::sort for small N, typical ~8 moves)
    for (int i = 1; i < n; ++i) {
        ScoredMove tmp = out[i];
        int j = i - 1;
        while (j >= 0 && out[j].score < tmp.score) {
            out[j + 1] = out[j];
            --j;
        }
        out[j + 1] = tmp;
    }

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


// ─── Symmetry-aware TT ──────────────────────────────────────────

int NNUESearchEngine::canonical_sym(const uttt::Board& board, uint64_t& canon_key) const {
    // Compute zobrist hash for all 8 D4 symmetries of the board.
    // Return the symmetry index whose hash is smallest (canonical).
    // This is done by recomputing hash from scratch for each symmetry
    // using the ZOBRIST table with transformed coordinates.
    //
    // For each symmetry s, we compute:
    //   hash_s = side_key ^ constraint_key_s ^ sum(piece_keys_s)
    // where piece_keys_s maps each (player, sub, cell) through SYM_TABLE[s].

    uint64_t best_hash = board.zobrist_hash;  // identity (s=0)
    int best_sym = 0;

    for (int s = 1; s < NUM_SYMMETRIES; ++s) {
        uint64_t h = 0;
        // Side to move
        if (board.current_player == 1) h ^= uttt::ZOBRIST.side;

        // Pieces: transform (sub, cell) → (new_sub, new_cell)
        for (int sub = 0; sub < 9; ++sub) {
            int new_sub = SYM_TABLE[s][sub];
            uint16_t x = board.x_masks[sub];
            while (x) {
                int cell = __builtin_ctz(x);
                x &= x - 1;
                int new_cell = SYM_TABLE[s][cell];
                h ^= uttt::ZOBRIST.piece[0][new_sub][new_cell];
            }
            uint16_t o = board.o_masks[sub];
            while (o) {
                int cell = __builtin_ctz(o);
                o &= o - 1;
                int new_cell = SYM_TABLE[s][cell];
                h ^= uttt::ZOBRIST.piece[1][new_sub][new_cell];
            }
        }

        // Constraint: transform the constraint sub-board index
        int ci = board.get_constraint_index();  // 0-8 or 9 (any)
        int new_ci = (ci < 9) ? SYM_TABLE[s][ci] : 9;
        h ^= uttt::ZOBRIST.constraint[new_ci];

        if (h < best_hash) {
            best_hash = h;
            best_sym = s;
        }
    }

    canon_key = best_hash;
    return best_sym;
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
