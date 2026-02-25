#include "nnue_datagen.hpp"
#include <thread>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cstring>

namespace nnue {

// ─── DataGenerator ──────────────────────────────────────────────

DataGenerator::DataGenerator(NNUEModel* model, const DataGenConfig& config)
    : model_(model), config_(config) {}

std::vector<TrainingSample> DataGenerator::self_play_game(uint64_t seed) {
    std::mt19937 rng(seed);
    
    // Each game gets its own search engine (owns TT, killers, history)
    NNUESearchEngine engine(model_, config_.tt_size_mb);
    engine.set_qsearch(config_.qsearch_mode);
    
    uttt::Board board;
    
    // Collected positions: board array + search eval + side-to-move
    struct RawSample {
        int8_t board[92];
        float search_eval;
        int stm;
    };
    std::vector<RawSample> raw_samples;
    
    int ply = 0;
    
    while (board.winner == -1) {
        auto moves = board.get_legal_moves();
        if (moves.empty()) break;
        
        // Early termination: stop when few empty cells remain
        if (config_.early_stop_empty > 0 &&
            board.count_playable_empty_cells() <= config_.early_stop_empty)
            break;
        
        // Opening randomization: play random moves for diversity
        if (ply < config_.random_move_count) {
            std::uniform_int_distribution<int> dist(0, (int)moves.size() - 1);
            auto [r, c] = moves[dist(rng)];
            board.make_move(r, c);
            ++ply;
            continue;
        }
        
        // Search
        engine.clear();
        SearchResult result = engine.search(board, config_.search_depth);
        
        float eval = result.score;
        
        // Position filtering
        if (should_save(board, ply, eval, rng)) {
            RawSample s;
            board_to_array(board, s.board);
            s.search_eval = eval;
            s.stm = board.current_player;
            raw_samples.push_back(s);
        }
        
        // Play the best move
        if (result.best_r < 0) break;
        board.make_move(result.best_r, result.best_c);
        ++ply;
    }
    
    // Determine game result: +1 = P1 win, -1 = P2 win, 0 = draw
    float game_result_p1 = 0.0f;
    if (board.winner == 1) game_result_p1 = 1.0f;
    else if (board.winner == 2) game_result_p1 = -1.0f;
    
    // Blend search score with game result in [-1, 1] space.
    // Raw search eval is unbounded, so convert to [-1,1] via tanh(eval/scaling)
    // before blending with game result (which is already -1/0/+1).
    constexpr float EVAL_SCALING = 2.5f;
    float lam = config_.lambda_search;
    std::vector<TrainingSample> samples;
    samples.reserve(raw_samples.size());
    
    for (auto& rs : raw_samples) {
        TrainingSample ts;
        std::memcpy(ts.board, rs.board, 92);
        
        float eval_normalized = std::tanh(rs.search_eval / EVAL_SCALING);
        float game_result_stm = (rs.stm == 1) ? game_result_p1 : -game_result_p1;
        ts.value = lam * eval_normalized + (1.0f - lam) * game_result_stm;
        samples.push_back(ts);
    }
    
    return samples;
}

std::vector<TrainingSample> DataGenerator::generate(int num_games, uint64_t base_seed) {
    std::vector<TrainingSample> all_samples;
    
    for (int i = 0; i < num_games; ++i) {
        auto game_samples = self_play_game(base_seed + i);
        all_samples.insert(all_samples.end(), game_samples.begin(), game_samples.end());
    }
    
    return all_samples;
}

// ─── Board conversion ───────────────────────────────────────────

void DataGenerator::board_to_array(const uttt::Board& board, int8_t* out) const {
    // 81 cells
    for (int sub = 0; sub < 9; ++sub) {
        int base_r = (sub / 3) * 3;
        int base_c = (sub % 3) * 3;
        uint16_t x = board.x_masks[sub];
        uint16_t o = board.o_masks[sub];
        
        for (int cell = 0; cell < 9; ++cell) {
            int r = base_r + cell / 3;
            int c = base_c + cell % 3;
            int idx = r * 9 + c;
            
            if (x & (1 << cell))      out[idx] = 1;
            else if (o & (1 << cell)) out[idx] = 2;
            else                      out[idx] = 0;
        }
    }
    
    // 9 meta-board states
    for (int i = 0; i < 9; ++i) {
        out[81 + i] = static_cast<int8_t>(board.completed_boards[i]);
    }
    
    // Active sub-board
    if (board.has_last_move) {
        int target_sub = (board.last_move_r % 3) * 3 + (board.last_move_c % 3);
        out[90] = (board.completed_boards[target_sub] == 0) ? target_sub : -1;
    } else {
        out[90] = -1;
    }
    
    // Current player
    out[91] = static_cast<int8_t>(board.current_player);
}

// ─── Position filtering ─────────────────────────────────────────

bool DataGenerator::should_save(const uttt::Board& board, int ply,
                                 float eval, std::mt19937& rng) const {
    // Ply range
    if (ply < config_.write_minply || ply > config_.write_maxply)
        return false;
    
    // Eval limit (skip decisive positions)
    if (eval > config_.eval_limit || eval < -config_.eval_limit)
        return false;
    
    // Random skip for diversity
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (dist(rng) < config_.random_skip_rate)
        return false;
    
    // Quiet position filter (early/mid game only)
    if (config_.skip_noisy && ply <= config_.skip_noisy_maxply) {
        if (!is_quiet(board))
            return false;
    }
    
    return true;
}

bool DataGenerator::is_quiet(const uttt::Board& board) const {
    auto moves = board.get_legal_moves();
    if (moves.empty()) return true;
    
    int player = board.current_player;
    
    // Check each active sub-board for 2-in-a-row threats
    uint16_t checked = 0;  // bitmask of checked sub-boards
    
    for (auto& [r, c] : moves) {
        int sub = (r / 3) * 3 + (c / 3);
        if (checked & (1 << sub)) continue;
        checked |= (1 << sub);
        
        uint16_t my_mask = (player == 1) ? board.x_masks[sub] : board.o_masks[sub];
        uint16_t opp_mask = (player == 1) ? board.o_masks[sub] : board.x_masks[sub];
        uint16_t occupied = my_mask | opp_mask;
        
        for (int w = 0; w < 8; ++w) {
            uint16_t wm = WIN_MASKS[w];
            uint16_t empty_in_line = wm & ~occupied;
            
            // Exactly 1 empty cell in this win line
            if (__builtin_popcount(empty_in_line) != 1) continue;
            
            // 2 of mine + 1 empty = can win
            if (__builtin_popcount(my_mask & wm) == 2) return false;
            // 2 of opponent + 1 empty = must block
            if (__builtin_popcount(opp_mask & wm) == 2) return false;
        }
    }
    
    return true;
}

// ─── Parallel generation ────────────────────────────────────────

std::vector<TrainingSample> generate_parallel(
    NNUEModel* model,
    const DataGenConfig& config,
    int num_games,
    int num_threads,
    uint64_t base_seed)
{
    num_threads = std::max(1, std::min(num_threads, num_games));
    
    std::vector<std::vector<TrainingSample>> thread_results(num_threads);
    std::vector<std::thread> threads;
    
    // Distribute games across threads
    int games_per_thread = num_games / num_threads;
    int remainder = num_games % num_threads;
    
    int game_offset = 0;
    for (int t = 0; t < num_threads; ++t) {
        int n = games_per_thread + (t < remainder ? 1 : 0);
        uint64_t thread_seed = base_seed + game_offset;
        
        threads.emplace_back([model, &config, &thread_results, t, n, thread_seed]() {
            DataGenerator gen(model, config);
            thread_results[t] = gen.generate(n, thread_seed);
        });
        
        game_offset += n;
    }
    
    for (auto& th : threads) th.join();
    
    // Merge results
    size_t total = 0;
    for (auto& v : thread_results) total += v.size();
    
    std::vector<TrainingSample> all_samples;
    all_samples.reserve(total);
    for (auto& v : thread_results) {
        all_samples.insert(all_samples.end(), v.begin(), v.end());
    }
    
    return all_samples;
}

// ─── Batch rescore ──────────────────────────────────────────────

static uttt::Board array_to_board(const int8_t* arr) {
    uttt::Board board;
    
    // Reconstruct cells from array
    for (int sub = 0; sub < 9; ++sub) {
        int base_r = (sub / 3) * 3;
        int base_c = (sub % 3) * 3;
        for (int cell = 0; cell < 9; ++cell) {
            int r = base_r + cell / 3;
            int c = base_c + cell % 3;
            int idx = r * 9 + c;
            int8_t v = arr[idx];
            if (v == 1) board.x_masks[sub] |= (1 << cell);
            else if (v == 2) board.o_masks[sub] |= (1 << cell);
        }
        // Recount sub_counts
        board.sub_counts[sub] = {
            __builtin_popcount(board.x_masks[sub]),
            __builtin_popcount(board.o_masks[sub])
        };
    }
    
    // Meta-board
    for (int i = 0; i < 9; ++i) {
        board.completed_boards[i] = arr[81 + i];
    }
    // Rebuild completed_mask
    board.completed_mask = 0;
    for (int i = 0; i < 9; ++i) {
        if (board.completed_boards[i] != 0)
            board.completed_mask |= (1 << i);
    }
    
    // Active sub-board constraint via last_move
    int active = arr[90];
    if (active >= 0) {
        board.has_last_move = true;
        board.last_move_r = active / 3;
        board.last_move_c = active % 3;
    } else {
        board.has_last_move = false;
        board.last_move_r = -1;
        board.last_move_c = -1;
    }
    
    board.current_player = arr[91];
    board.check_winner();
    
    return board;
}

std::vector<float> batch_rescore(
    NNUEModel* model,
    const int8_t* boards,
    int num_positions,
    int search_depth,
    int num_threads,
    int qsearch_mode,
    int tt_size_mb)
{
    std::vector<float> scores(num_positions, 0.0f);
    num_threads = std::max(1, std::min(num_threads, num_positions));
    
    std::vector<std::thread> threads;
    std::atomic<int> next_pos{0};
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            NNUESearchEngine engine(model, tt_size_mb);
            engine.set_qsearch(qsearch_mode);
            
            while (true) {
                int i = next_pos.fetch_add(1);
                if (i >= num_positions) break;
                
                const int8_t* arr = boards + i * 92;
                uttt::Board board = array_to_board(arr);
                
                // Terminal check
                if (board.winner != -1) {
                    int cp = arr[91];
                    if (board.winner == 3) {
                        scores[i] = 0.0f;
                    } else if (board.winner == cp) {
                        scores[i] = 100.0f;  // large positive = win
                    } else {
                        scores[i] = -100.0f; // large negative = loss
                    }
                    continue;
                }
                
                engine.clear();
                SearchResult result = engine.search(board, search_depth);
                scores[i] = result.score;
            }
        });
    }
    
    for (auto& th : threads) th.join();
    return scores;
}

} // namespace nnue
