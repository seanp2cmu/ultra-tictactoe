#pragma once

#include "nnue_model.hpp"
#include "nnue_search.hpp"
#include "game/cpp/board.hpp"
#include <vector>
#include <cstdint>
#include <random>

namespace nnue {

// ─── Configuration ──────────────────────────────────────────────

struct DataGenConfig {
    int search_depth = 8;
    int qsearch_mode = 2;           // 0=off, 1=on, 2=auto
    int tt_size_mb = 16;
    
    // Target blending: target = λ * search + (1-λ) * game_result
    float lambda_search = 0.75f;
    
    // Position filtering
    int write_minply = 4;
    int write_maxply = 60;
    float eval_limit = 5.0f;    // raw eval scale (~88% win prob at scaling=2.5)
    float random_skip_rate = 0.3f;
    bool skip_noisy = true;
    int skip_noisy_maxply = 30;
    
    // Opening randomization
    int random_move_count = 8;       // Play N random moves to diversify openings
    float random_move_temp = 1.5f;   // Temperature for random opening moves
    
    // Early termination
    int early_stop_empty = 15;       // Stop game when empty cells <= this (0=disabled)
};

// ─── Training sample ────────────────────────────────────────────

struct TrainingSample {
    int8_t board[92];    // 81 cells + 9 meta + active + current_player
    float value;         // Blended target
};

// ─── Data generator ─────────────────────────────────────────────

class DataGenerator {
public:
    DataGenerator(NNUEModel* model, const DataGenConfig& config = {});
    
    /**
     * Play one self-play game and collect training samples.
     * @param seed  RNG seed for this game
     * @return Vector of training samples from this game
     */
    std::vector<TrainingSample> self_play_game(uint64_t seed);
    
    /**
     * Generate multiple games (single-threaded).
     * @param num_games  Number of games to play
     * @param base_seed  Base RNG seed
     * @return All training samples
     */
    std::vector<TrainingSample> generate(int num_games, uint64_t base_seed = 42);

private:
    NNUEModel* model_;
    DataGenConfig config_;
    
    // Convert board state to flat (92,) array
    void board_to_array(const uttt::Board& board, int8_t* out) const;
    
    // Position filtering
    bool should_save(const uttt::Board& board, int ply,
                     float eval, std::mt19937& rng) const;
    bool is_quiet(const uttt::Board& board) const;
};

/**
 * Generate training data using multiple threads.
 * Each thread has its own engine instance.
 * 
 * @param model       Shared model (read-only during inference)
 * @param config      Data generation config
 * @param num_games   Total games to generate
 * @param num_threads Number of parallel threads
 * @param base_seed   Base RNG seed
 * @return All training samples
 */
std::vector<TrainingSample> generate_parallel(
    NNUEModel* model,
    const DataGenConfig& config,
    int num_games,
    int num_threads = 4,
    uint64_t base_seed = 42
);

/**
 * Batch rescore positions with NNUE deep search (multi-threaded).
 * Reconstructs boards from (N, 92) array, runs alpha-beta search on each.
 *
 * @param model       Shared NNUE model (read-only)
 * @param boards      Flat array of N*92 int8 board representations
 * @param num_positions Number of positions
 * @param search_depth Max search depth
 * @param num_threads  Number of parallel threads
 * @param qsearch_mode Quiescence search mode (0/1/2)
 * @param tt_size_mb   TT size per thread in MB
 * @return Vector of raw scores (one per position)
 */
std::vector<float> batch_rescore(
    NNUEModel* model,
    const int8_t* boards,
    int num_positions,
    int search_depth = 10,
    int num_threads = 16,
    int qsearch_mode = 2,
    int tt_size_mb = 32
);

} // namespace nnue
