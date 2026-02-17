#pragma once

#include <vector>
#include <tuple>
#include <cstdint>
#include <array>

namespace uttt {

class Board {
public:
    // Bitmask representation
    std::array<uint16_t, 9> x_masks;
    std::array<uint16_t, 9> o_masks;
    std::array<uint8_t, 9> completed_boards;  // 0=open, 1=X, 2=O, 3=draw
    std::array<std::array<int, 2>, 9> sub_counts;  // [x_count, o_count]
    
    int current_player;
    int winner;  // -1=none, 1=X, 2=O, 3=draw
    uint16_t completed_mask;
    int last_move_r;
    int last_move_c;
    bool has_last_move;
    
    // Win patterns for 3x3
    static constexpr uint16_t WIN_MASKS[8] = {
        0b000000111,  // Row 0
        0b000111000,  // Row 1
        0b111000000,  // Row 2
        0b001001001,  // Col 0
        0b010010010,  // Col 1
        0b100100100,  // Col 2
        0b100010001,  // Diag
        0b001010100   // Anti-diag
    };
    
    Board();
    Board clone() const;
    
    // Cell operations
    int get_cell(int r, int c) const;
    void set_cell(int r, int c, int player);
    
    // Move operations
    void make_move(int r, int c, bool validate = true);
    void undo_move(int r, int c, int prev_completed, int prev_winner, 
                   int prev_last_r, int prev_last_c, bool prev_has_last);
    bool _is_valid_move(int r, int c) const;
    
    // Legal moves
    std::vector<std::tuple<int, int>> get_legal_moves() const;
    
    // Sub-board operations
    std::vector<int> get_sub_board(int sub_idx) const;
    int get_completed_state(int sub_idx) const;
    std::pair<int, int> get_sub_count_pair(int sub_idx) const;
    
    // Game state
    bool is_game_over() const;
    int count_playable_empty_cells() const;
    void check_winner();
    void update_completed_boards(int r, int c);
    
    // Python compatibility
    std::tuple<int, int> get_last_move() const;
    void set_last_move(int r, int c);
    void set_last_move_none();
    std::vector<std::vector<int>> get_completed_boards_2d() const;
    void set_completed_boards_2d(const std::vector<std::vector<int>>& boards);
    std::vector<std::vector<int>> to_array() const;
    
private:
    static int popcount(uint16_t x);
};

} // namespace uttt
