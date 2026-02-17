#include "board.hpp"
#include <algorithm>

namespace uttt {

Board::Board() 
    : current_player(1), winner(-1), completed_mask(0),
      last_move_r(-1), last_move_c(-1), has_last_move(false) {
    x_masks.fill(0);
    o_masks.fill(0);
    completed_boards.fill(0);
    for (auto& sc : sub_counts) {
        sc = {0, 0};
    }
}

Board Board::clone() const {
    Board b;
    b.x_masks = x_masks;
    b.o_masks = o_masks;
    b.completed_boards = completed_boards;
    b.sub_counts = sub_counts;
    b.current_player = current_player;
    b.winner = winner;
    b.completed_mask = completed_mask;
    b.last_move_r = last_move_r;
    b.last_move_c = last_move_c;
    b.has_last_move = has_last_move;
    return b;
}

int Board::popcount(uint16_t x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

int Board::get_cell(int r, int c) const {
    int sub_idx = (r / 3) * 3 + (c / 3);
    int cell_idx = (r % 3) * 3 + (c % 3);
    uint16_t bit = 1 << cell_idx;
    
    if (x_masks[sub_idx] & bit) return 1;
    if (o_masks[sub_idx] & bit) return 2;
    return 0;
}

void Board::set_cell(int r, int c, int player) {
    int sub_idx = (r / 3) * 3 + (c / 3);
    int cell_idx = (r % 3) * 3 + (c % 3);
    uint16_t bit = 1 << cell_idx;
    
    // Clear both
    x_masks[sub_idx] &= ~bit;
    o_masks[sub_idx] &= ~bit;
    
    // Set new
    if (player == 1) {
        x_masks[sub_idx] |= bit;
        sub_counts[sub_idx][0]++;
    } else if (player == 2) {
        o_masks[sub_idx] |= bit;
        sub_counts[sub_idx][1]++;
    }
}

bool Board::_is_valid_move(int r, int c) const {
    if (r < 0 || r >= 9 || c < 0 || c >= 9) return false;
    if (get_cell(r, c) != 0) return false;
    
    int board_r = r / 3, board_c = c / 3;
    int sub_idx = board_r * 3 + board_c;
    if (completed_boards[sub_idx] != 0) return false;
    
    if (!has_last_move) return true;
    
    int target_r = last_move_r % 3;
    int target_c = last_move_c % 3;
    int target_sub = target_r * 3 + target_c;
    
    if (completed_boards[target_sub] != 0) return true;
    
    return board_r == target_r && board_c == target_c;
}

std::vector<std::tuple<int, int>> Board::get_legal_moves() const {
    std::vector<std::tuple<int, int>> moves;
    
    if (winner != -1) return moves;
    
    int constraint_sub = -1;
    if (has_last_move) {
        int target_r = last_move_r % 3;
        int target_c = last_move_c % 3;
        int target_sub = target_r * 3 + target_c;
        if (completed_boards[target_sub] == 0) {
            constraint_sub = target_sub;
        }
    }
    
    if (constraint_sub >= 0) {
        int sub_r = constraint_sub / 3;
        int sub_c = constraint_sub % 3;
        uint16_t filled = x_masks[constraint_sub] | o_masks[constraint_sub];
        for (int cell = 0; cell < 9; cell++) {
            if (!(filled & (1 << cell))) {
                int r = sub_r * 3 + (cell / 3);
                int c = sub_c * 3 + (cell % 3);
                moves.emplace_back(r, c);
            }
        }
    } else {
        for (int sub_idx = 0; sub_idx < 9; sub_idx++) {
            if (completed_boards[sub_idx] != 0) continue;
            int sub_r = sub_idx / 3;
            int sub_c = sub_idx % 3;
            uint16_t filled = x_masks[sub_idx] | o_masks[sub_idx];
            for (int cell = 0; cell < 9; cell++) {
                if (!(filled & (1 << cell))) {
                    int r = sub_r * 3 + (cell / 3);
                    int c = sub_c * 3 + (cell % 3);
                    moves.emplace_back(r, c);
                }
            }
        }
    }
    
    return moves;
}

void Board::make_move(int r, int c, bool validate) {
    if (validate && !_is_valid_move(r, c)) return;
    
    int sub_idx = (r / 3) * 3 + (c / 3);
    int cell_idx = (r % 3) * 3 + (c % 3);
    uint16_t bit = 1 << cell_idx;
    
    if (current_player == 1) {
        x_masks[sub_idx] |= bit;
        sub_counts[sub_idx][0]++;
    } else {
        o_masks[sub_idx] |= bit;
        sub_counts[sub_idx][1]++;
    }
    
    update_completed_boards(r, c);
    check_winner();
    
    last_move_r = r;
    last_move_c = c;
    has_last_move = true;
    current_player = 3 - current_player;
}

void Board::undo_move(int r, int c, int prev_completed, int prev_winner,
                      int prev_last_r, int prev_last_c, bool prev_has_last) {
    current_player = 3 - current_player;
    
    int sub_idx = (r / 3) * 3 + (c / 3);
    int cell_idx = (r % 3) * 3 + (c % 3);
    uint16_t bit = 1 << cell_idx;
    
    if (current_player == 1) {
        x_masks[sub_idx] &= ~bit;
        sub_counts[sub_idx][0]--;
    } else {
        o_masks[sub_idx] &= ~bit;
        sub_counts[sub_idx][1]--;
    }
    
    completed_boards[sub_idx] = prev_completed;
    if (prev_completed == 0) {
        completed_mask &= ~(1 << sub_idx);
    }
    
    winner = prev_winner;
    last_move_r = prev_last_r;
    last_move_c = prev_last_c;
    has_last_move = prev_has_last;
}

void Board::update_completed_boards(int r, int c) {
    int board_r = r / 3, board_c = c / 3;
    int sub_idx = board_r * 3 + board_c;
    
    if (completed_boards[sub_idx] != 0) return;
    
    uint16_t p_mask = (current_player == 1) ? x_masks[sub_idx] : o_masks[sub_idx];
    uint16_t filled = x_masks[sub_idx] | o_masks[sub_idx];
    
    for (auto mask : WIN_MASKS) {
        if ((p_mask & mask) == mask) {
            completed_boards[sub_idx] = current_player;
            completed_mask |= (1 << sub_idx);
            return;
        }
    }
    
    if (filled == 0b111111111) {
        completed_boards[sub_idx] = 3;  // Draw
        completed_mask |= (1 << sub_idx);
    }
}

void Board::check_winner() {
    uint16_t p_mask = 0, filled_mask = 0;
    
    for (int i = 0; i < 9; i++) {
        if (completed_boards[i] == current_player) {
            p_mask |= (1 << i);
        }
        if (completed_boards[i] != 0) {
            filled_mask |= (1 << i);
        }
    }
    
    for (auto mask : WIN_MASKS) {
        if ((p_mask & mask) == mask) {
            winner = current_player;
            return;
        }
    }
    
    if (filled_mask == 0b111111111) {
        winner = 3;  // Draw
    }
}

bool Board::is_game_over() const {
    return winner != -1;
}

int Board::count_playable_empty_cells() const {
    int count = 0;
    for (int sub_idx = 0; sub_idx < 9; sub_idx++) {
        if (completed_boards[sub_idx] != 0) continue;
        uint16_t filled = x_masks[sub_idx] | o_masks[sub_idx];
        count += 9 - popcount(filled);
    }
    return count;
}

std::vector<int> Board::get_sub_board(int sub_idx) const {
    std::vector<int> cells(9, 0);
    for (int i = 0; i < 9; i++) {
        uint16_t bit = 1 << i;
        if (x_masks[sub_idx] & bit) cells[i] = 1;
        else if (o_masks[sub_idx] & bit) cells[i] = 2;
    }
    return cells;
}

int Board::get_completed_state(int sub_idx) const {
    return completed_boards[sub_idx];
}

std::pair<int, int> Board::get_sub_count_pair(int sub_idx) const {
    return {sub_counts[sub_idx][0], sub_counts[sub_idx][1]};
}

std::tuple<int, int> Board::get_last_move() const {
    return std::make_tuple(last_move_r, last_move_c);
}

void Board::set_last_move(int r, int c) {
    last_move_r = r;
    last_move_c = c;
    has_last_move = true;
}

void Board::set_last_move_none() {
    has_last_move = false;
    last_move_r = -1;
    last_move_c = -1;
}

std::vector<std::vector<int>> Board::get_completed_boards_2d() const {
    std::vector<std::vector<int>> result(3, std::vector<int>(3));
    for (int i = 0; i < 9; i++) {
        result[i / 3][i % 3] = completed_boards[i];
    }
    return result;
}

void Board::set_completed_boards_2d(const std::vector<std::vector<int>>& boards) {
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            int idx = r * 3 + c;
            completed_boards[idx] = boards[r][c];
            if (boards[r][c] != 0) {
                completed_mask |= (1 << idx);
            }
        }
    }
}

std::vector<std::vector<int>> Board::to_array() const {
    std::vector<std::vector<int>> result(9, std::vector<int>(9, 0));
    for (int r = 0; r < 9; r++) {
        for (int c = 0; c < 9; c++) {
            result[r][c] = get_cell(r, c);
        }
    }
    return result;
}

} // namespace uttt
