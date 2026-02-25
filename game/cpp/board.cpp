#include "board.hpp"
#include <algorithm>

namespace uttt {

// Global Zobrist table
ZobristTable ZOBRIST;

void ZobristTable::init() {
    if (initialized) return;
    // Deterministic PRNG (splitmix64)
    uint64_t state = 0xBEEF1234DEADCAFEULL;
    auto next = [&]() -> uint64_t {
        state += 0x9e3779b97f4a7c15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    };
    for (int p = 0; p < 2; ++p)
        for (int s = 0; s < 9; ++s)
            for (int c = 0; c < 9; ++c)
                piece[p][s][c] = next();
    side = next();
    for (int i = 0; i < 10; ++i)
        constraint[i] = next();
    initialized = true;
}

Board::Board() 
    : current_player(1), winner(-1), completed_mask(0), x_meta(0), o_meta(0),
      last_move_r(-1), last_move_c(-1), has_last_move(false), zobrist_hash(0) {
    if (!ZOBRIST.initialized) ZOBRIST.init();
    x_masks.fill(0);
    o_masks.fill(0);
    completed_boards.fill(0);
    for (auto& sc : sub_counts) {
        sc = {0, 0};
    }
    // Initial hash: player 1 side + "any" constraint
    zobrist_hash = ZOBRIST.side ^ ZOBRIST.constraint[9];
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
    b.x_meta = x_meta;
    b.o_meta = o_meta;
    b.last_move_r = last_move_r;
    b.last_move_c = last_move_c;
    b.has_last_move = has_last_move;
    b.zobrist_hash = zobrist_hash;
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

int Board::get_legal_moves_fast(int* out_r, int* out_c) const {
    int n = 0;
    if (winner != -1) return 0;

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
        uint16_t empty = (~filled) & 0x1FF;
        while (empty) {
            int cell = __builtin_ctz(empty);
            empty &= empty - 1;
            out_r[n] = sub_r * 3 + (cell / 3);
            out_c[n] = sub_c * 3 + (cell % 3);
            ++n;
        }
    } else {
        for (int sub_idx = 0; sub_idx < 9; sub_idx++) {
            if (completed_boards[sub_idx] != 0) continue;
            int sub_r = sub_idx / 3;
            int sub_c = sub_idx % 3;
            uint16_t filled = x_masks[sub_idx] | o_masks[sub_idx];
            uint16_t empty = (~filled) & 0x1FF;
            while (empty) {
                int cell = __builtin_ctz(empty);
                empty &= empty - 1;
                out_r[n] = sub_r * 3 + (cell / 3);
                out_c[n] = sub_c * 3 + (cell % 3);
                ++n;
            }
        }
    }
    return n;
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
    
    // Remove old constraint from hash
    int old_ci = get_constraint_index();
    zobrist_hash ^= ZOBRIST.constraint[old_ci];
    
    if (current_player == 1) {
        x_masks[sub_idx] |= bit;
        sub_counts[sub_idx][0]++;
        zobrist_hash ^= ZOBRIST.piece[0][sub_idx][cell_idx];
    } else {
        o_masks[sub_idx] |= bit;
        sub_counts[sub_idx][1]++;
        zobrist_hash ^= ZOBRIST.piece[1][sub_idx][cell_idx];
    }
    
    update_completed_boards(r, c);
    check_winner();
    
    last_move_r = r;
    last_move_c = c;
    has_last_move = true;
    current_player = 3 - current_player;
    
    // Toggle side and add new constraint
    zobrist_hash ^= ZOBRIST.side;
    int new_ci = get_constraint_index();
    zobrist_hash ^= ZOBRIST.constraint[new_ci];
}

void Board::undo_move(int r, int c, int prev_completed, int prev_winner,
                      int prev_last_r, int prev_last_c, bool prev_has_last) {
    // Remove current constraint
    int old_ci = get_constraint_index();
    zobrist_hash ^= ZOBRIST.constraint[old_ci];
    // Toggle side back
    zobrist_hash ^= ZOBRIST.side;
    
    current_player = 3 - current_player;
    
    int sub_idx = (r / 3) * 3 + (c / 3);
    int cell_idx = (r % 3) * 3 + (c % 3);
    uint16_t bit = 1 << cell_idx;
    
    if (current_player == 1) {
        x_masks[sub_idx] &= ~bit;
        sub_counts[sub_idx][0]--;
        zobrist_hash ^= ZOBRIST.piece[0][sub_idx][cell_idx];
    } else {
        o_masks[sub_idx] &= ~bit;
        sub_counts[sub_idx][1]--;
        zobrist_hash ^= ZOBRIST.piece[1][sub_idx][cell_idx];
    }
    
    completed_boards[sub_idx] = prev_completed;
    uint16_t sub_bit = 1 << sub_idx;
    if (prev_completed == 0) {
        completed_mask &= ~sub_bit;
        x_meta &= ~sub_bit;
        o_meta &= ~sub_bit;
    }
    
    winner = prev_winner;
    last_move_r = prev_last_r;
    last_move_c = prev_last_c;
    has_last_move = prev_has_last;
    
    // Restore constraint
    int new_ci = get_constraint_index();
    zobrist_hash ^= ZOBRIST.constraint[new_ci];
}

void Board::update_completed_boards(int r, int c) {
    int board_r = r / 3, board_c = c / 3;
    int sub_idx = board_r * 3 + board_c;
    
    if (completed_boards[sub_idx] != 0) return;
    
    uint16_t p_mask = (current_player == 1) ? x_masks[sub_idx] : o_masks[sub_idx];
    uint16_t filled = x_masks[sub_idx] | o_masks[sub_idx];
    uint16_t sub_bit = 1 << sub_idx;
    
    for (auto mask : WIN_MASKS) {
        if ((p_mask & mask) == mask) {
            completed_boards[sub_idx] = current_player;
            completed_mask |= sub_bit;
            if (current_player == 1) x_meta |= sub_bit;
            else                     o_meta |= sub_bit;
            return;
        }
    }
    
    if (filled == 0b111111111) {
        completed_boards[sub_idx] = 3;  // Draw
        completed_mask |= sub_bit;
    }
}

void Board::check_winner() {
    uint16_t p_mask = (current_player == 1) ? x_meta : o_meta;
    
    for (auto mask : WIN_MASKS) {
        if ((p_mask & mask) == mask) {
            winner = current_player;
            return;
        }
    }
    
    if (completed_mask == 0x1FF) {
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

int Board::get_constraint_index() const {
    if (!has_last_move) return 9;  // "any"
    int target_r = last_move_r % 3;
    int target_c = last_move_c % 3;
    int target_sub = target_r * 3 + target_c;
    if (completed_boards[target_sub] != 0) return 9;  // "any"
    return target_sub;
}

void Board::recompute_zobrist() {
    if (!ZOBRIST.initialized) ZOBRIST.init();
    zobrist_hash = 0;
    // Side to move
    if (current_player == 1) zobrist_hash ^= ZOBRIST.side;
    // Pieces
    for (int sub = 0; sub < 9; ++sub) {
        uint16_t x = x_masks[sub];
        while (x) {
            int cell = __builtin_ctz(x);
            x &= x - 1;
            zobrist_hash ^= ZOBRIST.piece[0][sub][cell];
        }
        uint16_t o = o_masks[sub];
        while (o) {
            int cell = __builtin_ctz(o);
            o &= o - 1;
            zobrist_hash ^= ZOBRIST.piece[1][sub][cell];
        }
    }
    // Constraint
    zobrist_hash ^= ZOBRIST.constraint[get_constraint_index()];
}

} // namespace uttt
