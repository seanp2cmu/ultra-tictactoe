# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

cimport cython
from libc.string cimport memcpy, memset

# Win patterns as C array
cdef unsigned short WIN_MASKS[8]
WIN_MASKS[0] = 0b111000000  # row 0
WIN_MASKS[1] = 0b000111000  # row 1
WIN_MASKS[2] = 0b000000111  # row 2
WIN_MASKS[3] = 0b100100100  # col 0
WIN_MASKS[4] = 0b010010010  # col 1
WIN_MASKS[5] = 0b001001001  # col 2
WIN_MASKS[6] = 0b100010001  # diag
WIN_MASKS[7] = 0b001010100  # anti-diag


cdef class BoardCy:
    """Cython-optimized Board class using bitmasks."""
    
    # C-level attributes (fast access)
    cdef public unsigned short x_masks[9]      # X pieces per sub-board (9-bit each)
    cdef public unsigned short o_masks[9]      # O pieces per sub-board (9-bit each)
    cdef public unsigned short completed_mask  # which sub-boards are completed
    cdef public unsigned char completed_boards[9]  # 0=open, 1=X, 2=O, 3=draw
    cdef public unsigned char sub_counts[18]   # [x_count, o_count] * 9 flattened
    cdef public unsigned char current_player   # 1 or 2
    cdef public signed char winner             # -1=none, 1=X, 2=O, 3=draw
    cdef public signed char last_move_r        # -1 if no move
    cdef public signed char last_move_c
    cdef public signed char constraint         # -1=any, 0-8=sub-board index
    
    def __init__(self):
        cdef int i
        for i in range(9):
            self.x_masks[i] = 0
            self.o_masks[i] = 0
            self.completed_boards[i] = 0
            self.sub_counts[i*2] = 0      # x_count
            self.sub_counts[i*2 + 1] = 0  # o_count
        self.completed_mask = 0
        self.current_player = 1
        self.winner = -1
        self.last_move_r = -1
        self.last_move_c = -1
        self.constraint = -1
    
    @property
    def last_move(self):
        if self.last_move_r < 0:
            return None
        return (self.last_move_r, self.last_move_c)
    
    @last_move.setter
    def last_move(self, value):
        if value is None:
            self.last_move_r = -1
            self.last_move_c = -1
        else:
            self.last_move_r = value[0]
            self.last_move_c = value[1]
    
    cpdef BoardCy clone(self):
        """Fast board cloning using memcpy."""
        cdef BoardCy new_board = BoardCy.__new__(BoardCy)
        memcpy(new_board.x_masks, self.x_masks, 9 * sizeof(unsigned short))
        memcpy(new_board.o_masks, self.o_masks, 9 * sizeof(unsigned short))
        memcpy(new_board.completed_boards, self.completed_boards, 9)
        memcpy(new_board.sub_counts, self.sub_counts, 18)
        new_board.completed_mask = self.completed_mask
        new_board.current_player = self.current_player
        new_board.winner = self.winner
        new_board.last_move_r = self.last_move_r
        new_board.last_move_c = self.last_move_c
        new_board.constraint = self.constraint
        return new_board
    
    cdef inline int _get_cell_fast(self, int r, int c) noexcept nogil:
        """Fast cell access without GIL."""
        cdef int sub_idx = (r // 3) * 3 + (c // 3)
        cdef int cell_bit = 1 << ((r % 3) * 3 + (c % 3))
        if self.x_masks[sub_idx] & cell_bit:
            return 1
        if self.o_masks[sub_idx] & cell_bit:
            return 2
        return 0
    
    def get_cell(self, int r, int c):
        """Get cell value from bitmasks. Returns 0, 1, or 2."""
        return self._get_cell_fast(r, c)
    
    def get_sub_board(self, int sub_idx):
        """Get sub-board as flat list of 9 values."""
        cdef list result = [0] * 9
        cdef unsigned short x_mask = self.x_masks[sub_idx]
        cdef unsigned short o_mask = self.o_masks[sub_idx]
        cdef int i
        for i in range(9):
            if x_mask & (1 << i):
                result[i] = 1
            elif o_mask & (1 << i):
                result[i] = 2
        return result
    
    def to_array(self):
        """Convert bitmasks to 9x9 array."""
        cdef list arr = [[0] * 9 for _ in range(9)]
        cdef int sub_idx, cell_idx, sub_r, sub_c, r, c
        cdef unsigned short x_mask, o_mask
        for sub_idx in range(9):
            sub_r = sub_idx // 3
            sub_c = sub_idx % 3
            x_mask = self.x_masks[sub_idx]
            o_mask = self.o_masks[sub_idx]
            for cell_idx in range(9):
                r = sub_r * 3 + cell_idx // 3
                c = sub_c * 3 + cell_idx % 3
                if x_mask & (1 << cell_idx):
                    arr[r][c] = 1
                elif o_mask & (1 << cell_idx):
                    arr[r][c] = 2
        return arr
    
    cpdef list get_legal_moves(self):
        """Get list of legal moves as (r, c) tuples."""
        cdef list moves = []
        cdef int sub_idx, cell_idx, sub_r, sub_c, r, c
        cdef int target_sub_idx
        cdef unsigned short occupied
        
        if self.last_move_r < 0:
            # No constraint - any open sub-board
            for sub_idx in range(9):
                if self.completed_mask & (1 << sub_idx):
                    continue
                occupied = self.x_masks[sub_idx] | self.o_masks[sub_idx]
                sub_r = sub_idx // 3
                sub_c = sub_idx % 3
                for cell_idx in range(9):
                    if not (occupied & (1 << cell_idx)):
                        r = sub_r * 3 + cell_idx // 3
                        c = sub_c * 3 + cell_idx % 3
                        moves.append((r, c))
        else:
            target_sub_idx = (self.last_move_r % 3) * 3 + (self.last_move_c % 3)
            
            if not (self.completed_mask & (1 << target_sub_idx)):
                # Constrained to target sub-board
                occupied = self.x_masks[target_sub_idx] | self.o_masks[target_sub_idx]
                sub_r = target_sub_idx // 3
                sub_c = target_sub_idx % 3
                for cell_idx in range(9):
                    if not (occupied & (1 << cell_idx)):
                        r = sub_r * 3 + cell_idx // 3
                        c = sub_c * 3 + cell_idx % 3
                        moves.append((r, c))
            else:
                # Target completed - any open sub-board
                for sub_idx in range(9):
                    if self.completed_mask & (1 << sub_idx):
                        continue
                    occupied = self.x_masks[sub_idx] | self.o_masks[sub_idx]
                    sub_r = sub_idx // 3
                    sub_c = sub_idx % 3
                    for cell_idx in range(9):
                        if not (occupied & (1 << cell_idx)):
                            r = sub_r * 3 + cell_idx // 3
                            c = sub_c * 3 + cell_idx % 3
                            moves.append((r, c))
        return moves
    
    cdef inline bint _is_valid_move_fast(self, int r, int c) noexcept nogil:
        """Fast move validation without GIL."""
        cdef int board_r, board_c, sub_idx, cell_bit
        cdef int target_board_r, target_board_c
        
        if r < 0 or r >= 9 or c < 0 or c >= 9:
            return False
        
        sub_idx = (r // 3) * 3 + (c // 3)
        cell_bit = 1 << ((r % 3) * 3 + (c % 3))
        
        # Check if cell is occupied
        if (self.x_masks[sub_idx] | self.o_masks[sub_idx]) & cell_bit:
            return False
        
        # Check if sub-board is completed
        if self.completed_mask & (1 << sub_idx):
            return False
        
        if self.last_move_r < 0:
            return True
        
        target_board_r = self.last_move_r % 3
        target_board_c = self.last_move_c % 3
        
        # Check if target sub-board is completed
        if self.completed_boards[target_board_r * 3 + target_board_c] != 0:
            return True
        
        board_r = r // 3
        board_c = c // 3
        return board_r == target_board_r and board_c == target_board_c
    
    def _is_valid_move(self, int r, int c):
        return self._is_valid_move_fast(r, c)
    
    def set_cell(self, int r, int c, int player):
        """Set cell directly (for testing/setup)."""
        cdef int sub_idx = (r // 3) * 3 + (c // 3)
        cdef int cell_bit = 1 << ((r % 3) * 3 + (c % 3))
        cdef int old = self._get_cell_fast(r, c)
        
        # Clear old value
        if old == 1:
            self.x_masks[sub_idx] &= ~cell_bit
            self.sub_counts[sub_idx * 2] -= 1
        elif old == 2:
            self.o_masks[sub_idx] &= ~cell_bit
            self.sub_counts[sub_idx * 2 + 1] -= 1
        
        # Set new value
        if player == 1:
            self.x_masks[sub_idx] |= cell_bit
            self.sub_counts[sub_idx * 2] += 1
        elif player == 2:
            self.o_masks[sub_idx] |= cell_bit
            self.sub_counts[sub_idx * 2 + 1] += 1
    
    cpdef void make_move(self, int r, int c, bint validate=True) except *:
        """Make a move on the board."""
        cdef int sub_idx, cell_bit
        
        if validate and not self._is_valid_move_fast(r, c):
            raise ValueError("Illegal move")
        
        self.last_move_r = r
        self.last_move_c = c
        
        sub_idx = (r // 3) * 3 + (c // 3)
        cell_bit = 1 << ((r % 3) * 3 + (c % 3))
        
        # Update bitmasks and counts
        if self.current_player == 1:
            self.x_masks[sub_idx] |= cell_bit
            self.sub_counts[sub_idx * 2] += 1
        else:
            self.o_masks[sub_idx] |= cell_bit
            self.sub_counts[sub_idx * 2 + 1] += 1
        
        self._update_completed(r, c)
        self._check_winner()
        
        self.current_player = 3 - self.current_player
    
    cpdef void undo_move(self, int r, int c, int prev_completed, 
                         int prev_winner, tuple prev_last_move):
        """Undo a move."""
        cdef int sub_idx = (r // 3) * 3 + (c // 3)
        cdef int cell_bit = 1 << ((r % 3) * 3 + (c % 3))
        cdef int player = 3 - self.current_player  # Player who made the move
        cdef int board_r = r // 3
        cdef int board_c = c // 3
        
        # Clear bitmask
        if player == 1:
            self.x_masks[sub_idx] &= ~cell_bit
            self.sub_counts[sub_idx * 2] -= 1
        else:
            self.o_masks[sub_idx] &= ~cell_bit
            self.sub_counts[sub_idx * 2 + 1] -= 1
        
        self.completed_boards[board_r * 3 + board_c] = prev_completed
        if prev_completed == 0:
            self.completed_mask &= ~(1 << sub_idx)
        
        self.winner = prev_winner
        
        if prev_last_move is None:
            self.last_move_r = -1
            self.last_move_c = -1
        else:
            self.last_move_r = prev_last_move[0]
            self.last_move_c = prev_last_move[1]
        
        self.current_player = 3 - self.current_player
    
    cdef inline void _update_completed(self, int r, int c) noexcept:
        """Update completed boards after a move."""
        cdef int board_r = r // 3
        cdef int board_c = c // 3
        cdef int sub_idx = board_r * 3 + board_c
        cdef unsigned short p_mask, filled_mask
        cdef int i
        
        if self.current_player == 1:
            p_mask = self.x_masks[sub_idx]
        else:
            p_mask = self.o_masks[sub_idx]
        filled_mask = self.x_masks[sub_idx] | self.o_masks[sub_idx]
        
        # Check win patterns
        for i in range(8):
            if (p_mask & WIN_MASKS[i]) == WIN_MASKS[i]:
                self.completed_boards[sub_idx] = self.current_player
                self.completed_mask |= (1 << sub_idx)
                return
        
        # Check draw
        if filled_mask == 0b111111111:
            self.completed_boards[sub_idx] = 3
            self.completed_mask |= (1 << sub_idx)
    
    cdef inline void _check_winner(self) noexcept:
        """Check if current player wins the game."""
        cdef unsigned short p_mask = 0
        cdef unsigned short filled_mask = 0
        cdef int r, c, bit, i
        
        for r in range(3):
            for c in range(3):
                bit = 1 << (r * 3 + c)
                if self.completed_boards[r * 3 + c] == self.current_player:
                    p_mask |= bit
                if self.completed_boards[r * 3 + c] != 0:
                    filled_mask |= bit
        
        # Check win patterns
        for i in range(8):
            if (p_mask & WIN_MASKS[i]) == WIN_MASKS[i]:
                self.winner = self.current_player
                return
        
        # Check draw
        if filled_mask == 0b111111111:
            self.winner = 3
    
    def is_game_over(self):
        return self.winner >= 0
    
    cpdef int count_playable_empty_cells(self):
        """Count empty cells in non-completed sub-boards."""
        cdef int count = 0
        cdef int sub_idx
        cdef unsigned short occupied
        
        for sub_idx in range(9):
            if not (self.completed_mask & (1 << sub_idx)):
                occupied = self.x_masks[sub_idx] | self.o_masks[sub_idx]
                # popcount: count set bits
                count += 9 - self._popcount(occupied)
        return count
    
    cdef inline int _popcount(self, unsigned short x) noexcept nogil:
        """Count number of set bits."""
        cdef int count = 0
        while x:
            count += x & 1
            x >>= 1
        return count
    
    def swap_xo(self):
        """Swap X and O in place."""
        cdef unsigned short tmp[9]
        cdef unsigned char tmp_count
        cdef int i
        
        # Swap bitmasks
        memcpy(tmp, self.x_masks, 9 * sizeof(unsigned short))
        memcpy(self.x_masks, self.o_masks, 9 * sizeof(unsigned short))
        memcpy(self.o_masks, tmp, 9 * sizeof(unsigned short))
        
        # Swap sub_counts
        for i in range(9):
            tmp_count = self.sub_counts[i * 2]
            self.sub_counts[i * 2] = self.sub_counts[i * 2 + 1]
            self.sub_counts[i * 2 + 1] = tmp_count
        
        # Swap completed_boards (1 <-> 2)
        for i in range(9):
            if self.completed_boards[i] == 1:
                self.completed_boards[i] = 2
            elif self.completed_boards[i] == 2:
                self.completed_boards[i] = 1
        
        self.current_player = 3 - self.current_player
    
    # ========== Python compatibility layer ==========
    
    def get_completed_boards_2d(self):
        """Get completed_boards as 3x3 list (Python Board compatible)."""
        return [[self.completed_boards[r*3+c] for c in range(3)] for r in range(3)]
    
    def set_completed_boards_2d(self, value):
        """Set completed_boards from 3x3 list."""
        cdef int r, c
        for r in range(3):
            for c in range(3):
                self.completed_boards[r*3+c] = value[r][c]
                if value[r][c] != 0:
                    self.completed_mask |= (1 << (r*3+c))
                else:
                    self.completed_mask &= ~(1 << (r*3+c))
    
    def set_sub_count(self, int sub_idx, int x_count, int o_count):
        """Set sub_counts for a sub-board."""
        self.sub_counts[sub_idx * 2] = x_count
        self.sub_counts[sub_idx * 2 + 1] = o_count
    
    def get_sub_counts_2d(self):
        """Get sub_counts as list of [x,o] pairs (Python Board compatible)."""
        return [[self.sub_counts[i*2], self.sub_counts[i*2+1]] for i in range(9)]
    
    def get_completed_state(self, int sub_idx):
        """Get completed state for a sub-board (0-8)."""
        return self.completed_boards[sub_idx]
    
    def get_sub_count_pair(self, int sub_idx):
        """Get (x_count, o_count) for a sub-board."""
        return (self.sub_counts[sub_idx * 2], self.sub_counts[sub_idx * 2 + 1])
