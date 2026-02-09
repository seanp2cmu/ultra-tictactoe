import numpy as np

class Board:
    # Bitmask win patterns for fast check (position = r*3+c)
    WIN_MASKS = (
        0b111000000,  # row 0
        0b000111000,  # row 1
        0b000000111,  # row 2
        0b100100100,  # col 0
        0b010010010,  # col 1
        0b001001001,  # col 2
        0b100010001,  # diag
        0b001010100,  # anti-diag
    )
    
    def __init__(self):
        self.completed_boards = [[0 for _ in range(3)] for _ in range(3)] # empty: 0, player 1: 1, player 2: 2, draw: 3
        self.current_player = 1
        self.winner = None
        self.last_move = None
        self.sub_counts = [[0, 0] for _ in range(9)]  # [x_count, o_count] per sub-board
        # Bitmask representation: x_masks[sub_idx] and o_masks[sub_idx] are 9-bit masks
        self.x_masks = [0] * 9  # X pieces per sub-board
        self.o_masks = [0] * 9  # O pieces per sub-board
        self.completed_mask = 0  # 9-bit: which sub-boards are completed
    
    def clone(self):
        """
        Fast board cloning (80x faster than copy.deepcopy)
        Only copies mutable state, shares immutable data
        """
        new_board = Board.__new__(Board)
        new_board.completed_boards = [row[:] for row in self.completed_boards]
        new_board.current_player = self.current_player
        new_board.winner = self.winner
        new_board.last_move = self.last_move
        new_board.sub_counts = [c[:] for c in self.sub_counts]
        new_board.x_masks = self.x_masks[:]
        new_board.o_masks = self.o_masks[:]
        new_board.completed_mask = self.completed_mask
        return new_board
    
    def get_cell(self, r, c):
        """Get cell value from bitmasks. Returns 0, 1, or 2."""
        sub_idx = (r // 3) * 3 + (c // 3)
        cell_bit = 1 << ((r % 3) * 3 + (c % 3))
        if self.x_masks[sub_idx] & cell_bit:
            return 1
        if self.o_masks[sub_idx] & cell_bit:
            return 2
        return 0
    
    def get_sub_board(self, sub_idx):
        """Get sub-board as flat list of 9 values (for AI compatibility)."""
        result = [0] * 9
        x_mask = self.x_masks[sub_idx]
        o_mask = self.o_masks[sub_idx]
        for i in range(9):
            if x_mask & (1 << i):
                result[i] = 1
            elif o_mask & (1 << i):
                result[i] = 2
        return result
    
    def to_array(self):
        """Convert bitmasks to 9x9 array (for network input)."""
        arr = [[0] * 9 for _ in range(9)]
        for sub_idx in range(9):
            sub_r, sub_c = sub_idx // 3, sub_idx % 3
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

    def get_legal_moves(self):
        legal_moves = []
        
        if self.last_move is None:
            for sub_idx in range(9):
                if self.completed_mask & (1 << sub_idx):
                    continue
                occupied = self.x_masks[sub_idx] | self.o_masks[sub_idx]
                sub_r, sub_c = sub_idx // 3, sub_idx % 3
                for cell_idx in range(9):
                    if not (occupied & (1 << cell_idx)):
                        r = sub_r * 3 + cell_idx // 3
                        c = sub_c * 3 + cell_idx % 3
                        legal_moves.append((r, c))
        else:
            last_r, last_c = self.last_move
            target_sub_idx = (last_r % 3) * 3 + (last_c % 3)
            
            if not (self.completed_mask & (1 << target_sub_idx)):
                occupied = self.x_masks[target_sub_idx] | self.o_masks[target_sub_idx]
                sub_r, sub_c = target_sub_idx // 3, target_sub_idx % 3
                for cell_idx in range(9):
                    if not (occupied & (1 << cell_idx)):
                        r = sub_r * 3 + cell_idx // 3
                        c = sub_c * 3 + cell_idx % 3
                        legal_moves.append((r, c))
            else:
                for sub_idx in range(9):
                    if self.completed_mask & (1 << sub_idx):
                        continue
                    occupied = self.x_masks[sub_idx] | self.o_masks[sub_idx]
                    sub_r, sub_c = sub_idx // 3, sub_idx % 3
                    for cell_idx in range(9):
                        if not (occupied & (1 << cell_idx)):
                            r = sub_r * 3 + cell_idx // 3
                            c = sub_c * 3 + cell_idx % 3
                            legal_moves.append((r, c))
        return legal_moves
    
    def _is_valid_move(self, r, c) -> bool:
        """Fast validation without computing all legal moves. O(1) instead of O(81)."""
        if not (0 <= r < 9 and 0 <= c < 9): # loop bounds
            return False
        
        if self.get_cell(r, c) != 0: # check already placed
            return False
        
        board_r, board_c = r // 3, c // 3
        if self.completed_boards[board_r][board_c] != 0: # check completed
            return False
        
        if self.last_move is None:
            return True
        
        last_r, last_c = self.last_move
        target_board_r = last_r % 3
        target_board_c = last_c % 3
        
        if self.completed_boards[target_board_r][target_board_c] != 0:
            return True
        
        return board_r == target_board_r and board_c == target_board_c
    
    def set_cell(self, r, c, player):
        """Set cell directly (for testing/setup). Updates bitmasks."""
        sub_idx = (r // 3) * 3 + (c // 3)
        cell_bit = 1 << ((r % 3) * 3 + (c % 3))
        
        # Clear old value if any
        old = self.get_cell(r, c)
        if old == 1:
            self.x_masks[sub_idx] &= ~cell_bit
            self.sub_counts[sub_idx][0] -= 1
        elif old == 2:
            self.o_masks[sub_idx] &= ~cell_bit
            self.sub_counts[sub_idx][1] -= 1
        
        # Set new value
        if player == 1:
            self.x_masks[sub_idx] |= cell_bit
            self.sub_counts[sub_idx][0] += 1
        elif player == 2:
            self.o_masks[sub_idx] |= cell_bit
            self.sub_counts[sub_idx][1] += 1
    
    def make_move(self, r, c, validate=True):
        if validate and not self._is_valid_move(r, c):
            raise ValueError("Illegal move")
        
        self.last_move = (r, c)
        
        # Update sub_counts and bitmasks
        sub_idx = (r // 3) * 3 + (c // 3)
        cell_bit = 1 << ((r % 3) * 3 + (c % 3))
        self.sub_counts[sub_idx][self.current_player - 1] += 1
        if self.current_player == 1:
            self.x_masks[sub_idx] |= cell_bit
        else:
            self.o_masks[sub_idx] |= cell_bit
        
        self.update_completed_boards(r, c)
        self.check_winner()
        
        self.current_player = self.current_player % 2 + 1
    
    def undo_move(self, r, c, prev_completed, prev_winner, prev_last_move):
        """Undo a move (for solver optimization). Caller must save state before make_move."""
        sub_idx = (r // 3) * 3 + (c // 3)
        cell_bit = 1 << ((r % 3) * 3 + (c % 3))
        player = self.current_player % 2 + 1  # The player who made the move
        
        # Decrement sub_counts and clear bitmask bit
        self.sub_counts[sub_idx][player - 1] -= 1
        if player == 1:
            self.x_masks[sub_idx] &= ~cell_bit
        else:
            self.o_masks[sub_idx] &= ~cell_bit
        
        board_r, board_c = r // 3, c // 3
        self.completed_boards[board_r][board_c] = prev_completed
        # Update completed_mask
        if prev_completed == 0:
            self.completed_mask &= ~(1 << sub_idx)
        self.winner = prev_winner
        self.last_move = prev_last_move
        self.current_player = self.current_player % 2 + 1

    def update_completed_boards(self, r, c):
        board_r, board_c = r // 3, c // 3
        sub_idx = board_r * 3 + board_c
        
        # Use bitmasks directly
        if self.current_player == 1:
            p_mask = self.x_masks[sub_idx]
        else:
            p_mask = self.o_masks[sub_idx]
        filled_mask = self.x_masks[sub_idx] | self.o_masks[sub_idx]
        
        # Check win patterns
        for mask in Board.WIN_MASKS:
            if (p_mask & mask) == mask:
                self.completed_boards[board_r][board_c] = self.current_player
                self.completed_mask |= (1 << sub_idx)
                return
        
        # Check draw (all 9 filled)
        if filled_mask == 0b111111111:
            self.completed_boards[board_r][board_c] = 3
            self.completed_mask |= (1 << sub_idx)

    def check_winner(self):
        # Build bitmasks for current player and all filled
        p_mask = 0
        filled_mask = 0
        for r in range(3):
            for c in range(3):
                bit = 1 << (r * 3 + c)
                if self.completed_boards[r][c] == self.current_player:
                    p_mask |= bit
                if self.completed_boards[r][c] != 0:
                    filled_mask |= bit
        
        # Check win patterns with bitmask AND
        for mask in Board.WIN_MASKS:
            if (p_mask & mask) == mask:
                self.winner = self.current_player
                return
        
        # Check draw (all 9 filled)
        if filled_mask == 0b111111111:
            self.winner = 3

    def is_game_over(self): 
        return self.winner is not None
    
    def count_playable_empty_cells(self) -> int:
        """Count only playable empty cells (excluding completed small boards)."""
        empty_count = 0
        
        for sub_idx in range(9):
            if not (self.completed_mask & (1 << sub_idx)):
                occupied = self.x_masks[sub_idx] | self.o_masks[sub_idx]
                # Count zero bits in 9-bit mask
                empty_count += 9 - bin(occupied).count('1')
        
        return empty_count
    
    @staticmethod
    def get_phase_from_state(state) -> tuple[float, str]:
        """
        Get game phase from state array using playable empty cells.
        
        Args:
            state: numpy array (2, 9, 9) or (C, 9, 9) where [player1, player2, ...]
                   Can also be any object with indexable [0] and [1]
        
        Returns:
            (weight, category):
            - weight: Training weight (0.3-1.2)
            - category: Phase name
        
        Uses playable empty cells (excluding completed boards).
        """
        
        state = np.array(state)
        if state.ndim == 2:
            total_empty = np.sum(state == 0)
            if total_empty >= 50:
                return 1.0, "opening"
            elif total_empty >= 40:
                return 1.0, "early_mid"
            elif total_empty >= 30:
                return 1.0, "mid"
            elif total_empty >= 25:
                return 1.2, "transition"
            elif total_empty >= 20:
                return 1.0, "near_endgame"
            elif total_empty >= 10:
                return 0.6, "endgame"
            else:
                return 0.4, "deep_endgame"
        
        player1 = state[0]
        player2 = state[1]
        
        playable_empty = 0
        for br in range(3):
            for bc in range(3):
                start_r, start_c = br * 3, bc * 3
                small_p1 = player1[start_r:start_r+3, start_c:start_c+3]
                small_p2 = player2[start_r:start_r+3, start_c:start_c+3]
                
                if Board._is_small_board_completed(small_p1, small_p2):
                    continue
                
                small_empty = np.sum((small_p1 == 0) & (small_p2 == 0))
                playable_empty += small_empty
        
        if playable_empty >= 50:
            return 1.0, "opening"
        elif playable_empty >= 40:
            return 1.0, "early_mid"
        elif playable_empty >= 30:
            return 1.0, "mid"
        elif playable_empty >= 25:
            return 1.2, "transition"
        elif playable_empty >= 20:
            return 1.0, "near_endgame"
        elif playable_empty >= 10:
            return 0.6, "endgame"
        else:
            return 0.4, "deep_endgame"
    
    @staticmethod
    def _is_small_board_completed(p1_board, p2_board):
        """
        Check if a 3x3 small board is completed (won or full).
        
        Args:
            p1_board: (3, 3) player 1 positions
            p2_board: (3, 3) player 2 positions
        
        Returns:
            bool: True if completed
        """
        
        for player_board in [p1_board, p2_board]:
            if np.any(np.sum(player_board, axis=1) == 3):
                return True
            if np.any(np.sum(player_board, axis=0) == 3):
                return True
            if np.trace(player_board) == 3:
                return True
            if np.trace(np.fliplr(player_board)) == 3:
                return True
        
        if np.sum((p1_board == 0) & (p2_board == 0)) == 0:
            return True
        
        return False
    
    def get_phase(self) -> tuple[float, str]:
        """
        Get game phase weight and category for this board instance.
        
        Returns:
            (weight, category):
            - weight: Training weight (0.3-1.2)
            - category: Phase name
        
        Categories:
        - opening: 50+ empty cells
        - early_mid: 40-49 empty
        - mid: 30-39 empty
        - transition: 26-29 empty (most important!)
        - near_endgame: 20-25 empty
        - endgame: 10-19 empty
        - deep_endgame: 0-9 empty
        """
        playable_empty = self.count_playable_empty_cells()
        
        if playable_empty >= 50:
            return 1.0, "opening"
        elif playable_empty >= 40:
            return 1.0, "early_mid"
        elif playable_empty >= 30:
            return 1.0, "mid"
        elif playable_empty >= 25:
            return 1.2, "transition"
        elif playable_empty >= 20:
            return 1.0, "near_endgame"
        elif playable_empty >= 10:
            return 0.6, "endgame"
        else:
            return 0.4, "deep_endgame"
    
    def swap_xo(self):
        """Swap X and O in place. O(1) for masks."""
        # Swap bitmasks - O(1)
        self.x_masks, self.o_masks = self.o_masks[:], self.x_masks[:]
        
        # Swap sub_counts
        for i in range(9):
            self.sub_counts[i] = [self.sub_counts[i][1], self.sub_counts[i][0]]
        
        # Swap completed_boards (1 <-> 2, keep 0 and 3)
        for r in range(3):
            for c in range(3):
                if self.completed_boards[r][c] == 1:
                    self.completed_boards[r][c] = 2
                elif self.completed_boards[r][c] == 2:
                    self.completed_boards[r][c] = 1
        
        # Swap current player
        self.current_player = 3 - self.current_player
    
    def transform(self, perm_id: int):
        """Apply D4 symmetry transform in place using bitmasks. O(9) for masks."""
        from utils.symmetry import D4_TRANSFORMS, ROTATED_MASKS
        perm = D4_TRANSFORMS[perm_id]
        
        # Transform bitmasks using precomputed lookup
        # perm[i] means: new position i gets old position perm[i]
        new_x = [0] * 9
        new_o = [0] * 9
        new_counts = [[0, 0] for _ in range(9)]
        for i in range(9):
            old_pos = perm[i]
            new_x[i] = ROTATED_MASKS[perm_id][self.x_masks[old_pos]]
            new_o[i] = ROTATED_MASKS[perm_id][self.o_masks[old_pos]]
            new_counts[i] = self.sub_counts[old_pos][:]
        self.x_masks = new_x
        self.o_masks = new_o
        self.sub_counts = new_counts
        
        # Transform completed_mask
        # new[i] gets old[perm[i]]
        new_completed = 0
        for i in range(9):
            old_pos = perm[i]
            if self.completed_mask & (1 << old_pos):
                new_completed |= (1 << i)
        self.completed_mask = new_completed
        
        # Transform completed_boards
        # new[i] gets old[perm[i]]
        old_completed = [row[:] for row in self.completed_boards]
        for i in range(9):
            old_pos = perm[i]
            old_r, old_c = old_pos // 3, old_pos % 3
            new_r, new_c = i // 3, i % 3
            self.completed_boards[new_r][new_c] = old_completed[old_r][old_c]
        
        # Transform last_move using INV_TRANSFORMS (find where old pos went)
        if self.last_move is not None:
            from utils.symmetry import INV_TRANSFORMS
            inv = INV_TRANSFORMS[perm_id]
            r, c = self.last_move
            old_sub_i = (r // 3) * 3 + (c // 3)
            old_cell_i = (r % 3) * 3 + (c % 3)
            # inv[old_pos] = new_pos (where old position ends up)
            new_sub_i = inv[old_sub_i]
            new_cell_i = inv[old_cell_i]
            new_r = (new_sub_i // 3) * 3 + (new_cell_i // 3)
            new_c = (new_sub_i % 3) * 3 + (new_cell_i % 3)
            self.last_move = (new_r, new_c)