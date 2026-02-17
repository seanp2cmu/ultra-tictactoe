"""Alpha-beta search engine with NNUE evaluation.

Features:
  - Negamax with alpha-beta pruning
  - Iterative deepening
  - Transposition table
  - Move ordering: TT move → killer moves → history heuristic
  - Uses board.undo_move() for efficiency (no clone)
"""
import time


INF = 1e9
MAX_DEPTH = 64

# 3x3 win patterns (9-bit masks)
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

# Meta-board win patterns (same patterns, for global board)
META_WIN_MASKS = WIN_MASKS

# Transposition table entry types
TT_EXACT = 0
TT_LOWER = 1  # beta cutoff (score >= beta)
TT_UPPER = 2  # fail low (score <= alpha)


class NNUESearch:
    """Alpha-beta search with NNUE evaluation."""
    
    def __init__(self, model):
        """
        Args:
            model: NNUE model with .evaluate(board) method
        """
        self.model = model
        self.tt = {}           # {board_hash: (depth, score, flag, best_move)}
        self.nodes = 0
        self.tt_hits = 0
        
        # Killer moves: 2 slots per ply
        self.killers = [[(None, None), (None, None)] for _ in range(MAX_DEPTH)]
        
        # History heuristic: score per move (r*9+c)
        self.history = [0] * 81
    
    def search(self, board, max_depth=8, time_limit_ms=None):
        """Iterative deepening search.
        
        Args:
            board: Board object
            max_depth: Maximum search depth
            time_limit_ms: Time limit in milliseconds (optional)
            
        Returns:
            (best_move, score, info_dict)
            best_move: (row, col) tuple
            score: evaluation from current player's perspective
        """
        self.nodes = 0
        self.tt_hits = 0
        # Reset killers each search, keep history across depths
        self.killers = [[(None, None), (None, None)] for _ in range(MAX_DEPTH)]
        start_time = time.time()
        deadline = start_time + time_limit_ms / 1000.0 if time_limit_ms else None
        
        best_move = None
        best_score = -INF
        
        for depth in range(1, max_depth + 1):
            score, move = self._negamax_root(board, depth, deadline)
            
            if deadline and time.time() > deadline:
                break
            
            if move is not None:
                best_move = move
                best_score = score
        
        info = {
            'depth': depth,
            'score': best_score,
            'nodes': self.nodes,
            'tt_hits': self.tt_hits,
            'time_ms': (time.time() - start_time) * 1000,
        }
        return best_move, best_score, info
    
    def _negamax_root(self, board, depth, deadline):
        """Root-level negamax to track best move."""
        moves = board.get_legal_moves()
        if not moves:
            return 0.0, None
        
        moves = self._order_moves(moves, tt_move=None, ply=0, board=board)
        
        best_score = -INF
        best_move = moves[0]
        alpha = -INF
        beta = INF
        
        for r, c in moves:
            sub_idx = (r // 3) * 3 + (c // 3)
            prev_completed = board.get_completed_state(sub_idx)
            prev_winner = board.winner
            prev_last_move = board.last_move
            
            board.make_move(r, c)
            score = -self._negamax(board, depth - 1, -beta, -alpha, 1, deadline)
            board.undo_move(r, c, prev_completed, prev_winner, prev_last_move)
            
            if score > best_score:
                best_score = score
                best_move = (r, c)
            alpha = max(alpha, score)
        
        return best_score, best_move
    
    def _negamax(self, board, depth, alpha, beta, ply, deadline):
        """Negamax with alpha-beta pruning, TT, killers, history."""
        self.nodes += 1
        
        # Time check every 4096 nodes
        if deadline and (self.nodes & 4095) == 0:
            if time.time() > deadline:
                return 0.0
        
        # Terminal check
        if board.winner is not None and board.winner != -1:
            if board.winner == 3:
                return 0.0
            if board.winner == board.current_player:
                return 1.0 + depth * 0.001
            else:
                return -(1.0 + depth * 0.001)
        
        # Leaf evaluation
        if depth <= 0:
            return self.model.evaluate(board)
        
        # TT probe
        board_key = self._board_hash(board)
        tt_move = None
        tt_entry = self.tt.get(board_key)
        if tt_entry is not None:
            tt_depth, tt_score, tt_flag, tt_move = tt_entry
            if tt_depth >= depth:
                self.tt_hits += 1
                if tt_flag == TT_EXACT:
                    return tt_score
                elif tt_flag == TT_LOWER and tt_score >= beta:
                    return tt_score
                elif tt_flag == TT_UPPER and tt_score <= alpha:
                    return tt_score
        
        # Generate and order moves
        moves = board.get_legal_moves()
        if not moves:
            return 0.0
        
        moves = self._order_moves(moves, tt_move, ply, board=board)
        
        # Search
        best_score = -INF
        best_move = moves[0]
        orig_alpha = alpha
        
        for r, c in moves:
            sub_idx = (r // 3) * 3 + (c // 3)
            prev_completed = board.get_completed_state(sub_idx)
            prev_winner = board.winner
            prev_last_move = board.last_move
            
            board.make_move(r, c)
            score = -self._negamax(board, depth - 1, -beta, -alpha, ply + 1, deadline)
            board.undo_move(r, c, prev_completed, prev_winner, prev_last_move)
            
            if score > best_score:
                best_score = score
                best_move = (r, c)
            
            alpha = max(alpha, score)
            if alpha >= beta:
                # Beta cutoff: update killer moves and history
                self._update_killers(ply, (r, c))
                self.history[r * 9 + c] += depth * depth
                break
        
        # TT store
        if best_score <= orig_alpha:
            tt_flag = TT_UPPER
        elif best_score >= beta:
            tt_flag = TT_LOWER
        else:
            tt_flag = TT_EXACT
        
        self.tt[board_key] = (depth, best_score, tt_flag, best_move)
        
        return best_score
    
    def _order_moves(self, moves, tt_move, ply, board=None):
        """Order: TT move → killer → tactical(win/block/meta) + history."""
        tt_list = []
        killer_list = []
        rest = []
        
        k0, k1 = self.killers[ply] if ply < MAX_DEPTH else ((None, None), (None, None))
        
        for m in moves:
            if m == tt_move:
                tt_list.append(m)
            elif m == k0 or m == k1:
                killer_list.append(m)
            else:
                rest.append(m)
        
        # Sort rest by tactical score + history
        if board is not None:
            rest.sort(key=lambda m: self._tactical_score(m, board) + self.history[m[0] * 9 + m[1]], reverse=True)
        else:
            rest.sort(key=lambda m: self.history[m[0] * 9 + m[1]], reverse=True)
        
        return tt_list + killer_list + rest
    
    def _tactical_score(self, move, board):
        """UTTT tactical heuristic for move ordering.
        
        Scores (additive):
          +30: Wins a sub-board
          +20: Winning sub-board creates meta-board 2-in-a-row
          +15: Blocks opponent from winning a sub-board
        """
        r, c = move
        sub_idx = (r // 3) * 3 + (c // 3)
        local_cell = (r % 3) * 3 + (c % 3)
        cell_bit = 1 << local_cell
        score = 0
        
        player = board.current_player
        opponent = 3 - player
        
        my_mask = board.x_masks[sub_idx] if player == 1 else board.o_masks[sub_idx]
        opp_mask = board.x_masks[sub_idx] if player == 2 else board.o_masks[sub_idx]
        
        # Check if this move wins the sub-board
        new_my_mask = my_mask | cell_bit
        wins_sub = False
        for wm in WIN_MASKS:
            if (new_my_mask & wm) == wm:
                wins_sub = True
                break
        
        if wins_sub:
            score += 30
            # Check if winning this sub-board creates a meta-board threat
            my_meta = 0
            for i in range(9):
                if board.get_completed_state(i) == player:
                    my_meta |= (1 << i)
            new_meta = my_meta | (1 << sub_idx)
            for wm in META_WIN_MASKS:
                # 2-in-a-row with one empty = threat
                overlap = new_meta & wm
                if bin(overlap).count('1') == 2 and (overlap != (my_meta & wm)):
                    score += 20
                    break
        
        # Check if this move blocks opponent from winning the sub-board
        new_opp_check = opp_mask | cell_bit  # hypothetical opponent move here
        for wm in WIN_MASKS:
            # Opponent had 2 of 3 and this cell is the 3rd
            if (opp_mask & wm) == (wm & ~cell_bit) and (wm & cell_bit):
                score += 15
                break
        
        return score
    
    def _update_killers(self, ply, move):
        """Store killer move at this ply (keep 2 per ply)."""
        if ply >= MAX_DEPTH:
            return
        if move != self.killers[ply][0]:
            self.killers[ply][1] = self.killers[ply][0]
            self.killers[ply][0] = move
    
    def _board_hash(self, board):
        """Simple board hash for TT. Uses bitmask state."""
        return (
            tuple(board.x_masks[i] for i in range(9)) +
            tuple(board.o_masks[i] for i in range(9)) +
            (board.current_player, board.last_move_r, board.last_move_c)
        )
    
    def clear_tt(self):
        """Clear transposition table and heuristics."""
        self.tt.clear()
        self.history = [0] * 81
        self.killers = [[(None, None), (None, None)] for _ in range(MAX_DEPTH)]
