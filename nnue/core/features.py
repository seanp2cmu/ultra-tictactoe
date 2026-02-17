"""Perspective-based feature encoding for NNUE.

Feature layout per perspective (199 sparse features):
  [0-80]    My pieces on cells 0-80
  [81-161]  Opponent pieces on cells 0-80
  [162-170] Sub-boards I won (0-8)
  [171-179] Sub-boards opponent won (0-8)
  [180-188] Sub-boards drawn (0-8)
  [189-197] Active sub-board constraint (0-8)
  [198]     Any sub-board playable
"""
import numpy as np

NUM_FEATURES = 199

# Feature index offsets
MY_PIECE = 0          # +cell_idx (0-80)
OPP_PIECE = 81        # +cell_idx (0-80)
MY_SUB_WON = 162      # +sub_idx (0-8)
OPP_SUB_WON = 171     # +sub_idx (0-8)
SUB_DRAW = 180         # +sub_idx (0-8)
ACTIVE_SUB = 189       # +sub_idx (0-8)
ACTIVE_ANY = 198


def extract_features(board):
    """Extract perspective-based sparse features from board.
    
    Returns:
        (stm_indices, nstm_indices): lists of active feature indices
            stm = side to move, nstm = not side to move
    """
    stm = board.current_player  # 1 or 2
    nstm = 3 - stm
    
    stm_feats = []
    nstm_feats = []
    
    # Cell features
    for sub_idx in range(9):
        sub_r, sub_c = sub_idx // 3, sub_idx % 3
        x_mask = board.x_masks[sub_idx]
        o_mask = board.o_masks[sub_idx]
        
        for cell_idx in range(9):
            bit = 1 << cell_idx
            r = sub_r * 3 + cell_idx // 3
            c = sub_c * 3 + cell_idx % 3
            global_idx = r * 9 + c
            
            if x_mask & bit:  # P1 piece
                if stm == 1:
                    stm_feats.append(MY_PIECE + global_idx)
                    nstm_feats.append(OPP_PIECE + global_idx)
                else:
                    stm_feats.append(OPP_PIECE + global_idx)
                    nstm_feats.append(MY_PIECE + global_idx)
            elif o_mask & bit:  # P2 piece
                if stm == 2:
                    stm_feats.append(MY_PIECE + global_idx)
                    nstm_feats.append(OPP_PIECE + global_idx)
                else:
                    stm_feats.append(OPP_PIECE + global_idx)
                    nstm_feats.append(MY_PIECE + global_idx)
    
    # Sub-board status features
    for sub_idx in range(9):
        status = board.get_completed_state(sub_idx)
        if status == 0:
            continue
        elif status == 3:  # draw
            stm_feats.append(SUB_DRAW + sub_idx)
            nstm_feats.append(SUB_DRAW + sub_idx)
        elif status == stm:  # STM won this sub-board
            stm_feats.append(MY_SUB_WON + sub_idx)
            nstm_feats.append(OPP_SUB_WON + sub_idx)
        else:  # NSTM won this sub-board
            stm_feats.append(OPP_SUB_WON + sub_idx)
            nstm_feats.append(MY_SUB_WON + sub_idx)
    
    # Active sub-board constraint
    if board.last_move_r < 0:
        stm_feats.append(ACTIVE_ANY)
        nstm_feats.append(ACTIVE_ANY)
    else:
        target_sub = (board.last_move_r % 3) * 3 + (board.last_move_c % 3)
        if board.get_completed_state(target_sub) != 0:
            stm_feats.append(ACTIVE_ANY)
            nstm_feats.append(ACTIVE_ANY)
        else:
            stm_feats.append(ACTIVE_SUB + target_sub)
            nstm_feats.append(ACTIVE_SUB + target_sub)
    
    return stm_feats, nstm_feats


def features_to_tensor(stm_indices, nstm_indices, dtype=np.float32):
    """Convert sparse feature indices to dense tensors.
    
    Returns:
        (stm_tensor, nstm_tensor): shape (NUM_FEATURES,) each
    """
    stm = np.zeros(NUM_FEATURES, dtype=dtype)
    nstm = np.zeros(NUM_FEATURES, dtype=dtype)
    for i in stm_indices:
        stm[i] = 1.0
    for i in nstm_indices:
        nstm[i] = 1.0
    return stm, nstm


def board_array_to_features(arr, dtype=np.float32):
    """Convert stored board array (92,) to feature tensors.
    
    Used for training data conversion. The board array format:
      [0-80]: cell values (0=empty, 1=P1, 2=P2)
      [81-89]: sub-board status (0=open, 1=P1, 2=P2, 3=draw)
      [90]: active sub-board (-1=any, 0-8=sub_idx)
      [91]: current_player (1 or 2)
    """
    cells = arr[:81]
    meta = arr[81:90]
    active = int(arr[90])
    current_player = int(arr[91])
    
    stm = current_player
    
    stm_feats = []
    nstm_feats = []
    
    # Cell features
    for idx in range(81):
        v = cells[idx]
        if v == 0:
            continue
        if v == stm:
            stm_feats.append(MY_PIECE + idx)
            nstm_feats.append(OPP_PIECE + idx)
        else:
            stm_feats.append(OPP_PIECE + idx)
            nstm_feats.append(MY_PIECE + idx)
    
    # Sub-board status
    for sub_idx in range(9):
        status = meta[sub_idx]
        if status == 0:
            continue
        elif status == 3:
            stm_feats.append(SUB_DRAW + sub_idx)
            nstm_feats.append(SUB_DRAW + sub_idx)
        elif status == stm:
            stm_feats.append(MY_SUB_WON + sub_idx)
            nstm_feats.append(OPP_SUB_WON + sub_idx)
        else:
            stm_feats.append(OPP_SUB_WON + sub_idx)
            nstm_feats.append(MY_SUB_WON + sub_idx)
    
    # Active sub-board
    if active < 0:
        stm_feats.append(ACTIVE_ANY)
        nstm_feats.append(ACTIVE_ANY)
    else:
        stm_feats.append(ACTIVE_SUB + active)
        nstm_feats.append(ACTIVE_SUB + active)
    
    return features_to_tensor(stm_feats, nstm_feats, dtype=dtype)
