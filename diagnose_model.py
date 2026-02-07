"""
Model Diagnostic Script - Check if model is learning correctly
"""
import torch
import numpy as np
from ai.core import AlphaZeroNet
from game import Board

def diagnose_model(model_path: str):
    """Diagnose model predictions."""
    
    # Load model
    print(f"\n{'='*60}")
    print(f"Diagnosing: {model_path}")
    print('='*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    
    network = AlphaZeroNet(device=device)
    network.load(model_path)
    
    # Test 1: Empty board
    print(f"\n{'='*60}")
    print("TEST 1: Empty Board (Initial Position)")
    print('='*60)
    
    board = Board()
    policy, value = network.predict(board)
    
    print(f"\nValue prediction: {value:.4f}")
    print(f"  (Should be ~0 for balanced position)")
    
    # Policy analysis
    policy_arr = np.array(policy)
    legal_moves = board.get_legal_moves()
    legal_indices = [r*9 + c for r, c in legal_moves]
    
    legal_policy = policy_arr[legal_indices]
    illegal_policy = np.delete(policy_arr, legal_indices)
    
    print(f"\nPolicy Distribution:")
    print(f"  Legal moves count: {len(legal_moves)}")
    print(f"  Legal policy sum: {legal_policy.sum():.4f} (should be ~1.0)")
    print(f"  Illegal policy sum: {illegal_policy.sum():.6f} (should be ~0)")
    print(f"  Legal policy max: {legal_policy.max():.4f}")
    print(f"  Legal policy min: {legal_policy.min():.4f}")
    print(f"  Legal policy std: {legal_policy.std():.4f}")
    
    # Check if policy is collapsed (one move dominates)
    sorted_policy = np.sort(legal_policy)[::-1]
    top1 = sorted_policy[0]
    top3_sum = sorted_policy[:3].sum()
    
    print(f"\nPolicy Concentration:")
    print(f"  Top 1 move: {top1*100:.1f}%")
    print(f"  Top 3 moves: {top3_sum*100:.1f}%")
    
    if top1 > 0.9:
        print("  ⚠️ WARNING: Policy collapsed to single move!")
    elif top1 < 0.05:
        print("  ⚠️ WARNING: Policy is nearly uniform (not learning)")
    else:
        print("  ✓ Policy distribution looks reasonable")
    
    # Show top 5 moves
    top_indices = np.argsort(policy_arr)[::-1][:5]
    print(f"\nTop 5 moves (action, prob, row, col):")
    for idx in top_indices:
        row, col = idx // 9, idx % 9
        meta_row, meta_col = row // 3, col // 3
        local_row, local_col = row % 3, col % 3
        print(f"  Action {idx}: {policy_arr[idx]*100:.2f}% -> Board({meta_row},{meta_col}) Cell({local_row},{local_col})")
    
    # Test 2: After a few moves
    print(f"\n{'='*60}")
    print("TEST 2: Mid-game Position (after 10 moves)")
    print('='*60)
    
    board2 = Board()
    moves = [(4,4), (4,3), (3,4), (3,3), (3,0), (0,0), (0,4), (4,0), (0,3), (3,6)]
    for r, c in moves:
        if board2.winner is None:
            legal = board2.get_legal_moves()
            if (r, c) in legal:
                board2.make_move(r, c)
    
    policy2, value2 = network.predict(board2)
    print(f"\nValue prediction: {value2:.4f}")
    
    legal_moves2 = board2.get_legal_moves()
    if legal_moves2:
        legal_indices2 = [r*9 + c for r, c in legal_moves2]
        legal_policy2 = np.array(policy2)[legal_indices2]
        
        print(f"Legal moves: {len(legal_moves2)}")
        print(f"Policy max: {legal_policy2.max():.4f}")
        print(f"Policy min: {legal_policy2.min():.4f}")
    
    # Test 3: Play actual game vs Random
    print(f"\n{'='*60}")
    print("TEST 3: Actual Game vs Random (with move-by-move analysis)")
    print('='*60)
    
    from ai.mcts import AlphaZeroAgent
    from ai.baselines import RandomAgent
    from ai.endgame import DTWCalculator
    
    dtw = DTWCalculator(use_cache=True)
    az_agent = AlphaZeroAgent(
        network=network,
        num_simulations=200,
        c_puct=1.0,
        temperature=0.0,
        batch_size=8,
        dtw_calculator=dtw
    )
    random_agent = RandomAgent()
    
    board3 = Board()
    move_num = 0
    
    print("\nPlaying: AlphaZero (X) vs Random (O)")
    print("-" * 40)
    
    while board3.winner is None and move_num < 81:
        is_az_turn = board3.current_player == 1
        
        if is_az_turn:
            # Show what model thinks before moving
            policy, value = network.predict(board3)
            legal = board3.get_legal_moves()
            
            action = az_agent.select_action(board3, temperature=0.0)
            move_r, move_c = action // 9, action % 9
            
            # Get top policy move
            legal_probs = [(r*9+c, policy[r*9+c]) for r, c in legal]
            legal_probs.sort(key=lambda x: -x[1])
            top_policy_action = legal_probs[0][0]
            
            print(f"Move {move_num+1} (AZ): ({move_r},{move_c}) | Value: {value:.3f} | Top Policy: {top_policy_action} ({legal_probs[0][1]*100:.1f}%) | MCTS chose: {action}")
            if action != top_policy_action:
                print(f"  ⚠️ MCTS chose different from raw policy!")
        else:
            action = random_agent.select_action(board3)
            move_r, move_c = action // 9, action % 9
            print(f"Move {move_num+1} (Rand): ({move_r},{move_c})")
        
        board3.make_move(move_r, move_c)
        move_num += 1
    
    print("-" * 40)
    if board3.winner == 1:
        print("Result: AlphaZero WINS ✓")
    elif board3.winner == 2:
        print("Result: Random WINS ✗")
    else:
        print("Result: DRAW")
    
    # Test 4: Value distribution across random positions
    print(f"\n{'='*60}")
    print("TEST 4: Value Distribution (20 random positions)")
    print('='*60)
    
    values_p1 = []  # Values when player 1 to move
    values_p2 = []  # Values when player 2 to move
    
    for i in range(20):
        test_board = Board()
        # Make random moves
        num_moves = np.random.randint(5, 30)
        for _ in range(num_moves):
            if test_board.winner is not None:
                break
            legal = test_board.get_legal_moves()
            if not legal:
                break
            move = legal[np.random.randint(len(legal))]
            test_board.make_move(move[0], move[1])
        
        if test_board.winner is None:
            _, v = network.predict(test_board)
            if test_board.current_player == 1:
                values_p1.append(v)
            else:
                values_p2.append(v)
    
    print(f"\nPlayer 1 to move ({len(values_p1)} positions):")
    if values_p1:
        print(f"  Mean: {np.mean(values_p1):.3f}, Std: {np.std(values_p1):.3f}")
        print(f"  Min: {np.min(values_p1):.3f}, Max: {np.max(values_p1):.3f}")
    
    print(f"\nPlayer 2 to move ({len(values_p2)} positions):")
    if values_p2:
        print(f"  Mean: {np.mean(values_p2):.3f}, Std: {np.std(values_p2):.3f}")
        print(f"  Min: {np.min(values_p2):.3f}, Max: {np.max(values_p2):.3f}")
    
    all_values = values_p1 + values_p2
    if all_values:
        print(f"\nOverall ({len(all_values)} positions):")
        print(f"  Mean: {np.mean(all_values):.3f}")
        negative_count = sum(1 for v in all_values if v < 0)
        print(f"  Negative values: {negative_count}/{len(all_values)} ({100*negative_count/len(all_values):.0f}%)")
        
        if np.mean(all_values) < -0.5:
            print("\n  ❌ PROBLEM: Model predicts negative values on average!")
            print("     This means it thinks the current player is usually losing.")
            print("     Possible causes:")
            print("     - Bug in value target computation during training")
            print("     - Self-play games are heavily biased (P2 always wins?)")
            print("     - Value head overfitting to negative values")
    
    # Test 5: Raw Policy vs MCTS comparison
    print(f"\n{'='*60}")
    print("TEST 5: Raw Policy (no MCTS) vs Random")
    print('='*60)
    
    raw_wins = 0
    raw_losses = 0
    raw_draws = 0
    
    for game_i in range(10):
        board5 = Board()
        while board5.winner is None:
            legal = board5.get_legal_moves()
            if not legal:
                break
            
            if board5.current_player == 1:  # AlphaZero's turn - use raw policy
                policy, _ = network.predict(board5)
                # Get best legal move from policy
                legal_probs = [(r*9+c, policy[r*9+c]) for r, c in legal]
                best_action = max(legal_probs, key=lambda x: x[1])[0]
                move = (best_action // 9, best_action % 9)
            else:  # Random's turn
                move = legal[np.random.randint(len(legal))]
            
            board5.make_move(move[0], move[1])
        
        if board5.winner == 1:
            raw_wins += 1
        elif board5.winner == 2:
            raw_losses += 1
        else:
            raw_draws += 1
    
    print(f"\nRaw Policy vs Random (10 games):")
    print(f"  Wins: {raw_wins}, Losses: {raw_losses}, Draws: {raw_draws}")
    print(f"  Win Rate: {raw_wins/10*100:.0f}%")
    
    if raw_wins > 5:
        print("\n  ✓ Raw policy is decent - MCTS might be hurting performance!")
    else:
        print("\n  ❌ Raw policy is also weak - model training has issues")
    
    print(f"\n{'='*60}")
    print("DIAGNOSIS COMPLETE")
    print('='*60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        diagnose_model(sys.argv[1])
    else:
        # Default: check model_60 if exists
        import os
        model_path = "./model/model_60.pt"
        if os.path.exists(model_path):
            diagnose_model(model_path)
        else:
            print("Usage: python diagnose_model.py <model_path>")
            print("Or place model_60.pt in ./model/")
