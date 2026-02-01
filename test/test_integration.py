"""
Network 변경사항 통합 테스트
- Board.last_move 검증
- Network + Board 통합
- Agent 호환성
- Trainer 호환성
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from game import Board
from ai.core import AlphaZeroNet, Model
from ai.mcts import AlphaZeroAgent


def test_board_last_move():
    """Step 2: Board.last_move 기능 테스트"""
    print("\n=== Step 2: Board Last Move Test ===")
    
    board = Board()
    
    # 초기 상태
    assert board.last_move is None, "Initial last_move should be None"
    print("✓ Initial last_move is None")
    
    # 첫 수
    board.make_move(0, 0)
    assert board.last_move == (0, 0), f"Expected (0,0), got {board.last_move}"
    print(f"✓ After move (0,0): last_move = {board.last_move}")
    
    # 두 번째 수
    legal_moves = board.get_legal_moves()
    assert len(legal_moves) > 0, "Should have legal moves"
    
    # 합법수 중 하나 선택
    next_move = legal_moves[0]
    board.make_move(next_move[0], next_move[1])
    assert board.last_move == next_move, f"Expected {next_move}, got {board.last_move}"
    print(f"✓ After move {next_move}: last_move = {board.last_move}")
    
    # 세 번째 수
    legal_moves = board.get_legal_moves()
    if legal_moves:
        next_move = legal_moves[0]
        board.make_move(next_move[0], next_move[1])
        assert board.last_move == next_move
        print(f"✓ After move {next_move}: last_move = {board.last_move}")
    
    print("✓ Board.last_move tracking works correctly")


def test_network_board_integration():
    """Step 3: Network + Board 통합 테스트"""
    print("\n=== Step 3: Network + Board Integration ===")
    
    model = Model(num_res_blocks=5, num_channels=64)
    board = Board()
    
    # 게임 진행
    moves = [(0, 0), (0, 1), (1, 3)]
    for move in moves:
        board.make_move(move[0], move[1])
    
    # Tensor 생성
    tensor = model._board_to_tensor(board)
    
    # Shape 확인
    assert tensor.shape == (1, 7, 9, 9), f"Expected (1,7,9,9), got {tensor.shape}"
    print(f"✓ Tensor shape: {tensor.shape}")
    
    # Last move plane 확인
    last_move_plane = tensor[0, 5].cpu().numpy()
    last_move = board.last_move
    if last_move:
        assert last_move_plane[last_move[0], last_move[1]] == 1.0
        print(f"✓ Last move ({last_move}) correctly encoded in tensor")
    
    # Forward pass
    policy_logits, value = model(tensor)
    assert policy_logits.shape == (1, 81)
    assert value.shape == (1, 1)
    print(f"✓ Forward pass successful")
    print(f"  Policy shape: {policy_logits.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")


def test_agent_compatibility():
    """Step 4: Agent 호환성 테스트"""
    print("\n=== Step 4: Agent Compatibility Test ===")
    
    network = AlphaZeroNet(
        lr=0.001,
        total_iterations=100,
        use_amp=False
    )
    
    agent = AlphaZeroAgent(
        network=network,
        num_simulations=10,
        c_puct=1.0,
        temperature=1.0,
        batch_size=4,
        use_dtw=False
    )
    
    board = Board()
    
    # MCTS search
    print("Running MCTS search...")
    root = agent.search(board)
    
    assert root is not None, "MCTS should return root node"
    assert root.visits > 0, f"Root should have visits, got {root.visits}"
    print(f"✓ MCTS search completed")
    print(f"  Root visits: {root.visits}")
    print(f"  Root value: {root.value():.3f}")
    print(f"  Children: {len(root.children)}")
    
    # Select action
    action = agent.select_action(board, temperature=1.0)
    assert action is not None, "Should select an action"
    assert 0 <= action < 81, f"Action should be in [0,81), got {action}"
    print(f"✓ Action selection successful: {action}")
    
    # Verify action is legal
    row, col = action // 9, action % 9
    legal_moves = board.get_legal_moves()
    assert (row, col) in legal_moves, f"Selected action {(row,col)} should be legal"
    print(f"✓ Selected action is legal")


def test_network_training_step():
    """Step 5: Network training step 테스트"""
    print("\n=== Step 5: Network Training Step Test ===")
    
    network = AlphaZeroNet(
        lr=0.001,
        total_iterations=100,
        use_amp=False
    )
    
    model = network.model
    board = Board()
    
    # 여러 게임 상태 생성
    boards_list = []
    for i in range(8):
        b = Board()
        # 몇 수 진행
        for _ in range(min(i+1, 3)):
            legal_moves = b.get_legal_moves()
            if legal_moves:
                move = legal_moves[0]
                b.make_move(move[0], move[1])
        
        state = model._board_to_tensor(b).squeeze(0).cpu().numpy()
        boards_list.append(state)
    
    boards = np.array(boards_list)
    policies = np.random.rand(8, 81).astype(np.float32)
    policies = policies / policies.sum(axis=1, keepdims=True)
    values = (np.random.rand(8).astype(np.float32) - 0.5) * 2
    
    # Training step
    total_loss, policy_loss, value_loss = network.train_step(boards, policies, values)
    
    assert total_loss > 0, f"Total loss should be positive, got {total_loss}"
    assert policy_loss > 0, f"Policy loss should be positive, got {policy_loss}"
    print(f"✓ Training step successful")
    print(f"  Total loss: {total_loss:.4f}")
    print(f"  Policy loss: {policy_loss:.4f}")
    print(f"  Value loss: {value_loss:.4f}")


def test_full_game_simulation():
    """Step 6: 전체 게임 시뮬레이션"""
    print("\n=== Step 6: Full Game Simulation ===")
    
    network = AlphaZeroNet(
        lr=0.001,
        total_iterations=100,
        use_amp=False
    )
    
    agent = AlphaZeroAgent(
        network=network,
        num_simulations=10,
        batch_size=4,
        use_dtw=False
    )
    
    board = Board()
    moves_played = 0
    max_moves = 20
    
    print("Playing a game...")
    while board.winner is None and moves_played < max_moves:
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            break
        
        # MCTS로 액션 선택
        action = agent.select_action(board, temperature=1.0)
        row, col = action // 9, action % 9
        
        if (row, col) not in legal_moves:
            print(f"  ❌ Illegal move selected: {(row, col)}")
            break
        
        board.make_move(row, col)
        moves_played += 1
        
        if moves_played <= 5:
            print(f"  Move {moves_played}: {(row, col)}, Player {3 - board.current_player}")
    
    print(f"✓ Game simulation completed")
    print(f"  Moves played: {moves_played}")
    print(f"  Game over: {board.winner is not None}")
    if board.winner:
        print(f"  Winner: Player {board.winner}")
    
    assert moves_played > 0, "Should play at least one move"
    print(f"✓ Full game simulation successful")


if __name__ == "__main__":
    print("=" * 70)
    print("Network 변경사항 통합 테스트")
    print("=" * 70)
    
    try:
        test_board_last_move()
        test_network_board_integration()
        test_agent_compatibility()
        test_network_training_step()
        test_full_game_simulation()
        
        print("\n" + "=" * 70)
        print("✅ 모든 통합 테스트 통과!")
        print("=" * 70)
        print("\n검증 완료:")
        print("1. ✅ Board.last_move 정상 작동")
        print("2. ✅ Network + Board 통합 성공")
        print("3. ✅ Agent 호환성 확인")
        print("4. ✅ Training step 정상 작동")
        print("5. ✅ Full game simulation 성공")
        print("\n결론: 모든 컴포넌트가 새로운 7-channel input과 호환됩니다!")
        
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
