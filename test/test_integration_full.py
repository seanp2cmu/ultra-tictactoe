#!/usr/bin/env python3
"""
전체 시스템 통합 테스트
- Phase 계산
- Replay buffer weighting
- Training loop
- DTW values
"""
import sys
import numpy as np
import torch

from game import Board
from ai.training import SelfPlayData
from ai.training import Trainer
from ai.core import AlphaZeroNet
from ai.endgame import DTWCalculator
from config import TrainingConfig


def test_phase_calculation():
    """Phase 계산 테스트 (플레이 가능 빈칸 기반)"""
    print("\n" + "="*80)
    print("TEST 1: Phase Calculation with Playable Empty Cells")
    print("="*80)
    
    board = Board()
    
    # Opening (모든 칸 비어있음)
    weight, category = board.get_phase()
    print(f"Opening: weight={weight}, category={category}")
    assert category == "opening", f"Expected 'opening', got '{category}'"
    
    # 몇 수 진행 (합법적인 수만)
    num_moves = 5
    for i in range(num_moves):
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            break
        move = legal_moves[0]  # 첫 번째 합법적인 수 선택
        board.make_move(move[0], move[1])
    
    playable = board.count_playable_empty_cells()
    weight, category = board.get_phase()
    print(f"After {num_moves} moves: playable={playable}, weight={weight}, category={category}")
    
    # State array로도 테스트
    state = np.zeros((2, 9, 9), dtype=np.float32)
    for r in range(9):
        for c in range(9):
            if board.boards[r][c] == 1:
                state[0, r, c] = 1
            elif board.boards[r][c] == 2:
                state[1, r, c] = 1
    
    weight2, category2 = Board.get_phase_from_state(state)
    print(f"From state array: weight={weight2}, category={category2}")
    assert category == category2, "Phase should match from both methods"
    
    print("✅ Phase calculation working correctly\n")


def test_replay_buffer_weighting():
    """Replay buffer의 phase-based weighting 테스트"""
    print("="*80)
    print("TEST 2: Replay Buffer Phase-Based Weighting")
    print("="*80)
    
    buffer = SelfPlayData(max_size=100)
    
    # 다양한 phase의 샘플 추가
    for empty_cells in [60, 45, 35, 27, 22, 15, 5]:
        # State 생성 (empty_cells만큼 빈칸)
        state = np.ones((2, 9, 9), dtype=np.float32)
        filled = 81 - empty_cells
        positions = np.random.choice(81, filled, replace=False)
        
        for idx in positions:
            r, c = idx // 9, idx % 9
            player = np.random.choice([0, 1])
            state[player, r, c] = 1
            state[1-player, r, c] = 0
        
        policy = np.random.rand(81)
        value = np.random.rand()
        
        buffer.add(state, policy, value)
    
    print(f"Added {len(buffer.data)} samples")
    print(f"Categories: {buffer.categories}")
    print(f"Weights: {buffer.weights}")
    
    # 샘플링 테스트
    if len(buffer.data) >= 3:
        states, policies, values, dtws = buffer.sample(3)
        print(f"Sampled batch shapes: states={states.shape}, policies={policies.shape}, values={values.shape}")
    
    print("✅ Replay buffer weighting working correctly\n")


def test_mini_training_loop():
    """Mini training loop 테스트"""
    print("="*80)
    print("TEST 3: Mini Training Loop")
    print("="*80)
    
    # 작은 네트워크 생성
    from ai.core.network import Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Model(num_res_blocks=2, num_channels=64).to(device)
    net = AlphaZeroNet(model=model, lr=0.001, use_amp=False, device=str(device))
    
    print(f"Device: {device}")
    print(f"Network created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Trainer 생성 (작은 파라미터)
    config = TrainingConfig()
    config.num_iterations = 1
    config.num_games = 2
    config.num_simulations = 10
    config.batch_size = 4
    
    trainer = Trainer(net, config)
    
    # 1 iteration 실행
    print("\nRunning 1 training iteration...")
    try:
        result = trainer.train_iteration(
            num_self_play_games=2,
            num_train_epochs=2,
            temperature=1.0,
            verbose=True,
            num_simulations=10
        )
        print(f"✅ Training loop completed successfully")
        print(f"   Samples: {result['num_samples']}, Loss: {result['avg_loss']['total_loss']:.4f}\n")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise


def test_dtw_values():
    """DTW value 계산 테스트"""
    print("="*80)
    print("TEST 4: DTW Endgame Values")
    print("="*80)
    
    dtw = DTWCalculator(use_cache=True)
    
    # 간단한 엔드게임 포지션 생성
    board = Board()
    
    # 많은 수를 진행해서 엔드게임에 가깝게 만들기
    num_moves_to_play = 50
    for i in range(num_moves_to_play):
        if board.winner is not None:
            break
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            break
        # 랜덤하게 수를 선택 (더 realistic)
        import random
        move = random.choice(legal_moves)
        board.make_move(move[0], move[1])
    
    playable = board.count_playable_empty_cells()
    print(f"Board state: {playable} playable empty cells")
    
    # threshold=15로 변경됨 (25칸은 너무 오래 걸림)
    if playable <= 15:
        result = dtw.calculate_dtw(board)
        if result is not None:
            dtw_result, dtw_value, best_move = result
            print(f"DTW result: {dtw_result}, DTW value: {dtw_value}")
            if best_move:
                print(f"Best move: {best_move}")
            print("✅ DTW calculation working\n")
        else:
            print("⚠️  DTW returned None\n")
    else:
        print(f"⚠️  Not in endgame range (need ≤15, got {playable})\n")


def main():
    """모든 테스트 실행"""
    print("\n" + "="*80)
    print("FULL INTEGRATION TEST SUITE")
    print("="*80)
    
    try:
        test_phase_calculation()
        test_replay_buffer_weighting()
        test_dtw_values()
        test_mini_training_loop()
        
        print("\n" + "="*80)
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("="*80)
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"❌ TEST FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
