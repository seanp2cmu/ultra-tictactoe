import numpy as np
from game import Board
from network import AlphaZeroNet
from agent import AlphaZeroAgent
from trainer import AlphaZeroTrainer, SelfPlayWorker


def test_network_initialization():
    print("=== 신경망 초기화 테스트 ===")
    network = AlphaZeroNet()
    print(f"Device: {network.device}")
    print(f"Model: {network.model.__class__.__name__}")
    print("✓ 신경망 초기화 성공\n")
    return network


def test_network_prediction():
    print("=== 신경망 예측 테스트 ===")
    network = AlphaZeroNet()
    board = Board()
    
    policy_probs, value = network.predict(board)
    
    print(f"Policy shape: {policy_probs.shape}")
    print(f"Policy sum: {np.sum(policy_probs):.4f}")
    print(f"Value: {value:.4f}")
    print(f"Value range: [{value:.4f}] (should be in [-1, 1])")
    print(f"Top 5 actions:")
    top_5_idx = np.argsort(policy_probs)[-5:][::-1]
    for idx in top_5_idx:
        print(f"  Action {idx} (row={idx//9}, col={idx%9}): {policy_probs[idx]:.4f}")
    print("✓ 신경망 예측 성공\n")
    return network


def test_alphazero_agent():
    print("=== AlphaZero 에이전트 테스트 ===")
    network = AlphaZeroNet()
    agent = AlphaZeroAgent(network, num_simulations=50, temperature=1.0)
    
    board = Board()
    
    print("행동 선택 중...")
    action = agent.select_action(board)
    
    print(f"선택된 행동: {action} (row={action//9}, col={action%9})")
    
    legal_moves = board.get_legal_moves()
    is_legal = (action // 9, action % 9) in legal_moves
    print(f"합법적 행동: {is_legal}")
    
    print("✓ AlphaZero 에이전트 테스트 성공\n")
    return agent


def test_alphazero_game():
    print("=== AlphaZero 에이전트 게임 플레이 테스트 ===")
    network = AlphaZeroNet()
    agent = AlphaZeroAgent(network, num_simulations=50, temperature=1.0)
    
    board = Board()
    step = 0
    max_steps = 100
    
    while board.winner is None and step < max_steps:
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            break
        
        action = agent.select_action(board, temperature=0.5)
        row, col = action // 9, action % 9
        
        if (row, col) not in legal_moves:
            print(f"✗ 불법 행동: {action}")
            break
        
        board.make_move(row, col)
        step += 1
        
        if step % 10 == 0:
            print(f"Step {step}")
    
    print(f"게임 종료: {step} 스텝")
    print(f"승자: {board.winner}")
    print("✓ AlphaZero 게임 플레이 성공\n")


def test_self_play():
    print("=== Self-play 데이터 생성 테스트 ===")
    network = AlphaZeroNet()
    worker = SelfPlayWorker(network, num_simulations=30, temperature=1.0)
    
    print("Self-play 게임 진행 중...")
    game_data = worker.play_game(verbose=True)
    
    print(f"\n수집된 데이터 개수: {len(game_data)}")
    if game_data:
        state, policy, value = game_data[0]
        print(f"State shape: {state.shape}")
        print(f"Policy shape: {policy.shape}")
        print(f"Value: {value}")
    
    print("✓ Self-play 데이터 생성 성공\n")


def test_trainer_initialization():
    print("=== 트레이너 초기화 테스트 ===")
    trainer = AlphaZeroTrainer(lr=0.001, batch_size=16, num_simulations=30)
    
    print(f"Batch size: {trainer.batch_size}")
    print(f"Num simulations: {trainer.num_simulations}")
    print(f"Replay buffer size: {len(trainer.replay_buffer)}")
    print("✓ 트레이너 초기화 성공\n")
    return trainer


def test_training_iteration():
    print("=== 학습 반복 테스트 ===")
    trainer = AlphaZeroTrainer(lr=0.001, batch_size=16, num_simulations=30)
    
    print("학습 반복 시작...")
    result = trainer.train_iteration(
        num_self_play_games=3,
        num_train_epochs=10,
        temperature=1.0,
        verbose=True
    )
    
    print(f"\n학습 결과:")
    print(f"  수집된 샘플: {result['num_samples']}")
    print(f"  평균 손실: {result['avg_loss']:.4f}")
    print("✓ 학습 반복 성공\n")


def test_save_load():
    print("=== 모델 저장/로드 테스트 ===")
    import tempfile
    import os
    
    trainer = AlphaZeroTrainer(lr=0.001)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_model.pth")
        
        print(f"모델 저장: {filepath}")
        trainer.save(filepath)
        
        print("새 트레이너 생성 및 모델 로드")
        new_trainer = AlphaZeroTrainer(lr=0.001)
        new_trainer.load(filepath)
        
        print("✓ 모델 저장/로드 성공\n")


def test_comparison_vs_random():
    print("=== AlphaZero vs 랜덤 에이전트 대결 ===")
    network = AlphaZeroNet()
    alphazero = AlphaZeroAgent(network, num_simulations=30, temperature=0)
    
    board = Board()
    step = 0
    max_steps = 100
    
    while board.winner is None and step < max_steps:
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            break
        
        if board.current_player == 1:
            action = alphazero.select_action(board, temperature=0)
        else:
            move = np.random.choice(len(legal_moves))
            action = legal_moves[move][0] * 9 + legal_moves[move][1]
        
        row, col = action // 9, action % 9
        
        if (row, col) not in legal_moves:
            print(f"✗ 불법 행동: Player {board.current_player}, action {action}")
            break
        
        board.make_move(row, col)
        step += 1
        
        if step % 10 == 0:
            print(f"Step {step}")
    
    print(f"\n게임 결과:")
    print(f"  총 스텝: {step}")
    if board.winner == 1:
        print(f"  승자: AlphaZero (Player 1)")
    elif board.winner == 2:
        print(f"  승자: 랜덤 (Player 2)")
    else:
        print(f"  무승부")
    print("✓ 대결 테스트 완료\n")


if __name__ == "__main__":
    print("=" * 60)
    print("AlphaZero 시스템 테스트 시작")
    print("=" * 60 + "\n")
    
    try:
        test_network_initialization()
        test_network_prediction()
        test_alphazero_agent()
        test_alphazero_game()
        test_self_play()
        test_trainer_initialization()
        test_training_iteration()
        test_save_load()
        test_comparison_vs_random()
        
        print("=" * 60)
        print("모든 테스트 완료!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
