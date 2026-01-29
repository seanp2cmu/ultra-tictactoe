import time
from game import Board
from agent import Agent


def test_mcts_single_action():
    print("=== MCTS 단일 행동 선택 테스트 ===")
    board = Board()
    agent = Agent(num_simulations=100)
    
    start_time = time.time()
    action = agent.select_action(board)
    elapsed = time.time() - start_time
    
    print(f"선택된 행동: {action} (row={action//9}, col={action%9})")
    print(f"소요 시간: {elapsed:.2f}초")
    print(f"합법적 행동: {(action//9, action%9) in board.get_legal_moves()}")
    print("✓ 단일 행동 선택 성공\n")


def test_mcts_game_simulation():
    print("=== MCTS 에이전트 게임 시뮬레이션 ===")
    board = Board()
    agent = Agent(num_simulations=100)
    
    step = 0
    max_steps = 100
    
    start_time = time.time()
    
    while board.winner is None and step < max_steps:
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            break
        
        action = agent.select_action(board)
        row, col = action // 9, action % 9
        
        if (row, col) not in legal_moves:
            print(f"✗ 불법 행동 선택: {action}")
            break
        
        board.make_move(row, col)
        step += 1
        
        if step % 10 == 0:
            print(f"Step {step}: 진행 중...")
    
    elapsed = time.time() - start_time
    
    print(f"게임 종료: {step} 스텝")
    print(f"승자: {board.winner}")
    print(f"총 소요 시간: {elapsed:.2f}초")
    print(f"평균 행동 선택 시간: {elapsed/step:.2f}초")
    print("✓ 게임 시뮬레이션 완료\n")


def test_mcts_vs_mcts():
    print("=== MCTS vs MCTS 대결 ===")
    board = Board()
    agent1 = Agent(num_simulations=200)
    agent2 = Agent(num_simulations=200)
    
    step = 0
    max_steps = 100
    
    start_time = time.time()
    
    while board.winner is None and step < max_steps:
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            break
        
        current_agent = agent1 if board.current_player == 1 else agent2
        action = current_agent.select_action(board)
        row, col = action // 9, action % 9
        
        if (row, col) not in legal_moves:
            print(f"✗ Player {board.current_player} 불법 행동: {action}")
            break
        
        board.make_move(row, col)
        step += 1
        
        if step % 10 == 0:
            print(f"Step {step}: Player {3 - board.current_player}가 행동함")
    
    elapsed = time.time() - start_time
    
    print(f"\n게임 결과:")
    print(f"  총 스텝: {step}")
    print(f"  승자: Player {board.winner}" if board.winner in [1, 2] else f"  무승부")
    print(f"  총 소요 시간: {elapsed:.2f}초")
    print(f"  평균 행동 선택 시간: {elapsed/step:.2f}초")
    print("✓ MCTS vs MCTS 대결 완료\n")


def test_action_probabilities():
    print("=== 행동 확률 분포 테스트 ===")
    board = Board()
    agent = Agent(num_simulations=100)
    
    start_time = time.time()
    action_probs = agent.get_action_probs(board)
    elapsed = time.time() - start_time
    
    print(f"계산된 행동 개수: {len(action_probs)}")
    print(f"확률 합: {sum(action_probs.values()):.4f}")
    print(f"소요 시간: {elapsed:.2f}초")
    
    top_5 = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\n상위 5개 행동:")
    for action, prob in top_5:
        print(f"  Action {action} (row={action//9}, col={action%9}): {prob:.4f}")
    
    print("✓ 행동 확률 분포 테스트 완료\n")


def test_different_simulation_counts():
    print("=== 시뮬레이션 횟수별 성능 비교 ===")
    board = Board()
    simulation_counts = [50, 100, 200, 500]
    
    for num_sims in simulation_counts:
        agent = Agent(num_simulations=num_sims)
        start_time = time.time()
        action = agent.select_action(board)
        elapsed = time.time() - start_time
        
        print(f"시뮬레이션 {num_sims}회: 소요 시간 {elapsed:.3f}초, 선택된 행동 {action}")
    
    print("✓ 시뮬레이션 횟수별 성능 비교 완료\n")


if __name__ == "__main__":
    print("=" * 60)
    print("MCTS 에이전트 테스트 시작")
    print("=" * 60 + "\n")
    
    try:
        test_mcts_single_action()
        test_mcts_game_simulation()
        test_action_probabilities()
        test_different_simulation_counts()
        test_mcts_vs_mcts()
        
        print("=" * 60)
        print("모든 테스트 완료!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
