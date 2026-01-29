import numpy as np
from env import Env

def test_env_init():
    print("=== 환경 초기화 테스트 ===")
    env = Env()
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print("✓ 환경 초기화 성공\n")
    return env

def test_reset():
    print("=== Reset 테스트 ===")
    env = Env()
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Initial board state:\n{obs}")
    print(f"All zeros: {np.all(obs == 0)}")
    print("✓ Reset 성공\n")
    return env

def test_legal_move():
    print("=== 합법적 행동 테스트 ===")
    env = Env()
    env.reset()
    
    action = 0
    print(f"Action {action} (row=0, col=0) 실행")
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")
    print(f"Board state:\n{obs}")
    print("✓ 합법적 행동 성공\n")
    return env

def test_illegal_move():
    print("=== 불법 행동 테스트 ===")
    env = Env()
    env.reset()
    
    env.step(0)
    print("첫 번째 행동: action 0 실행")
    
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"같은 위치에 두 번째 행동 시도: action 0")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Info: {info}")
    
    if reward == -1.0 and terminated and info.get("illegal_move"):
        print("✓ 불법 행동 패널티 정상 작동\n")
    else:
        print("✗ 불법 행동 처리 오류\n")

def test_multiple_moves():
    print("=== 여러 행동 테스트 ===")
    env = Env()
    env.reset()
    
    actions = [0, 10, 20, 30, 40]
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward}, terminated={terminated}")
        
        if terminated:
            print("게임 종료")
            break
    
    print(f"Final board state:\n{obs}")
    print("✓ 여러 행동 테스트 완료\n")

def test_full_game():
    print("=== 전체 게임 시뮬레이션 ===")
    env = Env()
    obs, _ = env.reset()
    
    step_count = 0
    max_steps = 100
    
    while step_count < max_steps:
        legal_moves = env.board.get_legal_moves()
        
        if not legal_moves:
            print("더 이상 합법적인 행동이 없습니다")
            break
        
        move = legal_moves[0]
        action = move[0] * 9 + move[1]
        
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        
        if terminated:
            print(f"게임 종료! Steps: {step_count}, Reward: {reward}")
            print(f"Winner: {env.board.winner}")
            break
    
    if step_count >= max_steps:
        print(f"최대 스텝 수({max_steps}) 도달")
    
    print("✓ 전체 게임 시뮬레이션 완료\n")

def test_observation_space():
    print("=== Observation Space 테스트 ===")
    env = Env()
    obs, _ = env.reset()
    
    print(f"Observation in space: {env.observation_space.contains(obs)}")
    print(f"Observation shape: {obs.shape}")
    print(f"Expected shape: {env.observation_space.shape}")
    print(f"Min value: {obs.min()}, Max value: {obs.max()}")
    print(f"Expected range: [0, 2]")
    print("✓ Observation Space 테스트 완료\n")

if __name__ == "__main__":
    print("=" * 50)
    print("Gymnasium 환경 테스트 시작")
    print("=" * 50 + "\n")
    
    try:
        test_env_init()
        test_reset()
        test_legal_move()
        test_illegal_move()
        test_multiple_moves()
        test_observation_space()
        test_full_game()
        
        print("=" * 50)
        print("모든 테스트 완료!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
