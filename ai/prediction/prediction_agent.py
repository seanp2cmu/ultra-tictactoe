"""
실전 게임/예측용 AlphaZero Agent 헬퍼
DTW가 활성화된 강화된 Agent 생성
"""
from ai.mcts import AlphaZeroAgent
from ai.core import AlphaZeroNet
from ai.endgame import DTWCalculator


def create_prediction_agent(model_path=None, network=None, num_simulations=400, 
                            temperature=0.1, dtw_cache_path=None):
    """
    실전/예측용 Agent 생성 (DTW 활성화, endgame threshold=15 고정)
    
    Args:
        model_path: 모델 파일 경로
        network: 이미 로드된 네트워크 (model_path와 둘 중 하나만)
        num_simulations: MCTS 시뮬레이션 수 (기본 400, 실전에서 높게)
        temperature: Temperature (기본 0.1, 낮을수록 결정적)
        dtw_cache_path: DTW 캐시 파일 경로 (.pkl, 선택사항)
    
    Returns:
        AlphaZeroAgent with DTW enabled (endgame ≤15 cells)
    
    Example:
        # 1. 파일에서 로드
        agent = create_prediction_agent("./model/model_final.pth")
        
        # 2. 이미 로드된 네트워크 사용
        network = AlphaZeroNet()
        network.load("./model/model_final.pth")
        agent = create_prediction_agent(network=network)
        
        # 3. 게임 플레이
        board = Board()
        action = agent.select_action(board, temperature=0)
    """
    if network is None:
        if model_path is None:
            raise ValueError("model_path 또는 network 중 하나는 필수입니다")
        
        # 모델 로드
        network = AlphaZeroNet()
        network.load(model_path)
        print(f"✓ Model loaded from {model_path}")
    
    # DTW Calculator 생성 (endgame threshold=15 고정)
    dtw_calculator = DTWCalculator(
        use_cache=True,
        endgame_threshold=15,
        midgame_threshold=45,
        shallow_depth=8
    )
    
    # DTW 캐시 로드 (있으면)
    if dtw_cache_path:
        try:
            dtw_calculator.tt.load_from_file(dtw_cache_path)
            print(f"✓ DTW cache loaded from {dtw_cache_path}")
        except FileNotFoundError:
            print(f"⚠ DTW cache not found: {dtw_cache_path}")
    
    # DTW 활성화된 Agent 생성
    agent = AlphaZeroAgent(
        network=network,
        num_simulations=num_simulations,
        temperature=temperature,
        dtw_calculator=dtw_calculator
    )
    
    print(f"✓ DTW enabled (endgame ≤15 cells)")
    
    return agent
