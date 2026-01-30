"""모델 비교 시뮬레이션 관리"""
import time
from game import Board
from ai.agent import AlphaZeroAgent


class CompareManager:
    """모델 비교 시뮬레이션을 담당하는 매니저 클래스"""
    
    def __init__(self, ui):
        """
        Args:
            ui: UI 인스턴스 참조
        """
        self.ui = ui
    
    def run_simulation(self):
        """모델 비교 시뮬레이션 실행"""
        try:
            print("Starting model comparison simulation...")
            
            # 모델 경로
            model1_path = self.ui.available_models[self.ui.compare_model1_idx]
            model2_path = self.ui.available_models[self.ui.compare_model2_idx]
            
            print(f"Loading Model 1: {model1_path}")
            print(f"Loading Model 2: {model2_path}")
            
            # 에이전트 생성
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            from ai.network import AlphaZeroNet
            
            # Model 1 로드
            network1 = AlphaZeroNet(device=device)
            network1.load(model1_path)
            agent1 = AlphaZeroAgent(network1, num_simulations=self.ui.compare_simulations)
            print("Model 1 loaded successfully")
            
            # Model 2 로드
            network2 = AlphaZeroNet(device=device)
            network2.load(model2_path)
            agent2 = AlphaZeroAgent(network2, num_simulations=self.ui.compare_simulations)
            print("Model 2 loaded successfully")
            
            # 게임 결과 추적
            model1_wins = 0
            model2_wins = 0
            draws = 0
            
            # 게임 상세 기록
            self.ui.game_details = []
            
            print(f"Starting {self.ui.compare_num_games} games...")
            
            for game_num in range(self.ui.compare_num_games):
                if self.ui.simulation_cancelled:
                    print("Simulation cancelled by user")
                    break
                
                # 진행 상황 업데이트
                self.ui.simulation_progress = (game_num / self.ui.compare_num_games) * 100
                self.ui.current_game_num = game_num + 1
                
                # 새 보드 생성
                board = Board()
                start_time = time.time()
                
                # 절반은 Model 1 선공, 나머지 절반은 Model 2 선공
                if game_num < self.ui.compare_num_games // 2:
                    # Model 1: Player 1 (Blue), Model 2: Player 2 (Red)
                    current_agent = agent1
                    other_agent = agent2
                    model1_player = 1
                    first_player = "Model 1"
                    print(f"Game {game_num + 1}: Model 1 goes first (Player 1)")
                else:
                    # Model 2: Player 1 (Blue), Model 1: Player 2 (Red)
                    current_agent = agent2
                    other_agent = agent1
                    model1_player = 2
                    first_player = "Model 2"
                    print(f"Game {game_num + 1}: Model 2 goes first (Player 1), Model 1 is Player 2")
                
                # 게임 진행
                move_count = 0
                max_moves = 81
                move_history = []
                
                while not board.is_game_over() and move_count < max_moves:
                    # 현재 에이전트가 수를 선택
                    action = current_agent.select_action(board, temperature=self.ui.compare_temperature)
                    
                    if action is not None:
                        row = action // 9
                        col = action % 9
                        move_history.append((row, col))
                        board.make_move(row, col)
                        move_count += 1
                    else:
                        break
                    
                    # 에이전트 교체
                    current_agent, other_agent = other_agent, current_agent
                
                # 결과 집계
                board.check_winner()
                winner = board.winner
                elapsed_time = time.time() - start_time
                
                # 승자 결정
                if winner is None or winner == 0:
                    draws += 1
                    winner_name = "Draw"
                elif winner == 3:
                    draws += 1
                    winner_name = "Draw"
                elif winner == model1_player:
                    model1_wins += 1
                    winner_name = "Model 1"
                else:
                    model2_wins += 1
                    winner_name = "Model 2"
                
                # 게임 상세 정보 저장
                game_detail = {
                    'game_num': game_num + 1,
                    'winner': winner_name,
                    'moves': move_count,
                    'time': elapsed_time,
                    'first_player': first_player,
                    'move_history': move_history
                }
                self.ui.game_details.append(game_detail)
                
                print(f"Game {game_num + 1}/{self.ui.compare_num_games} completed. Winner: {winner_name}")
            
            # 최종 결과
            if not self.ui.simulation_cancelled:
                self.ui.compare_results = {
                    'model1_wins': model1_wins,
                    'model2_wins': model2_wins,
                    'draws': draws,
                    'total_games': self.ui.compare_num_games
                }
                print("Simulation completed successfully!")
            
            self.ui.simulation_running = False
            self.ui.simulation_cancelled = False
            
        except Exception as e:
            print(f"Simulation error: {e}")
            import traceback
            traceback.print_exc()
            self.ui.simulation_running = False
            self.ui.simulation_cancelled = False
