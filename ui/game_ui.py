"""Enhanced Game UI - 메인 클래스 (모듈화됨)"""
import pygame
import sys
import os
import glob
import torch
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from game import Board
from ai.network import AlphaZeroNet
from ai.agent import AlphaZeroAgent
from ui.game_constants import GameMode, UIConstants, Colors
from ui.game_renderer import GameRenderer
from ui.menu_renderer import MenuRenderer
from ui.utils.game_history import GameHistory
from ui.components.move_list_panel import MoveListPanel
from ui.components.review_controls import ReviewControls
from ui.components.compare_renderer import CompareRenderer
from ui.managers.compare_manager import CompareManager

class UI:
    def __init__(self):
        pygame.init()
        
        # Window dimensions (고정 크기)
        self.WINDOW_WIDTH = 1400
        self.WINDOW_HEIGHT = 1000
        
        # Initialize display
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        self.font_tiny = pygame.font.Font(None, 18)
        
        # Game state
        self.mode = GameMode.MENU
        self.board = None
        self.running = True
        self.game_over = False
        self.move_history = []
        
        # AI state
        self.ai_agent = None
        self.ai_network = None
        self.ai_player = 2
        self.num_simulations = 100
        self.available_models = []
        self.selected_model_idx = 0
        self.player_goes_first = True  # True: 플레이어 선공, False: AI 선공
        
        # Analysis state
        self.show_analysis = False
        self.top_n_moves = 5
        self.analysis_data = None
        
        # Menu/UI state
        self.selecting_model = False
        self.selected_mode_for_model = None
        self.menu_buttons = []
        self.model_buttons = []
        self.slider_rect = None
        self.slider_handle_rect = None
        self.dragging_slider = False
        self.dragging_compare_games = False
        self.dragging_compare_sims = False
        self.dragging_compare_temp = False
        self.loading_game = False
        self.saved_game_buttons = []
        self.saved_games_list = []
        
        # Game history & Review
        self.game_history = None
        self.move_list_panel = None
        self.review_controls = None
        self.mode_name = 'MENU'
        
        # Compare Models state
        self.comparing_models = False
        self.compare_model1_idx = 0
        self.compare_model2_idx = 1 if len(self.available_models) > 1 else 0
        self.compare_num_games = 50
        self.compare_simulations = 100
        self.compare_temperature = 0.5  # Temperature for comparison games
        self.model1_scroll_offset = 0
        self.model2_scroll_offset = 0
        self.game_details = []
        self.viewing_game_detail = None
        self.simulation_progress = 0
        self.current_game_num = 0
        self.simulation_running = False
        self.simulation_cancelled = False
        self.review_model1_name = None  # Review 모드에서 Model 1 이름
        self.review_model2_name = None  # Review 모드에서 Model 2 이름
        self.review_first_player = None  # Review 모드에서 선공
        self.from_comparison = False  # Review 모드가 비교 결과에서 왔는지 여부
        self.compare_results = None
        
        # Renderers & Managers
        self.game_renderer = GameRenderer(self)
        self.menu_renderer = MenuRenderer(self)
        self.compare_renderer = CompareRenderer(self)
        self.move_list_panel = MoveListPanel(self)
        self.review_controls = ReviewControls(self)
        self.compare_manager = CompareManager(self)  # 모델 비교 시뮬레이션 매니저
        
        self.setup_menu()
    
    def setup_menu(self):
        """메뉴 화면 설정"""
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Ultimate Tic-Tac-Toe")
        
        self.find_available_models()
        self.create_menu_buttons()
    
    def create_menu_buttons(self):
        """메뉴 버튼 생성"""
        button_width = 400
        button_height = 50
        button_x = (self.WINDOW_WIDTH - button_width) // 2
        start_y = 180
        spacing = 70
        
        modes = [
            ("Single Player", GameMode.SINGLE),
            ("Play vs AI", GameMode.AI),
            ("Analysis Mode", GameMode.ANALYSIS),
            ("Load Game", GameMode.REVIEW),
            ("Compare Models", GameMode.COMPARE)
        ]
        
        self.menu_buttons = []
        for i, (label, mode) in enumerate(modes):
            y = start_y + i * spacing
            rect = pygame.Rect(button_x, y, button_width, button_height)
            self.menu_buttons.append((rect, mode, label))
    
    def find_available_models(self):
        """저장된 모델 파일 찾기"""
        model_dir = "./model"
        if os.path.exists(model_dir):
            models = glob.glob(os.path.join(model_dir, "*.pth"))
            self.available_models = sorted(models)
        else:
            self.available_models = []
    
    def create_model_buttons(self):
        """모델 선택 버튼 생성"""
        self.model_buttons = []
        center_x = self.WINDOW_WIDTH // 2
        
        if not self.available_models:
            back_rect = pygame.Rect(center_x - 100, 400, 200, 50)
            self.model_buttons = [(back_rect, "back")]
        else:
            start_y = 120
            button_height = 40
            button_width = 500
            spacing = 50
            
            # 모델 버튼들
            for i, model_path in enumerate(self.available_models[:8]):
                y = start_y + i * spacing
                x = center_x - button_width // 2
                rect = pygame.Rect(x, y, button_width, button_height)
                self.model_buttons.append((rect, i))
            
            # Back 버튼과 Start 버튼을 맨 아래로 (720)
            button_y = 720
            button_width_small = 150
            total_width = button_width_small * 2 + 100  # 두 버튼 + 간격
            
            # Back 버튼 (왼쪽)
            back_x = center_x - total_width // 2
            back_rect = pygame.Rect(back_x, button_y, button_width_small, 50)
            self.model_buttons.append((back_rect, "back"))
            
            # Start 버튼 (오른쪽)
            start_x = center_x + total_width // 2 - button_width_small
            start_rect = pygame.Rect(start_x, button_y, button_width_small, 50)
            self.model_buttons.append((start_rect, "start"))
    
    def load_ai_model(self, model_path):
        """AI 모델 로드"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.ai_network = AlphaZeroNet(device=device)
            self.ai_network.load(model_path)
            self.ai_agent = AlphaZeroAgent(
                self.ai_network, 
                num_simulations=self.num_simulations
            )
            return True
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return False
    
    def start_mode(self, mode, model_path=None):
        """게임 모드 시작"""
        self.mode = mode
        self.board = Board()
        self.game_over = False
        self.move_history = []
        
        # 게임 히스토리 초기화
        initial_state = self._get_board_state()
        self.game_history = GameHistory(initial_state)
        
        # 모드명 설정
        if mode == GameMode.SINGLE:
            self.mode_name = 'SINGLE'
        elif mode == GameMode.AI:
            self.mode_name = 'AI'
        elif mode == GameMode.ANALYSIS:
            self.mode_name = 'ANALYSIS'
        
        # AI 모드일 경우 모델 로드
        if mode in [GameMode.AI, GameMode.ANALYSIS] and model_path:
            if not self.load_ai_model(model_path):
                print("모델 로드 실패, Single 모드로 전환")
                self.mode = GameMode.SINGLE
                self.ai_agent = None
            else:
                # AI 선공/후공 설정
                if mode == GameMode.AI:
                    if self.player_goes_first:
                        self.ai_player = 2  # AI는 Player 2 (Red)
                    else:
                        self.ai_player = 1  # AI는 Player 1 (Blue)
                        # AI가 먼저 두도록 즉시 수를 둠
                        self.ai_make_move()
        
        self.show_analysis = (mode == GameMode.ANALYSIS)
    
    def _get_board_state(self):
        """현재 보드 상태를 딕셔너리로 반환"""
        return {
            'boards': [row[:] for row in self.board.boards],
            'completed_boards': [row[:] for row in self.board.completed_boards],
            'current_player': self.board.current_player,
            'last_move': self.board.last_move,
            'winner': self.board.winner
        }
    
    def _restore_board_state(self, state):
        """보드 상태 복원"""
        # boards와 completed_boards를 깊은 복사로 복원
        import copy
        self.board.boards = copy.deepcopy(state['boards'])
        self.board.completed_boards = copy.deepcopy(state['completed_boards'])
        self.board.current_player = state['current_player']
        self.board.last_move = tuple(state['last_move']) if state['last_move'] else None
        self.board.winner = state['winner']
    
    def get_cell_from_pos(self, pos):
        """마우스 위치를 보드 좌표로 변환"""
        x, y = pos
        # game_renderer의 board offset 사용 (get_cell_center와 동일)
        board_offset_x = self.game_renderer.get_board_offset_x()
        x -= UIConstants.MARGIN + board_offset_x
        y -= UIConstants.MARGIN + UIConstants.INFO_HEIGHT
        
        if x < 0 or y < 0 or x >= UIConstants.BOARD_SIZE or y >= UIConstants.BOARD_SIZE:
            return None
        
        col = x // UIConstants.CELL_SIZE
        row = y // UIConstants.CELL_SIZE
        return (row, col)
    
    def get_cell_center(self, row, col):
        """셀 중심 위치"""
        # game_renderer의 board offset 사용 (가운데 정렬 포함)
        board_offset_x = self.game_renderer.get_board_offset_x()
        
        x = UIConstants.MARGIN + board_offset_x + col * UIConstants.CELL_SIZE + UIConstants.CELL_SIZE // 2
        y = UIConstants.MARGIN + UIConstants.INFO_HEIGHT + row * UIConstants.CELL_SIZE + UIConstants.CELL_SIZE // 2
        return (x, y)
    
    def calculate_analysis(self):
        """AI 분석 계산"""
        if not self.ai_agent:
            return
        
        # Review 모드는 game_over 상관없이 분석
        if self.mode != GameMode.REVIEW and self.game_over:
            return
        
        try:
            policy, value = self.ai_network.predict(self.board)
            legal_moves = self.board.get_legal_moves()
            
            move_probs = []
            for move in legal_moves:
                row, col = move
                idx = row * 9 + col
                if idx < len(policy):
                    move_probs.append((move, policy[idx]))
            
            move_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Top 5 moves에 대한 continuation 계산 (3수까지)
            top_moves_with_continuation = []
            for move, prob in move_probs[:5]:
                continuation = self._calculate_continuation(move, depth=3)
                top_moves_with_continuation.append((move, prob, continuation))
            
            self.analysis_data = {
                'value': value,
                'top_moves': move_probs,
                'top_moves_with_continuation': top_moves_with_continuation
            }
        except Exception as e:
            print(f"분석 오류: {e}")
            self.analysis_data = None
    
    def _calculate_continuation(self, first_move, depth=3):
        """특정 수 이후의 continuation(이어지는 기보) 계산"""
        if not self.ai_agent or depth <= 0:
            return []
        
        try:
            # 현재 보드 상태 저장
            from copy import deepcopy
            temp_board = deepcopy(self.board)
            
            continuation = [first_move]
            
            # 첫 수를 둠
            row, col = first_move
            temp_board.make_move(row, col)
            
            # 이어지는 수들을 계산
            for _ in range(depth - 1):
                if temp_board.is_game_over():
                    break
                
                # AI가 다음 수 예측
                policy, _ = self.ai_network.predict(temp_board)
                legal_moves = temp_board.get_legal_moves()
                
                if not legal_moves:
                    break
                
                # 최선의 수 선택
                best_move = None
                best_prob = -1
                for move in legal_moves:
                    r, c = move
                    idx = r * 9 + c
                    if idx < len(policy) and policy[idx] > best_prob:
                        best_prob = policy[idx]
                        best_move = move
                
                if best_move:
                    continuation.append(best_move)
                    r, c = best_move
                    temp_board.make_move(r, c)
                else:
                    break
            
            return continuation
        except Exception as e:
            print(f"Continuation 계산 오류: {e}")
            return [first_move]
    
    def save_board_state(self):
        """현재 보드 상태 저장 (Undo용)"""
        state = {
            'boards': [row[:] for row in self.board.boards],
            'completed_boards': [row[:] for row in self.board.completed_boards],
            'current_player': self.board.current_player,
            'last_move': self.board.last_move,
            'winner': self.board.winner
        }
        self.move_history.append(state)
    
    def undo_move(self):
        """이전 수로 되돌리기"""
        # Analysis 모드는 히스토리 기반으로 동작
        if self.mode == GameMode.ANALYSIS:
            if self.game_history and self.game_history.go_to_parent():
                self._restore_board_state(self.game_history.current_node.board_state)
                self.analysis_data = None
                self.game_over = False
            return
        
        # Single 모드는 기존 방식
        if not self.move_history or self.game_over:
            return
        
        last_state = self.move_history.pop()
        self.board.boards = [row[:] for row in last_state['boards']]
        self.board.completed_boards = [row[:] for row in last_state['completed_boards']]
        self.board.current_player = last_state['current_player']
        self.board.last_move = last_state['last_move']
        self.board.winner = last_state['winner']
    
    def handle_click(self, pos):
        """클릭 처리"""
        if self.game_over:
            return
        
        if self.mode == GameMode.AI and self.board.current_player == self.ai_player:
            return
        
        cell = self.get_cell_from_pos(pos)
        if cell is None:
            return
        
        row, col = cell
        
        try:
            self.save_board_state()
            self.board.make_move(row, col)
            
            # 히스토리에 수 기록
            if self.game_history:
                new_state = self._get_board_state()
                self.game_history.add_move((row, col), new_state)
            
            # 게임이 끝나면 game_over 설정
            if self.board.winner is not None:
                self.game_over = True
                self.on_game_end()
            elif self.mode == GameMode.AI:
                # 플레이어 수를 즉시 렌더링
                self.render_game()
                pygame.display.flip()
                
                # AI 차례
                self.ai_make_move()
        except ValueError:
            if self.move_history:
                self.move_history.pop()
    
    def handle_analysis_board_click(self, pos):
        """Analysis 모드에서 보드 클릭 - Variation 생성"""
        cell = self.get_cell_from_pos(pos)
        if cell is None:
            return
        
        row, col = cell
        
        try:
            # 보드 상태 저장
            self.save_board_state()
            
            # 수를 둠
            self.board.make_move(row, col)
            
            # 히스토리에 새 변화 추가
            if self.game_history:
                new_state = self._get_board_state()
                self.game_history.add_move((row, col), new_state)
            
            # 분석 데이터 초기화 (새로운 위치에서 다시 분석)
            self.analysis_data = None
            
            # 게임 종료 체크
            if self.board.winner is not None:
                self.game_over = True
                self.on_game_end()
                
        except ValueError as e:
            # 불가능한 수
            print(f"Invalid move: {e}")
            if self.move_history:
                self.move_history.pop()
    
    def handle_review_board_click(self, pos):
        """Review 모드에서 보드 클릭 - Variation 생성"""
        cell = self.get_cell_from_pos(pos)
        if cell is None:
            return
        
        row, col = cell
        
        try:
            # 수를 둠
            self.board.make_move(row, col)
            
            # 히스토리에 새 변화 추가
            if self.game_history:
                new_state = self._get_board_state()
                self.game_history.add_move((row, col), new_state)
            
            # 분석 데이터 초기화 (새로운 위치에서 다시 분석)
            if self.show_analysis:
                self.analysis_data = None
                
        except ValueError as e:
            # 불가능한 수
            print(f"Invalid move: {e}")
            pass
    
    def ai_make_move(self):
        """AI가 수를 둠"""
        if not self.ai_agent or self.game_over:
            return
        
        # AI 생각 중 표시
        self.draw_ai_thinking()
        pygame.display.flip()
        pygame.event.pump()  # 이벤트 처리하여 화면이 멈추지 않도록
        
        try:
            action = self.ai_agent.select_action(self.board, temperature=0)
            if action is not None:
                row = action // 9
                col = action % 9
                self.board.make_move(row, col)
                
                # AI 수도 히스토리에 기록
                if self.game_history:
                    new_state = self._get_board_state()
                    self.game_history.add_move((row, col), new_state)
                
                if self.board.winner is not None:
                    self.game_over = True
                    self.on_game_end()
        except Exception as e:
            print(f"AI 이동 오류: {e}")
    
    def on_game_end(self):
        """게임 종료 시 호출"""
        if self.game_history and self.mode in [GameMode.SINGLE, GameMode.AI, GameMode.ANALYSIS]:
            # 게임 정보 업데이트
            self.game_history.game_info['result'] = f"Player {self.board.winner} wins" if self.board.winner else "Draw"
            self.game_history.game_info['mode'] = self.mode_name
            
            # 자동 저장
            self.save_current_game()
    
    def save_current_game(self):
        """현재 게임을 파일로 저장"""
        if not self.game_history:
            return
        
        # saved_games 폴더 생성
        save_dir = "./saved_games"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"game_{self.mode_name}_{timestamp}.json"
        filepath = os.path.join(save_dir, filename)
        
        try:
            self.game_history.save_to_file(filepath)
            print(f"게임 저장됨: {filepath}")
        except Exception as e:
            print(f"게임 저장 실패: {e}")
    
    def start_review_mode(self, game_filepath, model_path=None):
        """Review 모드로 게임 로드 (Analysis 기능 포함)"""
        try:
            # 게임 히스토리 로드
            self.game_history = GameHistory.load_from_file(game_filepath)
            
            # 모드 설정
            self.mode = GameMode.REVIEW
            self.mode_name = 'REVIEW'
            self.board = Board()
            self.game_over = False
            
            # AI 모델 로드 (Analysis 기능)
            if model_path:
                if self.load_ai_model(model_path):
                    self.show_analysis = True
                else:
                    print("AI 모델 로드 실패, 분석 기능 없이 진행")
                    self.show_analysis = False
            else:
                self.show_analysis = False
            
            # 시작 위치로 이동
            self.game_history.go_to_start()
            self._restore_board_state(self.game_history.current_node.board_state)
            
            return True
        except Exception as e:
            print(f"게임 로드 실패: {e}")
            return False
    
    def _load_comparison_game(self, game_detail):
        """비교 게임을 리뷰 모드로 로드"""
        try:
            from ui.utils.game_history import GameHistory
            
            # 새 보드 생성
            board = Board()
            
            # GameHistory 초기화
            initial_state = self._board_to_state(board)
            self.game_history = GameHistory(initial_state)
            
            # move_history 재생 (히스토리 빌드용)
            for row, col in game_detail['move_history']:
                board.make_move(row, col)
                new_state = self._board_to_state(board)
                self.game_history.add_move((row, col), new_state)
            
            # 플레이어 정보 저장
            self.review_model1_name = "Model 1"
            self.review_model2_name = "Model 2"
            self.review_first_player = game_detail.get('first_player', 'Unknown')
            
            # 리뷰 모드로 전환
            self.mode = GameMode.REVIEW
            self.mode_name = 'REVIEW'  # mode_name도 설정!
            self.from_comparison = True  # 비교 결과에서 왔음을 표시
            self.comparing_models = False  # 비교 화면에서 벗어남
            self.board = Board()  # 항상 새 보드 생성!
            self.game_over = False
            self.show_analysis = False
            
            # 시작 위치로 이동하고 상태 복원
            self.game_history.go_to_start()
            self._restore_board_state(self.game_history.current_node.board_state)
            
            print(f"Loaded comparison game #{game_detail['game_num']} for review")
            
        except Exception as e:
            print(f"Failed to load comparison game: {e}")
            import traceback
            traceback.print_exc()
    
    def _board_to_state(self, board):
        """Board 객체를 state dictionary로 변환"""
        return {
            'boards': [row[:] for row in board.boards],
            'completed_boards': [row[:] for row in board.completed_boards],
            'current_player': board.current_player,
            'last_move': board.last_move,
            'winner': board.winner
        }
    
    def get_saved_games(self):
        """저장된 게임 목록 반환"""
        save_dir = "./saved_games"
        if not os.path.exists(save_dir):
            return []
        
        games = []
        for filename in os.listdir(save_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(save_dir, filename)
                games.append(filepath)
        
        return sorted(games, reverse=True)  # 최신 순
    
    def restart_game(self):
        """게임 재시작"""
        self.board = Board()
        self.game_over = False
        self.analysis_data = None
        self.move_history = []
        
        # 히스토리도 재시작
        if self.mode != GameMode.REVIEW:
            initial_state = self._get_board_state()
            self.game_history = GameHistory(initial_state)
    
    def render_game(self):
        """게임 화면 렌더링"""
        self.screen.fill(Colors.BG_COLOR)
        self.game_renderer.draw_info_bar()
        
        if self.show_analysis:
            self.game_renderer.draw_eval_bar()
        
        self.game_renderer.draw_grid()
        self.game_renderer.draw_last_move_highlight()
        
        # Legal moves highlight (Analysis 모드는 게임 종료 후에도 표시)
        if self.mode == GameMode.ANALYSIS:
            self.game_renderer.draw_legal_moves_highlight()
            if self.show_analysis:
                self.game_renderer.draw_ai_suggestion()
        elif not self.game_over and self.mode != GameMode.REVIEW:
            self.game_renderer.draw_legal_moves_highlight()
            if self.show_analysis:
                self.game_renderer.draw_ai_suggestion()
        
        self.game_renderer.draw_completed_boards()
        self.game_renderer.draw_marks()
        
        if self.show_analysis:
            self.game_renderer.draw_analysis_panel()
        
        # Review 및 Analysis 모드 UI
        if self.mode == GameMode.REVIEW:
            self.move_list_panel.draw(self.game_history)
            self.review_controls.draw(self.game_history)
        elif self.mode == GameMode.ANALYSIS:
            # Analysis 모드에도 Move List와 네비게이션 버튼 표시
            self.move_list_panel.draw(self.game_history)
            self.review_controls.draw(self.game_history)
        
        if self.game_over and self.mode != GameMode.REVIEW:
            self.game_renderer.draw_winner_overlay()
    
    def draw_ai_thinking(self):
        """AI 생각 중 표시"""
        # 반투명 오버레이
        overlay = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # "AI Thinking..." 텍스트
        text = self.font_medium.render("AI Thinking...", True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.WINDOW_WIDTH // 2, self.WINDOW_HEIGHT // 2))
        self.screen.blit(text, text_rect)
    
    def update_slider_value(self, mouse_x):
        """슬라이더 값 업데이트"""
        if not self.slider_rect:
            return
        
        # 마우스 위치를 슬라이더 범위 내로 제한
        slider_x = self.slider_rect.x
        slider_width = self.slider_rect.width
        
        relative_x = max(0, min(mouse_x - slider_x, slider_width))
        normalized = relative_x / slider_width
        
        # 값 범위: 10 ~ 500
        min_val = 10
        max_val = 500
        self.num_simulations = int(min_val + normalized * (max_val - min_val))
        
        # AI 에이전트가 이미 로드되어 있으면 시뮬레이션 수 업데이트
        if self.ai_agent:
            self.ai_agent.num_simulations = self.num_simulations
    
    def run(self):
        """메인 루프"""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        if self.selecting_model:
                            # 모델 선택 화면
                            turn_order_clicked = False
                            
                            # Turn Order 버튼 클릭 체크
                            if hasattr(self, 'turn_order_buttons'):
                                for rect, order in self.turn_order_buttons:
                                    if rect.collidepoint(event.pos):
                                        self.player_goes_first = (order == "You First")
                                        turn_order_clicked = True
                                        break
                            
                            if self.slider_rect and (self.slider_rect.collidepoint(event.pos) or 
                                                     self.slider_handle_rect.collidepoint(event.pos)):
                                self.dragging_slider = True
                                self.update_slider_value(event.pos[0])
                            elif not turn_order_clicked:
                                for rect, action in self.model_buttons:
                                    if rect.collidepoint(event.pos):
                                        if action == "back":
                                            self.selecting_model = False
                                            self.mode = GameMode.MENU
                                            self.setup_menu()
                                        elif action == "start":
                                            if self.available_models:
                                                model_path = self.available_models[self.selected_model_idx]
                                                if self.selected_mode_for_model == GameMode.REVIEW:
                                                    # Review 모드: 모델 선택 후 게임 목록으로
                                                    self.selecting_model = False
                                                    self.loading_game = True
                                                    self.saved_games_list = self.get_saved_games()
                                                    self.selected_model_path = model_path
                                                else:
                                                    self.start_mode(self.selected_mode_for_model, model_path)
                                                    self.selecting_model = False
                                        elif isinstance(action, int):
                                            self.selected_model_idx = action
                                        break
                        
                        elif self.loading_game:
                            # 저장된 게임 목록에서 선택
                            for rect, action in self.saved_game_buttons:
                                if rect.collidepoint(event.pos):
                                    if action == "back":
                                        self.loading_game = False
                                        self.mode = GameMode.MENU
                                        self.setup_menu()
                                    elif isinstance(action, str) and action.endswith('.json'):
                                        # 게임 로드 (선택한 모델과 함께)
                                        model_path = getattr(self, 'selected_model_path', None)
                                        if self.start_review_mode(action, model_path):
                                            self.loading_game = False
                                    break
                        
                        elif self.comparing_models:
                            # Compare Models 화면 클릭 처리
                            
                            # 시뮬레이션 실행 중 Cancel 버튼
                            if self.simulation_running:
                                center_x = self.WINDOW_WIDTH // 2
                                center_y = self.WINDOW_HEIGHT // 2
                                button_width = 120
                                button_height = 40
                                cancel_button_rect = pygame.Rect(center_x - button_width // 2, center_y + 80, button_width, button_height)
                                
                                if cancel_button_rect.collidepoint(event.pos):
                                    self.simulation_cancelled = True
                                    print("Cancel button clicked - stopping simulation...")
                            
                            elif self.compare_results:
                                # 결과 화면
                                center_x = self.WINDOW_WIDTH // 2
                                
                                # Back 버튼
                                back_button = pygame.Rect(center_x - 75, 700, 150, 50)
                                if back_button.collidepoint(event.pos):
                                    self.comparing_models = False
                                    self.compare_results = None
                                    self.mode = GameMode.MENU
                                    self.setup_menu()
                                else:
                                    # 게임 리스트 클릭 처리
                                    list_start_y = 205  # 게임 리스트 시작 y 좌표 (렌더러와 동일)
                                    item_height = 30
                                    visible_games = 12
                                    
                                    for i, game in enumerate(self.game_details[:visible_games]):
                                        y = list_start_y + i * item_height
                                        game_rect = pygame.Rect(center_x - 400, y - 10, 800, item_height)
                                        
                                        if game_rect.collidepoint(event.pos) and 'move_history' in game:
                                            # 게임 리뷰 모드로 전환
                                            self._load_comparison_game(game)
                                            break
                            else:
                                # 설정 화면
                                center_x = self.WINDOW_WIDTH // 2
                                
                                # 모델이 2개 이상인지 확인
                                if not self.available_models or len(self.available_models) < 2:
                                    back_rect = pygame.Rect(center_x - 75, 400, 150, 50)
                                    if back_rect.collidepoint(event.pos):
                                        self.comparing_models = False
                                        self.mode = GameMode.MENU
                                        self.setup_menu()
                                else:
                                    # 좌우 2열 레이아웃 좌표 (margin-div-gap-div-margin)
                                    list_width = 250
                                    gap = 40
                                    total_content_width = list_width * 2 + gap
                                    start_x = (self.WINDOW_WIDTH - total_content_width) // 2
                                    
                                    left_x = start_x + list_width // 2
                                    right_x = start_x + list_width + gap + list_width // 2
                                    
                                    list_y = 100
                                    list_height = 300
                                    item_height = 40
                                    
                                    # 버튼 위치 계산 (먼저)
                                    settings_y = list_y + list_height + 20
                                    slider_width = 300
                                    slider_x = center_x - slider_width // 2
                                    games_slider_y = settings_y + 30
                                    sims_slider_y = games_slider_y + 80
                                    temp_slider_y_calc = sims_slider_y + 80  # Temperature 슬라이더 위치
                                    button_y = temp_slider_y_calc + 50  # Temperature 슬라이더 다음
                                    button_width_small = 150
                                    total_width = button_width_small * 2 + 100
                                    back_x = center_x - total_width // 2
                                    start_x = center_x + total_width // 2 - button_width_small
                                    
                                    # 버튼 Rect 정의
                                    back_button = pygame.Rect(back_x, button_y, button_width_small, 50)
                                    start_button = pygame.Rect(start_x, button_y, button_width_small, 50)
                                    
                                    # 1순위: 슬라이더 체크 (먼저 - 화면 위쪽에 있으므로)
                                    slider_height = 8
                                    games_slider = pygame.Rect(slider_x, games_slider_y, slider_width, slider_height)
                                    sims_slider = pygame.Rect(slider_x, sims_slider_y, slider_width, slider_height)
                                    
                                    # 슬라이더 핸들 영역
                                    handle_radius = 12
                                    min_games = 10
                                    max_games = 100
                                    normalized_games = (self.compare_num_games - min_games) / (max_games - min_games)
                                    handle_x_games = slider_x + int(normalized_games * slider_width)
                                    games_handle = pygame.Rect(handle_x_games - handle_radius, games_slider_y - handle_radius + slider_height // 2,
                                                               handle_radius * 2, handle_radius * 2)
                                    
                                    min_sims = 10
                                    max_sims = 500
                                    normalized_sims = (self.compare_simulations - min_sims) / (max_sims - min_sims)
                                    handle_x_sims = slider_x + int(normalized_sims * slider_width)
                                    sims_handle = pygame.Rect(handle_x_sims - handle_radius, sims_slider_y - handle_radius + slider_height // 2,
                                                             handle_radius * 2, handle_radius * 2)
                                    
                                    # Temperature 슬라이더
                                    temp_slider_y = sims_slider_y + 80
                                    temp_slider = pygame.Rect(slider_x, temp_slider_y, slider_width, slider_height)
                                    min_temp = 0.0
                                    max_temp = 2.0
                                    normalized_temp = (self.compare_temperature - min_temp) / (max_temp - min_temp)
                                    handle_x_temp = slider_x + int(normalized_temp * slider_width)
                                    temp_handle = pygame.Rect(handle_x_temp - handle_radius, temp_slider_y - handle_radius + slider_height // 2,
                                                             handle_radius * 2, handle_radius * 2)
                                    
                                    # 슬라이더 클릭 체크
                                    if games_slider.collidepoint(event.pos) or games_handle.collidepoint(event.pos):
                                        self.dragging_slider = True
                                        self.dragging_compare_games = True
                                        self.dragging_compare_sims = False
                                        self.dragging_compare_temp = False
                                        # 즉시 값 업데이트
                                        relative_x = max(0, min(event.pos[0] - slider_x, slider_width))
                                        normalized = relative_x / slider_width
                                        self.compare_num_games = int(min_games + normalized * (max_games - min_games))
                                    elif sims_slider.collidepoint(event.pos) or sims_handle.collidepoint(event.pos):
                                        self.dragging_slider = True
                                        self.dragging_compare_games = False
                                        self.dragging_compare_sims = True
                                        self.dragging_compare_temp = False
                                        # 즉시 값 업데이트
                                        relative_x = max(0, min(event.pos[0] - slider_x, slider_width))
                                        normalized = relative_x / slider_width
                                        self.compare_simulations = int(min_sims + normalized * (max_sims - min_sims))
                                    elif temp_slider.collidepoint(event.pos) or temp_handle.collidepoint(event.pos):
                                        self.dragging_slider = True
                                        self.dragging_compare_games = False
                                        self.dragging_compare_sims = False
                                        self.dragging_compare_temp = True
                                        # 즉시 값 업데이트
                                        relative_x = max(0, min(event.pos[0] - slider_x, slider_width))
                                        normalized = relative_x / slider_width
                                        self.compare_temperature = min_temp + normalized * (max_temp - min_temp)
                                    
                                    # 2순위: Back/Start 버튼
                                    elif back_button.collidepoint(event.pos):
                                        self.comparing_models = False
                                        self.mode = GameMode.MENU
                                        self.setup_menu()
                                    elif start_button.collidepoint(event.pos):
                                        # 시뮬레이션 시작
                                        print(f"Starting simulation: {self.compare_num_games} games with {self.compare_simulations} simulations")
                                        self.simulation_running = True
                                        import threading
                                        thread = threading.Thread(target=self.compare_manager.run_simulation)
                                        thread.daemon = True
                                        thread.start()
                                    
                                    # 3순위: 모델 리스트 클릭
                                    else:
                                        list1_rect = pygame.Rect(left_x - list_width // 2, list_y, list_width, list_height)
                                        list2_rect = pygame.Rect(right_x - list_width // 2, list_y, list_width, list_height)
                                        
                                        max_visible = list_height // item_height
                                        
                                        # Model 1 리스트 클릭
                                        if list1_rect.collidepoint(event.pos):
                                            visible_start1 = self.model1_scroll_offset
                                            for i in range(min(max_visible, len(self.available_models) - visible_start1)):
                                                actual_idx = visible_start1 + i
                                                item_y = list_y + i * item_height
                                                item_rect = pygame.Rect(left_x - list_width // 2 + 3, item_y + 3, list_width - 6, item_height - 6)
                                                
                                                if item_rect.collidepoint(event.pos):
                                                    self.compare_model1_idx = actual_idx
                                                    break
                                        
                                        # Model 2 리스트 클릭
                                        elif list2_rect.collidepoint(event.pos):
                                            visible_start2 = self.model2_scroll_offset
                                            for i in range(min(max_visible, len(self.available_models) - visible_start2)):
                                                actual_idx = visible_start2 + i
                                                item_y = list_y + i * item_height
                                                item_rect = pygame.Rect(right_x - list_width // 2 + 3, item_y + 3, list_width - 6, item_height - 6)
                                                
                                                if item_rect.collidepoint(event.pos):
                                                    self.compare_model2_idx = actual_idx
                                                    break
                        
                        elif self.mode == GameMode.MENU:
                            for rect, mode, label in self.menu_buttons:
                                if rect.collidepoint(event.pos):
                                    if mode in [GameMode.AI, GameMode.ANALYSIS]:
                                        self.selecting_model = True
                                        self.selected_mode_for_model = mode
                                        self.create_model_buttons()
                                    elif mode == GameMode.REVIEW:
                                        # Load Game: 먼저 모델 선택, 그 다음 게임 선택
                                        self.selecting_model = True
                                        self.selected_mode_for_model = GameMode.REVIEW
                                        self.create_model_buttons()
                                    elif mode == GameMode.COMPARE:
                                        # Compare Models 화면으로 전환
                                        self.comparing_models = True
                                        self.compare_results = None
                                    else:
                                        self.start_mode(mode)
                                    break
                        
                        elif self.mode == GameMode.ANALYSIS:
                            # Analysis 모드 클릭 처리
                            # 1. Move list 클릭 - 히스토리 이동
                            if self.move_list_panel.handle_click(event.pos, self.game_history):
                                self._restore_board_state(self.game_history.current_node.board_state)
                                self.analysis_data = None
                            # 2. Review controls 클릭 - 네비게이션
                            elif self.review_controls.handle_click(event.pos, self.game_history):
                                self._restore_board_state(self.game_history.current_node.board_state)
                                self.analysis_data = None
                            # 3. 보드 클릭 - Variation 생성
                            else:
                                self.handle_analysis_board_click(event.pos)
                        
                        elif self.mode == GameMode.REVIEW:
                            # Review 모드 클릭 처리
                            # Back 버튼 체크 (최우선)
                            if hasattr(self, 'back_button_rect') and self.back_button_rect.collidepoint(event.pos):
                                if self.from_comparison and self.compare_results:
                                    # 비교 결과 화면으로 복귀
                                    self.mode = GameMode.MENU
                                    self.from_comparison = False
                                    self.comparing_models = True  # 비교 결과 화면 표시
                                else:
                                    # 일반 저장 게임에서 온 경우 메뉴로
                                    self.mode = GameMode.MENU
                                    self.setup_menu()
                            # 1. Move list 클릭
                            elif self.move_list_panel.handle_click(event.pos, self.game_history):
                                self._restore_board_state(self.game_history.current_node.board_state)
                                # 분석 데이터 초기화 (다시 계산하도록)
                                if self.show_analysis:
                                    self.analysis_data = None
                            # 2. Review controls 클릭
                            elif self.review_controls.handle_click(event.pos, self.game_history):
                                self._restore_board_state(self.game_history.current_node.board_state)
                                # 분석 데이터 초기화
                                if self.show_analysis:
                                    self.analysis_data = None
                            # 3. 보드 클릭 - Variation 생성
                            else:
                                self.handle_review_board_click(event.pos)
                        
                        elif self.mode == GameMode.AI:
                            # AI 모드 클릭 처리
                            # Back 버튼 체크
                            if hasattr(self, 'back_button_rect') and self.back_button_rect.collidepoint(event.pos):
                                self.mode = GameMode.MENU
                                self.setup_menu()
                            else:
                                self.handle_click(event.pos)
                        
                        else:
                            self.handle_click(event.pos)
                
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging_slider = False
                
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging_slider:
                        if self.comparing_models and (self.dragging_compare_games or self.dragging_compare_sims or self.dragging_compare_temp):
                            # Compare Models 슬라이더 드래그
                            center_x = self.WINDOW_WIDTH // 2
                            slider_width = 400
                            slider_x = center_x - slider_width // 2
                            
                            relative_x = max(0, min(event.pos[0] - slider_x, slider_width))
                            normalized = relative_x / slider_width
                            
                            if self.dragging_compare_games:
                                min_games = 10
                                max_games = 100
                                self.compare_num_games = int(min_games + normalized * (max_games - min_games))
                            elif self.dragging_compare_sims:
                                min_sims = 10
                                max_sims = 500
                                self.compare_simulations = int(min_sims + normalized * (max_sims - min_sims))
                            elif self.dragging_compare_temp:
                                min_temp = 0.0
                                max_temp = 2.0
                                self.compare_temperature = min_temp + normalized * (max_temp - min_temp)
                        elif self.slider_rect:
                            self.update_slider_value(event.pos[0])
                
                elif event.type == pygame.MOUSEWHEEL:
                    # Compare Models 리스트에서 스크롤
                    if self.comparing_models and not self.compare_results:
                        mouse_pos = pygame.mouse.get_pos()
                        
                        # 레이아웃 계산 (margin-div-gap-div-margin)
                        list_width = 250
                        gap = 40
                        total_content_width = list_width * 2 + gap
                        start_x = (self.WINDOW_WIDTH - total_content_width) // 2
                        
                        left_x = start_x + list_width // 2
                        right_x = start_x + list_width + gap + list_width // 2
                        
                        list_y = 100
                        list_height = 300
                        item_height = 40
                        max_visible = list_height // item_height
                        
                        # Model 1 리스트 위에 마우스가 있는지 확인
                        list1_rect = pygame.Rect(left_x - list_width // 2, list_y, list_width, list_height)
                        if list1_rect.collidepoint(mouse_pos):
                            max_scroll = max(0, len(self.available_models) - max_visible)
                            self.model1_scroll_offset = max(0, min(self.model1_scroll_offset - event.y, max_scroll))
                        
                        # Model 2 리스트 위에 마우스가 있는지 확인
                        list2_rect = pygame.Rect(right_x - list_width // 2, list_y, list_width, list_height)
                        if list2_rect.collidepoint(mouse_pos):
                            max_scroll = max(0, len(self.available_models) - max_visible)
                            self.model2_scroll_offset = max(0, min(self.model2_scroll_offset - event.y, max_scroll))
                    # Move List 패널에서 스크롤
                    elif self.mode in [GameMode.ANALYSIS, GameMode.REVIEW]:
                        mouse_pos = pygame.mouse.get_pos()
                        if self.move_list_panel.is_over_panel(mouse_pos):
                            self.move_list_panel.handle_scroll(event.y)
                
                elif event.type == pygame.KEYDOWN:
                    if event.scancode == 21:  # R key
                        if self.mode != GameMode.MENU and not self.selecting_model:
                            self.restart_game()
                    elif event.scancode == 16:  # M key
                        if not self.selecting_model:
                            self.comparing_models = False
                            self.loading_game = False
                            self.mode = GameMode.MENU
                            self.setup_menu()
                    elif event.scancode == 24:  # U key
                        if self.mode in [GameMode.SINGLE, GameMode.ANALYSIS] and not self.selecting_model:
                            self.undo_move()
                    elif event.key == pygame.K_ESCAPE:
                        self.running = False
            
            # 그리기
            self.screen.fill(Colors.BG_COLOR)
            
            if self.selecting_model:
                self.menu_renderer.draw_model_selection()
            elif self.loading_game:
                self.menu_renderer.draw_saved_games_list()
            elif self.comparing_models:
                if self.compare_results:
                    self.compare_renderer.draw_results()
                else:
                    self.compare_renderer.draw_settings()
                    # 시뮬레이션 실행 중 표시
                    if self.simulation_running:
                        overlay = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
                        overlay.set_alpha(200)
                        overlay.fill((0, 0, 0))
                        self.screen.blit(overlay, (0, 0))
                        
                        center_x = self.WINDOW_WIDTH // 2
                        center_y = self.WINDOW_HEIGHT // 2
                        
                        # 제목
                        text = self.font_medium.render("Running Simulation...", True, (255, 255, 255))
                        text_rect = text.get_rect(center=(center_x, center_y - 60))
                        self.screen.blit(text, text_rect)
                        
                        # 현재 게임 번호
                        game_text = self.font_small.render(f"Game {self.current_game_num} / {self.compare_num_games}", True, (255, 255, 255))
                        game_text_rect = game_text.get_rect(center=(center_x, center_y - 20))
                        self.screen.blit(game_text, game_text_rect)
                        
                        # Progress Bar
                        bar_width = 400
                        bar_height = 30
                        bar_x = center_x - bar_width // 2
                        bar_y = center_y + 20
                        
                        # 배경 (회색)
                        background_rect = pygame.Rect(bar_x, bar_y, bar_width, bar_height)
                        pygame.draw.rect(self.screen, (80, 80, 80), background_rect, border_radius=15)
                        
                        # 진행 바 (파란색)
                        progress_width = int((self.simulation_progress / 100) * bar_width)
                        if progress_width > 0:
                            progress_rect = pygame.Rect(bar_x, bar_y, progress_width, bar_height)
                            pygame.draw.rect(self.screen, Colors.PLAYER1_COLOR, progress_rect, border_radius=15)
                        
                        # 테두리
                        pygame.draw.rect(self.screen, (255, 255, 255), background_rect, 2, border_radius=15)
                        
                        # 퍼센트 표시
                        percent_text = self.font_small.render(f"{self.simulation_progress}%", True, (255, 255, 255))
                        percent_rect = percent_text.get_rect(center=(center_x, bar_y + bar_height // 2))
                        self.screen.blit(percent_text, percent_rect)
                        
                        # Cancel 버튼
                        button_width = 120
                        button_height = 40
                        cancel_button_rect = pygame.Rect(center_x - button_width // 2, center_y + 80, button_width, button_height)
                        
                        # 버튼 배경
                        pygame.draw.rect(self.screen, (180, 40, 40), cancel_button_rect, border_radius=8)
                        pygame.draw.rect(self.screen, (255, 255, 255), cancel_button_rect, 2, border_radius=8)
                        
                        # 버튼 텍스트
                        cancel_text = self.font_small.render("Cancel", True, (255, 255, 255))
                        cancel_text_rect = cancel_text.get_rect(center=cancel_button_rect.center)
                        self.screen.blit(cancel_text, cancel_text_rect)
            elif self.mode == GameMode.MENU:
                self.menu_renderer.draw_menu()
            else:
                self.render_game()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
        sys.exit()
