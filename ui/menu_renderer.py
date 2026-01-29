"""메뉴 렌더링 관련 함수들"""
import pygame
import os
from ui.game_constants import Colors, GameMode

class MenuRenderer:
    """메뉴 화면 렌더링을 담당하는 클래스"""
    
    def __init__(self, ui):
        self.ui = ui
    
    def draw_menu(self):
        """메뉴 화면 그리기"""
        self.ui.screen.fill(Colors.BG_COLOR)
        
        # 타이틀
        title = self.ui.font_large.render("Ultimate Tic-Tac-Toe", True, Colors.TEXT_COLOR)
        title_rect = title.get_rect(center=(self.ui.WINDOW_WIDTH // 2, 80))
        self.ui.screen.blit(title, title_rect)
        
        # 모드 선택 버튼 그리기
        mouse_pos = pygame.mouse.get_pos()
        
        for rect, mode, label in self.ui.menu_buttons:
            # 호버 효과
            color = Colors.BUTTON_HOVER_COLOR if rect.collidepoint(mouse_pos) else Colors.BUTTON_COLOR
            
            pygame.draw.rect(self.ui.screen, color, rect, border_radius=10)
            pygame.draw.rect(self.ui.screen, Colors.THICK_GRID_COLOR, rect, 2, border_radius=10)
            
            text = self.ui.font_medium.render(label, True, (255, 255, 255))
            text_rect = text.get_rect(center=rect.center)
            self.ui.screen.blit(text, text_rect)
    
    def draw_model_selection(self):
        """모델 선택 화면"""
        self.ui.screen.fill(Colors.BG_COLOR)
        
        # 타이틀
        title = self.ui.font_large.render("Select AI Model", True, Colors.TEXT_COLOR)
        title_rect = title.get_rect(center=(self.ui.WINDOW_WIDTH // 2, 50))
        self.ui.screen.blit(title, title_rect)
        
        if not self.ui.available_models:
            # 모델이 없을 경우
            no_model = self.ui.font_medium.render("No trained models found", True, Colors.TEXT_COLOR)
            no_model_rect = no_model.get_rect(center=(self.ui.WINDOW_WIDTH // 2, 200))
            self.ui.screen.blit(no_model, no_model_rect)
            
            info = self.ui.font_small.render("Train a model first using train.py", True, Colors.TEXT_COLOR)
            info_rect = info.get_rect(center=(self.ui.WINDOW_WIDTH // 2, 250))
            self.ui.screen.blit(info, info_rect)
            
            # Back 버튼 그리기
            for rect, action in self.ui.model_buttons:
                if action == "back":
                    pygame.draw.rect(self.ui.screen, Colors.BUTTON_COLOR, rect, border_radius=10)
                    back_text = self.ui.font_medium.render("Back", True, (255, 255, 255))
                    back_text_rect = back_text.get_rect(center=rect.center)
                    self.ui.screen.blit(back_text, back_text_rect)
        else:
            # 모델 목록 그리기
            mouse_pos = pygame.mouse.get_pos()
            
            for i, model_path in enumerate(self.ui.available_models[:8]):
                model_name = os.path.basename(model_path)
                
                # 버튼 찾기
                rect = None
                for btn_rect, action in self.ui.model_buttons:
                    if isinstance(action, int) and action == i:
                        rect = btn_rect
                        break
                
                if rect:
                    # 선택된 모델은 파란색, 호버는 회색, 기본은 어두운 회색
                    if i == self.ui.selected_model_idx:
                        color = Colors.PLAYER1_COLOR
                    elif rect.collidepoint(mouse_pos):
                        color = Colors.BUTTON_HOVER_COLOR
                    else:
                        color = Colors.BUTTON_COLOR
                    
                    pygame.draw.rect(self.ui.screen, color, rect, border_radius=5)
                    pygame.draw.rect(self.ui.screen, Colors.THICK_GRID_COLOR, rect, 2, border_radius=5)
                    
                    text = self.ui.font_small.render(model_name, True, (255, 255, 255))
                    text_rect = text.get_rect(midleft=(rect.left + 10, rect.centery))
                    self.ui.screen.blit(text, text_rect)
            
            # 시뮬레이션 수 슬라이더
            self.draw_simulation_slider()
            
            # Back 및 Start 버튼 그리기
            mouse_pos = pygame.mouse.get_pos()
            for rect, action in self.ui.model_buttons:
                if action == "back":
                    # Back 버튼 hover 효과
                    if rect.collidepoint(mouse_pos):
                        color = Colors.BUTTON_HOVER_COLOR
                    else:
                        color = Colors.BUTTON_COLOR
                    
                    pygame.draw.rect(self.ui.screen, color, rect, border_radius=10)
                    back_text = self.ui.font_medium.render("Back", True, (255, 255, 255))
                    back_text_rect = back_text.get_rect(center=rect.center)
                    self.ui.screen.blit(back_text, back_text_rect)
                    
                elif action == "start":
                    # Start 버튼 hover 효과
                    if rect.collidepoint(mouse_pos):
                        color = (70, 180, 255)  # 밝은 파란색
                    else:
                        color = Colors.PLAYER1_COLOR
                    
                    pygame.draw.rect(self.ui.screen, color, rect, border_radius=10)
                    start_text = self.ui.font_medium.render("Start Game", True, (255, 255, 255))
                    start_text_rect = start_text.get_rect(center=rect.center)
                    self.ui.screen.blit(start_text, start_text_rect)
    
    def draw_simulation_slider(self):
        """시뮬레이션 수 조정 슬라이더"""
        center_x = self.ui.WINDOW_WIDTH // 2
        slider_width = 400
        slider_x = center_x - slider_width // 2
        
        # 라벨
        label = self.ui.font_small.render("MCTS Simulations:", True, Colors.TEXT_COLOR)
        label_rect = label.get_rect(center=(center_x, 590))
        self.ui.screen.blit(label, label_rect)
        
        # 슬라이더 바
        slider_y = 610
        slider_height = 8
        
        self.ui.slider_rect = pygame.Rect(slider_x, slider_y, slider_width, slider_height)
        pygame.draw.rect(self.ui.screen, Colors.GRID_COLOR, self.ui.slider_rect, border_radius=4)
        
        # 값 범위: 10 ~ 500
        min_val = 10
        max_val = 500
        normalized = (self.ui.num_simulations - min_val) / (max_val - min_val)
        handle_x = slider_x + int(normalized * slider_width)
        
        # 슬라이더 핸들
        handle_radius = 12
        self.ui.slider_handle_rect = pygame.Rect(handle_x - handle_radius, slider_y - handle_radius + slider_height // 2, 
                                                  handle_radius * 2, handle_radius * 2)
        pygame.draw.circle(self.ui.screen, Colors.PLAYER1_COLOR, 
                          (handle_x, slider_y + slider_height // 2), handle_radius)
        pygame.draw.circle(self.ui.screen, Colors.THICK_GRID_COLOR, 
                          (handle_x, slider_y + slider_height // 2), handle_radius, 2)
        
        # 현재 값 표시
        value_text = self.ui.font_small.render(f"{self.ui.num_simulations}", True, Colors.PLAYER1_COLOR)
        self.ui.screen.blit(value_text, (slider_x + slider_width + 20, 605))
    
    def draw_saved_games_list(self):
        """저장된 게임 목록 표시"""
        # 제목
        title = self.ui.font_large.render("Load Game", True, Colors.TEXT_COLOR)
        title_rect = title.get_rect(center=(300, 60))
        self.ui.screen.blit(title, title_rect)
        
        # 저장된 게임 목록
        if not self.ui.saved_games_list:
            no_games = self.ui.font_medium.render("No saved games", True, Colors.TEXT_COLOR)
            no_games_rect = no_games.get_rect(center=(300, 300))
            self.ui.screen.blit(no_games, no_games_rect)
        else:
            start_y = 120
            button_height = 40
            spacing = 50
            
            self.ui.saved_game_buttons = []
            mouse_pos = pygame.mouse.get_pos()
            
            for i, game_path in enumerate(self.ui.saved_games_list[:8]):
                y = start_y + i * spacing
                rect = pygame.Rect(50, y, 500, button_height)
                self.ui.saved_game_buttons.append((rect, game_path))
                
                # Hover 효과
                if rect.collidepoint(mouse_pos):
                    color = (70, 180, 255)
                else:
                    color = Colors.PLAYER1_COLOR
                
                pygame.draw.rect(self.ui.screen, color, rect, border_radius=5)
                pygame.draw.rect(self.ui.screen, Colors.THICK_GRID_COLOR, rect, 2, border_radius=5)
                
                # 게임 이름
                import os
                game_name = os.path.basename(game_path).replace('.json', '')
                text = self.ui.font_small.render(game_name, True, (255, 255, 255))
                text_rect = text.get_rect(midleft=(rect.left + 10, rect.centery))
                self.ui.screen.blit(text, text_rect)
        
        # Back 버튼
        back_rect = pygame.Rect(200, 520, 200, 50)
        self.ui.saved_game_buttons.append((back_rect, "back"))
        
        mouse_pos = pygame.mouse.get_pos()
        if back_rect.collidepoint(mouse_pos):
            color = Colors.BUTTON_HOVER_COLOR
        else:
            color = Colors.BUTTON_COLOR
        
        pygame.draw.rect(self.ui.screen, color, back_rect, border_radius=10)
        back_text = self.ui.font_medium.render("Back", True, (255, 255, 255))
        back_text_rect = back_text.get_rect(center=back_rect.center)
        self.ui.screen.blit(back_text, back_text_rect)
