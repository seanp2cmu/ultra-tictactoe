"""Compare Models 화면 렌더링"""
import pygame
import os
from ui.game_constants import Colors


class CompareRenderer:
    """Compare Models 관련 화면 렌더링을 담당하는 클래스"""
    
    def __init__(self, ui):
        self.ui = ui
    
    def draw_settings(self):
        """모델 비교 설정 화면 - 좌우 2열 레이아웃"""
        self.ui.screen.fill(Colors.BG_COLOR)
        center_x = self.ui.WINDOW_WIDTH // 2
        
        # 타이틀
        title = self.ui.font_large.render("Compare AI Models", True, Colors.TEXT_COLOR)
        title_rect = title.get_rect(center=(center_x, 40))
        self.ui.screen.blit(title, title_rect)
        
        # 모델이 2개 이상 없으면 경고 메시지
        if not self.ui.available_models or len(self.ui.available_models) < 2:
            warning = self.ui.font_medium.render("Need at least 2 trained models", True, Colors.TEXT_COLOR)
            warning_rect = warning.get_rect(center=(center_x, 200))
            self.ui.screen.blit(warning, warning_rect)
            
            # Back 버튼
            back_rect = pygame.Rect(center_x - 75, 400, 150, 50)
            mouse_pos = pygame.mouse.get_pos()
            color = Colors.BUTTON_HOVER_COLOR if back_rect.collidepoint(mouse_pos) else Colors.BUTTON_COLOR
            pygame.draw.rect(self.ui.screen, color, back_rect, border_radius=10)
            back_text = self.ui.font_medium.render("Back", True, (255, 255, 255))
            back_text_rect = back_text.get_rect(center=back_rect.center)
            self.ui.screen.blit(back_text, back_text_rect)
            return
        
        mouse_pos = pygame.mouse.get_pos()
        
        # 좌우 2열 레이아웃 (margin-div-gap-div-margin)
        list_width = 250
        gap = 40  # 두 리스트 사이 간격
        total_content_width = list_width * 2 + gap
        start_x = (self.ui.WINDOW_WIDTH - total_content_width) // 2
        
        left_x = start_x + list_width // 2  # 왼쪽 리스트 중심
        right_x = start_x + list_width + gap + list_width // 2  # 오른쪽 리스트 중심
        
        list_y = 100
        list_height = 300
        item_height = 40
        
        # Model 1 (왼쪽)
        model1_label = self.ui.font_medium.render("Model 1", True, Colors.TEXT_COLOR)
        model1_label_rect = model1_label.get_rect(center=(left_x, 80))
        self.ui.screen.blit(model1_label, model1_label_rect)
        
        # Model 1 리스트 배경
        list1_rect = pygame.Rect(left_x - list_width // 2, list_y, list_width, list_height)
        pygame.draw.rect(self.ui.screen, (250, 250, 250), list1_rect, border_radius=5)
        pygame.draw.rect(self.ui.screen, Colors.THICK_GRID_COLOR, list1_rect, 2, border_radius=5)
        
        # Model 1 리스트 항목들
        visible_start1 = self.ui.model1_scroll_offset
        max_visible = list_height // item_height
        visible_models1 = self.ui.available_models[visible_start1:visible_start1 + max_visible]
        
        for i, model_path in enumerate(visible_models1):
            actual_idx = visible_start1 + i
            model_name = os.path.basename(model_path)
            item_y = list_y + i * item_height
            item_rect = pygame.Rect(left_x - list_width // 2 + 3, item_y + 3, list_width - 6, item_height - 6)
            
            # 선택된 모델 강조
            if actual_idx == self.ui.compare_model1_idx:
                pygame.draw.rect(self.ui.screen, Colors.PLAYER1_COLOR, item_rect, border_radius=3)
                text_color = (255, 255, 255)
            elif item_rect.collidepoint(mouse_pos):
                pygame.draw.rect(self.ui.screen, Colors.BUTTON_HOVER_COLOR, item_rect, border_radius=3)
                text_color = (255, 255, 255)
            else:
                text_color = Colors.TEXT_COLOR
            
            item_text = self.ui.font_small.render(model_name, True, text_color)
            item_text_rect = item_text.get_rect(midleft=(item_rect.left + 10, item_rect.centery))
            self.ui.screen.blit(item_text, item_text_rect)
        
        # Model 2 (오른쪽)
        model2_label = self.ui.font_medium.render("Model 2", True, Colors.TEXT_COLOR)
        model2_label_rect = model2_label.get_rect(center=(right_x, 80))
        self.ui.screen.blit(model2_label, model2_label_rect)
        
        # Model 2 리스트 배경
        list2_rect = pygame.Rect(right_x - list_width // 2, list_y, list_width, list_height)
        pygame.draw.rect(self.ui.screen, (250, 250, 250), list2_rect, border_radius=5)
        pygame.draw.rect(self.ui.screen, Colors.THICK_GRID_COLOR, list2_rect, 2, border_radius=5)
        
        # Model 2 리스트 항목들
        visible_start2 = self.ui.model2_scroll_offset
        visible_models2 = self.ui.available_models[visible_start2:visible_start2 + max_visible]
        
        for i, model_path in enumerate(visible_models2):
            actual_idx = visible_start2 + i
            model_name = os.path.basename(model_path)
            item_y = list_y + i * item_height
            item_rect = pygame.Rect(right_x - list_width // 2 + 3, item_y + 3, list_width - 6, item_height - 6)
            
            # 선택된 모델 강조
            if actual_idx == self.ui.compare_model2_idx:
                pygame.draw.rect(self.ui.screen, Colors.PLAYER2_COLOR, item_rect, border_radius=3)
                text_color = (255, 255, 255)
            elif item_rect.collidepoint(mouse_pos):
                pygame.draw.rect(self.ui.screen, Colors.BUTTON_HOVER_COLOR, item_rect, border_radius=3)
                text_color = (255, 255, 255)
            else:
                text_color = Colors.TEXT_COLOR
            
            item_text = self.ui.font_small.render(model_name, True, text_color)
            item_text_rect = item_text.get_rect(midleft=(item_rect.left + 10, item_rect.centery))
            self.ui.screen.blit(item_text, item_text_rect)
        
        # 설정 섹션 (리스트 아래)
        settings_y = list_y + list_height + 20
        
        # 게임 수 선택
        games_label = self.ui.font_small.render("Number of Games:", True, Colors.TEXT_COLOR)
        games_label_rect = games_label.get_rect(center=(center_x, settings_y))
        self.ui.screen.blit(games_label, games_label_rect)
        
        # 게임 수 슬라이더
        slider_width = 300
        slider_x = center_x - slider_width // 2
        slider_y = settings_y + 30
        slider_height = 8
        games_slider = pygame.Rect(slider_x, slider_y, slider_width, slider_height)
        pygame.draw.rect(self.ui.screen, Colors.GRID_COLOR, games_slider, border_radius=4)
        
        # 게임 수: 10 ~ 100
        min_games = 10
        max_games = 100
        normalized_games = (self.ui.compare_num_games - min_games) / (max_games - min_games)
        handle_x_games = slider_x + int(normalized_games * slider_width)
        
        handle_radius = 12
        pygame.draw.circle(self.ui.screen, Colors.PLAYER1_COLOR, 
                          (handle_x_games, slider_y + slider_height // 2), handle_radius)
        pygame.draw.circle(self.ui.screen, Colors.THICK_GRID_COLOR, 
                          (handle_x_games, slider_y + slider_height // 2), handle_radius, 2)
        
        games_value_text = self.ui.font_small.render(f"{self.ui.compare_num_games}", True, Colors.PLAYER1_COLOR)
        self.ui.screen.blit(games_value_text, (slider_x + slider_width + 20, slider_y - 5))
        
        # MCTS 시뮬레이션 수 선택
        sims_y = slider_y + 50
        sims_label = self.ui.font_small.render("MCTS Simulations:", True, Colors.TEXT_COLOR)
        sims_label_rect = sims_label.get_rect(center=(center_x, sims_y))
        self.ui.screen.blit(sims_label, sims_label_rect)
        
        # MCTS 시뮬레이션 수 슬라이더
        slider_y2 = sims_y + 30
        sims_slider = pygame.Rect(slider_x, slider_y2, slider_width, slider_height)
        pygame.draw.rect(self.ui.screen, Colors.GRID_COLOR, sims_slider, border_radius=4)
        
        # MCTS 수: 10 ~ 500
        min_sims = 10
        max_sims = 500
        normalized_sims = (self.ui.compare_simulations - min_sims) / (max_sims - min_sims)
        handle_x_sims = slider_x + int(normalized_sims * slider_width)
        
        pygame.draw.circle(self.ui.screen, Colors.PLAYER1_COLOR, 
                          (handle_x_sims, slider_y2 + slider_height // 2), handle_radius)
        pygame.draw.circle(self.ui.screen, Colors.THICK_GRID_COLOR, 
                          (handle_x_sims, slider_y2 + slider_height // 2), handle_radius, 2)
        
        sims_value_text = self.ui.font_small.render(f"{self.ui.compare_simulations}", True, Colors.PLAYER1_COLOR)
        self.ui.screen.blit(sims_value_text, (slider_x + slider_width + 20, slider_y2 - 5))
        
        # Temperature 선택
        temp_y = slider_y2 + 50
        temp_label = self.ui.font_small.render("Temperature (Randomness):", True, Colors.TEXT_COLOR)
        temp_label_rect = temp_label.get_rect(center=(center_x, temp_y))
        self.ui.screen.blit(temp_label, temp_label_rect)
        
        # Temperature 슬라이더
        slider_y3 = temp_y + 30
        temp_slider = pygame.Rect(slider_x, slider_y3, slider_width, slider_height)
        pygame.draw.rect(self.ui.screen, Colors.GRID_COLOR, temp_slider, border_radius=4)
        
        # Temperature: 0.0 ~ 2.0
        min_temp = 0.0
        max_temp = 2.0
        normalized_temp = (self.ui.compare_temperature - min_temp) / (max_temp - min_temp)
        handle_x_temp = slider_x + int(normalized_temp * slider_width)
        
        pygame.draw.circle(self.ui.screen, Colors.PLAYER2_COLOR, 
                          (handle_x_temp, slider_y3 + slider_height // 2), handle_radius)
        pygame.draw.circle(self.ui.screen, Colors.THICK_GRID_COLOR, 
                          (handle_x_temp, slider_y3 + slider_height // 2), handle_radius, 2)
        
        temp_value_text = self.ui.font_small.render(f"{self.ui.compare_temperature:.1f}", True, Colors.PLAYER2_COLOR)
        self.ui.screen.blit(temp_value_text, (slider_x + slider_width + 20, slider_y3 - 5))
        
        # Back 및 Start 버튼
        button_y = slider_y3 + 50
        button_width_small = 150
        total_width = button_width_small * 2 + 100
        
        back_x = center_x - total_width // 2
        back_button = pygame.Rect(back_x, button_y, button_width_small, 50)
        color = Colors.BUTTON_HOVER_COLOR if back_button.collidepoint(mouse_pos) else Colors.BUTTON_COLOR
        pygame.draw.rect(self.ui.screen, color, back_button, border_radius=10)
        back_text = self.ui.font_medium.render("Back", True, (255, 255, 255))
        back_text_rect = back_text.get_rect(center=back_button.center)
        self.ui.screen.blit(back_text, back_text_rect)
        
        start_x = center_x + total_width // 2 - button_width_small
        start_button = pygame.Rect(start_x, button_y, button_width_small, 50)
        color = (70, 180, 255) if start_button.collidepoint(mouse_pos) else Colors.PLAYER1_COLOR
        pygame.draw.rect(self.ui.screen, color, start_button, border_radius=10)
        start_text = self.ui.font_medium.render("Start", True, (255, 255, 255))
        start_text_rect = start_text.get_rect(center=start_button.center)
        self.ui.screen.blit(start_text, start_text_rect)
    
    def draw_results(self):
        """모델 비교 결과 화면 - 통계 + 게임 리스트"""
        self.ui.screen.fill(Colors.BG_COLOR)
        center_x = self.ui.WINDOW_WIDTH // 2
        
        # 타이틀
        title = self.ui.font_large.render("Comparison Results", True, Colors.TEXT_COLOR)
        title_rect = title.get_rect(center=(center_x, 30))
        self.ui.screen.blit(title, title_rect)
        
        if not self.ui.compare_results:
            return
        
        results = self.ui.compare_results
        
        # 모델 이름
        model1_name = os.path.basename(self.ui.available_models[self.ui.compare_model1_idx])
        model2_name = os.path.basename(self.ui.available_models[self.ui.compare_model2_idx])
        
        # 통계 표시 (간결하게)
        y_offset = 80
        
        # Model 1 vs Model 2
        stats_text = f"{model1_name} vs {model2_name}"
        stats_render = self.ui.font_small.render(stats_text, True, Colors.TEXT_COLOR)
        stats_rect = stats_render.get_rect(center=(center_x, y_offset))
        self.ui.screen.blit(stats_render, stats_rect)
        
        y_offset += 35
        winrate1 = (results['model1_wins'] / results['total_games']) * 100 if results['total_games'] > 0 else 0
        winrate2 = (results['model2_wins'] / results['total_games']) * 100 if results['total_games'] > 0 else 0
        
        score_text = f"Score: {results['model1_wins']}-{results['model2_wins']}-{results['draws']}  |  Win Rate: {winrate1:.1f}% - {winrate2:.1f}%"
        score_render = self.ui.font_tiny.render(score_text, True, Colors.TEXT_COLOR)
        score_rect = score_render.get_rect(center=(center_x, y_offset))
        self.ui.screen.blit(score_render, score_rect)
        
        # 게임 리스트
        y_offset += 50
        list_title = self.ui.font_medium.render("Games", True, Colors.TEXT_COLOR)
        list_title_rect = list_title.get_rect(center=(center_x, y_offset))
        self.ui.screen.blit(list_title, list_title_rect)
        
        y_offset += 40
        
        # 게임 리스트 영역
        mouse_pos = pygame.mouse.get_pos()
        list_start_y = y_offset
        item_height = 30
        visible_games = 12
        
        for i, game in enumerate(self.ui.game_details[:visible_games]):
            y = list_start_y + i * item_height
            
            # 게임 번호, 승자, 시간
            game_text = f"#{game['game_num']}  Winner: {game['winner']}  |  Moves: {game['moves']}  |  Time: {game['time']:.1f}s"
            
            # 승자에 따라 색상 변경
            if game['winner'] == "Model 1":
                color = Colors.PLAYER1_COLOR
            elif game['winner'] == "Model 2":
                color = Colors.PLAYER2_COLOR
            else:
                color = Colors.TEXT_COLOR
            
            # 클릭 가능한 영역
            game_rect = pygame.Rect(center_x - 400, y - 10, 800, item_height)
            
            # Hover 효과
            if game_rect.collidepoint(mouse_pos):
                pygame.draw.rect(self.ui.screen, (240, 240, 250), game_rect, border_radius=3)
            
            game_render = self.ui.font_tiny.render(game_text, True, color)
            game_text_rect = game_render.get_rect(midleft=(center_x - 390, y))
            self.ui.screen.blit(game_render, game_text_rect)
        
        # 스크롤 힌트
        if len(self.ui.game_details) > visible_games:
            hint = self.ui.font_tiny.render("(Scroll for more games)", True, (150, 150, 150))
            hint_rect = hint.get_rect(center=(center_x, list_start_y + visible_games * item_height + 10))
            self.ui.screen.blit(hint, hint_rect)
        
        # Back 버튼
        back_button = pygame.Rect(center_x - 75, 700, 150, 50)
        color = Colors.BUTTON_HOVER_COLOR if back_button.collidepoint(mouse_pos) else Colors.BUTTON_COLOR
        pygame.draw.rect(self.ui.screen, color, back_button, border_radius=10)
        back_text = self.ui.font_medium.render("Back", True, (255, 255, 255))
        back_text_rect = back_text.get_rect(center=back_button.center)
        self.ui.screen.blit(back_text, back_text_rect)
