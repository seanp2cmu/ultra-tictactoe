"""게임 렌더링 관련 함수들"""
import pygame
from ui.game_constants import Colors, UIConstants, GameMode

class GameRenderer:
    """게임 화면 렌더링을 담당하는 클래스"""
    
    def __init__(self, ui):
        self.ui = ui
    
    def get_content_base_offset(self):
        """컨텐츠 가운데 정렬을 위한 base offset"""
        # 각 모드별 총 콘텐츠 너비 계산
        if self.ui.mode == GameMode.ANALYSIS:
            content_width = UIConstants.EVAL_BAR_WIDTH + UIConstants.BOARD_SIZE + UIConstants.ANALYSIS_WIDTH + 350
        elif self.ui.mode == GameMode.REVIEW and self.ui.show_analysis:
            content_width = UIConstants.EVAL_BAR_WIDTH + UIConstants.BOARD_SIZE + UIConstants.ANALYSIS_WIDTH + 350
        elif self.ui.mode == GameMode.REVIEW:
            content_width = UIConstants.BOARD_SIZE + 350
        else:
            content_width = UIConstants.BOARD_SIZE
        
        # 가운데 정렬
        base_offset = (self.ui.WINDOW_WIDTH - content_width - 2 * UIConstants.MARGIN) // 2
        return max(0, base_offset)
        
    def get_board_offset_x(self):
        """보드 X 오프셋 (eval bar + 가운데 정렬)"""
        base_offset = self.get_content_base_offset()
        eval_bar_offset = 0
        if self.ui.mode == GameMode.ANALYSIS or (self.ui.mode == GameMode.REVIEW and self.ui.show_analysis):
            eval_bar_offset = UIConstants.EVAL_BAR_WIDTH
        return base_offset + eval_bar_offset
    
    def draw_grid(self):
        """그리드 그리기"""
        board_offset_x = self.get_board_offset_x()
        offset_x = UIConstants.MARGIN + board_offset_x
        offset_y = UIConstants.MARGIN + UIConstants.INFO_HEIGHT
        
        for i in range(10):
            thickness = 1
            color = Colors.GRID_COLOR
            
            if i % 3 == 0:
                thickness = 4
                color = Colors.THICK_GRID_COLOR
            
            x = offset_x + i * UIConstants.CELL_SIZE
            pygame.draw.line(self.ui.screen, color, 
                           (x, offset_y), 
                           (x, offset_y + UIConstants.BOARD_SIZE), 
                           thickness)
            
            y = offset_y + i * UIConstants.CELL_SIZE
            pygame.draw.line(self.ui.screen, color, 
                           (offset_x, y), 
                           (offset_x + UIConstants.BOARD_SIZE, y), 
                           thickness)
    
    def draw_x(self, row, col, color):
        """X 마크 그리기"""
        center_x, center_y = self.ui.get_cell_center(row, col)
        size = UIConstants.CELL_SIZE // 3
        
        pygame.draw.line(self.ui.screen, color,
                        (center_x - size, center_y - size),
                        (center_x + size, center_y + size), 4)
        pygame.draw.line(self.ui.screen, color,
                        (center_x + size, center_y - size),
                        (center_x - size, center_y + size), 4)
    
    def draw_o(self, row, col, color):
        """O 마크 그리기"""
        center_x, center_y = self.ui.get_cell_center(row, col)
        radius = UIConstants.CELL_SIZE // 3
        pygame.draw.circle(self.ui.screen, color, (center_x, center_y), radius, 4)
    
    def draw_marks(self):
        """모든 마크 그리기"""
        for row in range(9):
            for col in range(9):
                if self.ui.board.boards[row][col] == 1:
                    self.draw_x(row, col, Colors.PLAYER1_COLOR)
                elif self.ui.board.boards[row][col] == 2:
                    self.draw_o(row, col, Colors.PLAYER2_COLOR)
    
    def draw_last_move_highlight(self):
        """마지막 수 하이라이트"""
        if self.ui.board.last_move is None:
            return
        
        row, col = self.ui.board.last_move
        
        highlight_surface = pygame.Surface((UIConstants.CELL_SIZE, UIConstants.CELL_SIZE))
        highlight_surface.set_alpha(100)
        highlight_surface.fill(Colors.LAST_MOVE_COLOR[:3])
        
        board_offset_x = self.get_board_offset_x()
        x = UIConstants.MARGIN + board_offset_x + col * UIConstants.CELL_SIZE
        y = UIConstants.MARGIN + UIConstants.INFO_HEIGHT + row * UIConstants.CELL_SIZE
        self.ui.screen.blit(highlight_surface, (x, y))
    
    def draw_legal_moves_highlight(self):
        """합법 수 하이라이트"""
        legal_moves = self.ui.board.get_legal_moves()
        
        highlight_surface = pygame.Surface((UIConstants.CELL_SIZE, UIConstants.CELL_SIZE))
        highlight_surface.set_alpha(100)
        highlight_surface.fill(Colors.HIGHLIGHT_COLOR[:3])
        
        board_offset_x = self.get_board_offset_x()
        for row, col in legal_moves:
            x = UIConstants.MARGIN + board_offset_x + col * UIConstants.CELL_SIZE
            y = UIConstants.MARGIN + UIConstants.INFO_HEIGHT + row * UIConstants.CELL_SIZE
            self.ui.screen.blit(highlight_surface, (x, y))
    
    def draw_completed_boards(self):
        """완료된 보드 표시"""
        board_offset_x = self.get_board_offset_x()
        for board_row in range(3):
            for board_col in range(3):
                status = self.ui.board.completed_boards[board_row][board_col]
                
                if status != 0:
                    overlay = pygame.Surface((UIConstants.CELL_SIZE * 3, UIConstants.CELL_SIZE * 3))
                    overlay.set_alpha(150)
                    overlay.fill(Colors.COMPLETED_COLOR[:3])
                    
                    x = UIConstants.MARGIN + board_offset_x + board_col * UIConstants.CELL_SIZE * 3
                    y = UIConstants.MARGIN + UIConstants.INFO_HEIGHT + board_row * UIConstants.CELL_SIZE * 3
                    self.ui.screen.blit(overlay, (x, y))
                    
                    center_row = board_row * 3 + 1
                    center_col = board_col * 3 + 1
                    
                    if status == 1:
                        self.draw_big_x(center_row, center_col, Colors.PLAYER1_COLOR)
                    elif status == 2:
                        self.draw_big_o(center_row, center_col, Colors.PLAYER2_COLOR)
    
    def draw_big_x(self, center_row, center_col, color):
        """큰 X 마크"""
        center_x, center_y = self.ui.get_cell_center(center_row, center_col)
        size = UIConstants.CELL_SIZE * 1.2
        
        pygame.draw.line(self.ui.screen, color,
                        (center_x - size, center_y - size),
                        (center_x + size, center_y + size), 8)
        pygame.draw.line(self.ui.screen, color,
                        (center_x + size, center_y - size),
                        (center_x - size, center_y + size), 8)
    
    def draw_big_o(self, center_row, center_col, color):
        """큰 O 마크"""
        center_x, center_y = self.ui.get_cell_center(center_row, center_col)
        radius = int(UIConstants.CELL_SIZE * 1.2)
        pygame.draw.circle(self.ui.screen, color, (center_x, center_y), radius, 8)
    
    def draw_eval_bar(self):
        """평가치 막대 그리기 (체스닷컴 스타일)"""
        if not self.ui.show_analysis or not self.ui.analysis_data:
            return
        
        if self.ui.mode not in [GameMode.ANALYSIS, GameMode.REVIEW]:
            return
        
        base_offset = self.get_content_base_offset()
        bar_x = base_offset
        bar_y = UIConstants.MARGIN + UIConstants.INFO_HEIGHT
        bar_width = UIConstants.EVAL_BAR_WIDTH
        bar_height = UIConstants.BOARD_SIZE
        
        # 배경
        pygame.draw.rect(self.ui.screen, (230, 230, 230), 
                        (bar_x, bar_y, bar_width, bar_height))
        
        # 평가치는 현재 플레이어 관점이므로 절대적 Player 1 관점으로 변환
        value = self.ui.analysis_data['value']
        if self.ui.board.current_player == 2:
            value = -value  # Player 2 차례면 반전
        
        # value > 0: Player1 (Blue) 유리 -> Blue가 아래에서 위로 확장
        # value < 0: Player2 (Red) 유리 -> Red가 위에서 아래로 확장
        normalized_value = (value + 1) / 2  # 0 ~ 1
        
        # Blue (Player1) 영역 - 아래에서 위로
        blue_height = int(bar_height * normalized_value)
        pygame.draw.rect(self.ui.screen, Colors.PLAYER1_COLOR,
                        (bar_x, bar_y + bar_height - blue_height, bar_width, blue_height))
        
        # Red (Player2) 영역 - 위에서 아래로
        red_height = bar_height - blue_height
        pygame.draw.rect(self.ui.screen, Colors.PLAYER2_COLOR,
                        (bar_x, bar_y, bar_width, red_height))
        
        # 테두리
        pygame.draw.rect(self.ui.screen, Colors.THICK_GRID_COLOR,
                        (bar_x, bar_y, bar_width, bar_height), 2)
    
    def draw_info_bar(self):
        """정보 바 그리기"""
        info_rect = pygame.Rect(0, 0, self.ui.WINDOW_WIDTH, UIConstants.MARGIN + UIConstants.INFO_HEIGHT)
        pygame.draw.rect(self.ui.screen, Colors.BG_COLOR, info_rect)
        
        board_offset_x = self.get_board_offset_x()
        
        # 타이틀
        title_text = self.ui.font_large.render("Ultimate Tic-Tac-Toe", True, Colors.TEXT_COLOR)
        title_rect = title_text.get_rect(center=(UIConstants.BOARD_SIZE // 2 + UIConstants.MARGIN + board_offset_x, 40))
        self.ui.screen.blit(title_text, title_rect)
        
        # 모드 표시
        mode_text = ""
        if self.ui.mode == GameMode.AI:
            # AI 모드에서는 누가 Player 1/2인지 표시
            if self.ui.ai_player == 1:
                mode_text = "You (Red) vs AI (Blue)"
            else:
                mode_text = "You (Blue) vs AI (Red)"
        elif self.ui.mode == GameMode.ANALYSIS:
            mode_text = "Analysis Mode"
        elif self.ui.mode == GameMode.REVIEW and self.ui.review_first_player:
            # Review 모드에서 비교 게임 정보 표시
            if self.ui.review_first_player == "Model 1":
                mode_text = "Model 1 (Blue) vs Model 2 (Red)"
            else:
                mode_text = "Model 2 (Blue) vs Model 1 (Red)"
        
        if mode_text:
            mode_render = self.ui.font_small.render(mode_text, True, Colors.PLAYER2_COLOR)
            mode_rect = mode_render.get_rect(center=(UIConstants.BOARD_SIZE // 2 + UIConstants.MARGIN + board_offset_x, 80))
            self.ui.screen.blit(mode_render, mode_rect)
        
        # 현재 플레이어
        if not self.ui.game_over:
            player_color = Colors.PLAYER1_COLOR if self.ui.board.current_player == 1 else Colors.PLAYER2_COLOR
            player_text = f"Current Turn: Player {self.ui.board.current_player}"
            
            if self.ui.mode == GameMode.AI and self.ui.board.current_player == self.ui.ai_player:
                player_text += " (AI)"
            
            info_text = self.ui.font_medium.render(player_text, True, player_color)
            info_rect = info_text.get_rect(center=(UIConstants.BOARD_SIZE // 2 + UIConstants.MARGIN + board_offset_x, 110))
            self.ui.screen.blit(info_text, info_rect)
        
        # 단축키
        shortcuts = "R: Restart  |  M: Menu  |  U: Undo  |  ESC: Quit"
        shortcut_text = self.ui.font_tiny.render(shortcuts, True, Colors.TEXT_COLOR)
        shortcut_rect = shortcut_text.get_rect(center=(UIConstants.BOARD_SIZE // 2 + UIConstants.MARGIN + board_offset_x, 150))
        self.ui.screen.blit(shortcut_text, shortcut_rect)
        
        # AI 모드와 Review 모드에서 Back 버튼 추가
        if self.ui.mode in [GameMode.AI, GameMode.REVIEW]:
            self.draw_back_button()
    
    def draw_back_button(self):
        """뒤로가기 버튼 그리기"""
        button_width = 120
        button_height = 40
        button_x = 20
        button_y = UIConstants.MARGIN + UIConstants.INFO_HEIGHT + UIConstants.BOARD_SIZE + 20
        
        mouse_pos = pygame.mouse.get_pos()
        button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
        
        # 마우스 오버 시 색상 변경
        if button_rect.collidepoint(mouse_pos):
            color = Colors.BUTTON_HOVER_COLOR
        else:
            color = Colors.BUTTON_COLOR
        
        pygame.draw.rect(self.ui.screen, color, button_rect, border_radius=8)
        pygame.draw.rect(self.ui.screen, Colors.THICK_GRID_COLOR, button_rect, 2, border_radius=8)
        
        # 버튼 텍스트
        text = self.ui.font_small.render("← Back", True, (255, 255, 255))
        text_rect = text.get_rect(center=button_rect.center)
        self.ui.screen.blit(text, text_rect)
        
        # 버튼 rect 저장 (클릭 처리용)
        self.ui.back_button_rect = button_rect
    
    def draw_ai_suggestion(self):
        """AI 추천수 하이라이트 - Top 5 moves를 알파값 차등으로 표시"""
        if not self.ui.show_analysis or not self.ui.analysis_data:
            return
        
        board_offset_x = self.get_board_offset_x()
        top_moves = self.ui.analysis_data['top_moves'][:5]  # Top 5만
        
        # 알파값: Top 1이 가장 진하고, Top 5가 가장 연함
        alpha_values = [180, 140, 100, 70, 50]
        
        for i, (move, prob) in enumerate(top_moves):
            row, col = move
            
            # 순위별로 다른 알파값
            alpha = alpha_values[i] if i < len(alpha_values) else 30
            
            # 금색 하이라이트
            highlight_surface = pygame.Surface((UIConstants.CELL_SIZE, UIConstants.CELL_SIZE))
            highlight_surface.set_alpha(alpha)
            highlight_surface.fill(Colors.AI_MOVE_COLOR[:3])
            
            x = UIConstants.MARGIN + board_offset_x + col * UIConstants.CELL_SIZE
            y = UIConstants.MARGIN + UIConstants.INFO_HEIGHT + row * UIConstants.CELL_SIZE
            self.ui.screen.blit(highlight_surface, (x, y))
            
            # 순위 번호 표시 (작게)
            rank_text = self.ui.font_tiny.render(str(i + 1), True, (0, 0, 0))
            text_x = x + UIConstants.CELL_SIZE - 15
            text_y = y + 5
            self.ui.screen.blit(rank_text, (text_x, text_y))
    
    def draw_analysis_panel(self):
        """분석 패널 그리기"""
        if not self.ui.show_analysis or not self.ui.ai_agent:
            return
        
        base_offset = self.get_content_base_offset()
        panel_x = base_offset + UIConstants.EVAL_BAR_WIDTH + UIConstants.BOARD_SIZE + 2 * UIConstants.MARGIN
        panel_y = UIConstants.MARGIN + UIConstants.INFO_HEIGHT
        panel_width = UIConstants.ANALYSIS_WIDTH
        panel_height = UIConstants.BOARD_SIZE
        
        # 배경
        panel_rect = pygame.Rect(panel_x, panel_y, panel_width, panel_height)
        pygame.draw.rect(self.ui.screen, (240, 240, 250), panel_rect)
        pygame.draw.line(self.ui.screen, Colors.THICK_GRID_COLOR, 
                        (panel_x, panel_y), (panel_x, panel_y + panel_height), 3)
        
        # 타이틀
        title = self.ui.font_medium.render("AI Analysis", True, Colors.TEXT_COLOR)
        self.ui.screen.blit(title, (panel_x + 10, panel_y + 10))
        
        # 분석 데이터 계산
        if not self.ui.game_over:
            self.ui.calculate_analysis()
        
        if self.ui.analysis_data:
            y_offset = panel_y + 60
            
            # 평가치 (소수점 2자리)
            value = self.ui.analysis_data['value']
            
            value_text = f"Position Value: {value:.2f}"
            value_color = Colors.PLAYER1_COLOR if value > 0 else Colors.PLAYER2_COLOR if value < 0 else Colors.TEXT_COLOR
            
            value_render = self.ui.font_small.render(value_text, True, value_color)
            self.ui.screen.blit(value_render, (panel_x + 10, y_offset))
            y_offset += 40
            
            # Top N 추천수 (continuation 포함)
            top_title = self.ui.font_small.render(f"Top {self.ui.top_n_moves} Moves:", True, Colors.TEXT_COLOR)
            self.ui.screen.blit(top_title, (panel_x + 10, y_offset))
            y_offset += 30
            
            # continuation이 있으면 그것을 사용, 없으면 기본 표시
            if 'top_moves_with_continuation' in self.ui.analysis_data:
                for i, (move, prob, continuation) in enumerate(self.ui.analysis_data['top_moves_with_continuation'][:self.ui.top_n_moves]):
                    row, col = move
                    
                    # 첫 번째 줄: 순위, 좌표, 확률
                    move_text = f"{i+1}. ({row},{col}): {prob*100:.1f}%"
                    move_render = self.ui.font_tiny.render(move_text, True, Colors.TEXT_COLOR)
                    self.ui.screen.blit(move_render, (panel_x + 15, y_offset))
                    y_offset += 20
                    
                    # 두 번째 줄: continuation (이어지는 수들)
                    if len(continuation) > 1:
                        cont_moves = []
                        for j, (r, c) in enumerate(continuation[1:], 1):  # 첫 수는 제외
                            cont_moves.append(f"({r},{c})")
                        cont_text = "   " + " ".join(cont_moves)
                        cont_render = self.ui.font_tiny.render(cont_text, True, (100, 100, 100))
                        self.ui.screen.blit(cont_render, (panel_x + 15, y_offset))
                    y_offset += 25
                    
                    # 공간이 부족하면 중단
                    if y_offset > panel_y + panel_height - 30:
                        break
            else:
                # 기존 방식
                for i, (move, prob) in enumerate(self.ui.analysis_data['top_moves'][:self.ui.top_n_moves]):
                    row, col = move
                    move_text = f"{i+1}. ({row},{col}): {prob*100:.1f}%"
                    
                    move_render = self.ui.font_tiny.render(move_text, True, Colors.TEXT_COLOR)
                    self.ui.screen.blit(move_render, (panel_x + 15, y_offset))
                    y_offset += 25
    
    def draw_winner_overlay(self):
        """승자 오버레이"""
        overlay = pygame.Surface((self.ui.WINDOW_WIDTH, self.ui.WINDOW_HEIGHT))
        overlay.set_alpha(220)
        overlay.fill((0, 0, 0))
        self.ui.screen.blit(overlay, (0, 0))
        
        box_width, box_height = 500, 300
        box_x = (self.ui.WINDOW_WIDTH - box_width) // 2
        box_y = (self.ui.WINDOW_HEIGHT - box_height) // 2
        
        box_rect = pygame.Rect(box_x, box_y, box_width, box_height)
        pygame.draw.rect(self.ui.screen, (255, 255, 255), box_rect, border_radius=20)
        pygame.draw.rect(self.ui.screen, Colors.THICK_GRID_COLOR, box_rect, 4, border_radius=20)
        
        if self.ui.board.winner == 3:
            winner_text = "It's a Draw!"
            winner_color = Colors.TEXT_COLOR
        else:
            winner_text = f"Player {self.ui.board.winner} Wins!"
            if self.ui.mode == GameMode.AI and self.ui.board.winner == self.ui.ai_player:
                winner_text = "AI Wins!"
            winner_color = Colors.PLAYER1_COLOR if self.ui.board.winner == 1 else Colors.PLAYER2_COLOR
        
        text_surface = self.ui.font_large.render(winner_text, True, winner_color)
        text_rect = text_surface.get_rect(center=(self.ui.WINDOW_WIDTH // 2, self.ui.WINDOW_HEIGHT // 2 - 30))
        self.ui.screen.blit(text_surface, text_rect)
        
        restart_text = self.ui.font_medium.render("Press R to Restart", True, Colors.TEXT_COLOR)
        restart_rect = restart_text.get_rect(center=(self.ui.WINDOW_WIDTH // 2, self.ui.WINDOW_HEIGHT // 2 + 40))
        self.ui.screen.blit(restart_text, restart_rect)
        
        menu_text = self.ui.font_small.render("Press M for Menu", True, Colors.TEXT_COLOR)
        menu_rect = menu_text.get_rect(center=(self.ui.WINDOW_WIDTH // 2, self.ui.WINDOW_HEIGHT // 2 + 80))
        self.ui.screen.blit(menu_text, menu_rect)
