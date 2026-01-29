"""Review Controls - 게임 네비게이션 버튼"""
import pygame
from ui.game_constants import Colors

class ReviewControls:
    """게임 리뷰를 위한 네비게이션 컨트롤"""
    def __init__(self, ui):
        self.ui = ui
        self.buttons = {}
        
    def draw(self, game_history):
        """네비게이션 버튼 그리기"""
        if not game_history:
            return
        
        from ui.game_constants import UIConstants
        
        # 버튼 위치 - 보드 바로 아래
        board_bottom = UIConstants.MARGIN + UIConstants.INFO_HEIGHT + UIConstants.BOARD_SIZE
        center_x = self.ui.WINDOW_WIDTH // 2
        bottom_y = board_bottom + 15  # 보드 아래 15px 간격
        
        button_width = 50
        button_height = 40
        spacing = 10
        
        # 버튼 정의: [아이콘, 동작]
        buttons_def = [
            ("<<", "start"),   # 시작으로
            ("<", "prev"),     # 이전 수
            (">", "next"),     # 다음 수
            (">>", "end")      # 끝으로
        ]
        
        total_width = len(buttons_def) * button_width + (len(buttons_def) - 1) * spacing
        start_x = center_x - total_width // 2
        
        self.buttons.clear()
        mouse_pos = pygame.mouse.get_pos()
        
        for i, (icon, action) in enumerate(buttons_def):
            x = start_x + i * (button_width + spacing)
            rect = pygame.Rect(x, bottom_y, button_width, button_height)
            self.buttons[action] = rect
            
            # 버튼이 활성화되어 있는지 확인
            is_enabled = self._is_button_enabled(action, game_history)
            
            # Hover 효과
            if rect.collidepoint(mouse_pos) and is_enabled:
                color = Colors.BUTTON_HOVER_COLOR
            else:
                color = Colors.BUTTON_COLOR if is_enabled else (150, 150, 150)
            
            # 버튼 그리기
            pygame.draw.rect(self.ui.screen, color, rect, border_radius=5)
            pygame.draw.rect(self.ui.screen, Colors.THICK_GRID_COLOR, rect, 2, border_radius=5)
            
            # 아이콘
            text_color = (255, 255, 255) if is_enabled else (200, 200, 200)
            text = self.ui.font_medium.render(icon, True, text_color)
            text_rect = text.get_rect(center=rect.center)
            self.ui.screen.blit(text, text_rect)
        
        # 현재 수 표시 - 버튼 아래로 이동
        move_num = game_history.get_move_number()
        total_moves = len(game_history.get_main_line_moves())
        move_info = self.ui.font_small.render(
            f"Move: {move_num} / {total_moves}", 
            True, Colors.TEXT_COLOR
        )
        move_info_rect = move_info.get_rect(center=(center_x, bottom_y + button_height + 15))
        self.ui.screen.blit(move_info, move_info_rect)
    
    def _is_button_enabled(self, action, game_history):
        """버튼이 활성화되어 있는지 확인"""
        if action == "start":
            return game_history.current_node != game_history.root
        elif action == "prev":
            return game_history.current_node.parent is not None
        elif action == "next":
            return len(game_history.current_node.children) > 0
        elif action == "end":
            return game_history.current_node.get_main_line_child() is not None
        return True
    
    def handle_click(self, pos, game_history):
        """클릭 처리"""
        for action, rect in self.buttons.items():
            if rect.collidepoint(pos):
                if self._is_button_enabled(action, game_history):
                    self._perform_action(action, game_history)
                    return True
        return False
    
    def _perform_action(self, action, game_history):
        """동작 수행"""
        if action == "start":
            game_history.go_to_start()
        elif action == "prev":
            game_history.go_to_parent()
        elif action == "next":
            game_history.go_to_main_line_child()
        elif action == "end":
            game_history.go_to_end()
