"""Move List Panel - 기보 표시"""
import pygame
from ui.game_constants import Colors

class MoveListPanel:
    """기보를 표시하는 패널"""
    def __init__(self, ui):
        self.ui = ui
        self.panel_width = 350
        self.scroll_offset = 0
        self.hovered_node = None
        self.move_rects = {}  # node -> rect 매핑
        self.max_scroll = 0
        self.scrollbar_dragging = False
        
    def get_panel_x(self):
        """패널 X 위치"""
        if self.ui.mode_name in ['REVIEW', 'ANALYSIS']:
            from ui.game_constants import UIConstants
            
            # 가운데 정렬된 컨텐츠의 오른쪽 끝
            base_offset = self.ui.game_renderer.get_content_base_offset()
            
            if self.ui.mode_name == 'ANALYSIS':
                content_width = UIConstants.EVAL_BAR_WIDTH + UIConstants.BOARD_SIZE + 2 * UIConstants.MARGIN + UIConstants.ANALYSIS_WIDTH + self.panel_width
            elif self.ui.show_analysis:
                content_width = UIConstants.EVAL_BAR_WIDTH + UIConstants.BOARD_SIZE + 2 * UIConstants.MARGIN + UIConstants.ANALYSIS_WIDTH + self.panel_width
            else:
                content_width = UIConstants.BOARD_SIZE + 2 * UIConstants.MARGIN + self.panel_width
            
            return base_offset + content_width - self.panel_width
        return 0
    
    def draw(self, game_history):
        """기보 패널 그리기 - 체스 스타일 (두 수씩 한 줄)"""
        if not game_history:
            return
        
        from ui.game_constants import UIConstants
        
        panel_x = self.get_panel_x()
        panel_y = UIConstants.MARGIN + UIConstants.INFO_HEIGHT
        panel_height = UIConstants.BOARD_SIZE
        
        # 배경
        panel_rect = pygame.Rect(panel_x, panel_y, self.panel_width, panel_height)
        pygame.draw.rect(self.ui.screen, (250, 250, 250), panel_rect)
        pygame.draw.rect(self.ui.screen, Colors.THICK_GRID_COLOR, panel_rect, 2)
        
        # 제목
        title = self.ui.font_small.render("Move List", True, Colors.TEXT_COLOR)
        self.ui.screen.blit(title, (panel_x + 10, panel_y + 5))
        
        # 클리핑 영역 설정 (스크롤 영역)
        content_y = panel_y + 35
        content_height = panel_height - 35
        clip_rect = pygame.Rect(panel_x, content_y, self.panel_width, content_height)
        self.ui.screen.set_clip(clip_rect)
        
        # 수 목록을 선형적으로 수집
        moves_list = []  # [(node, depth, move_number)]
        
        def collect_moves_linear(node, depth=0, move_num=[0]):
            """선형적으로 모든 수 수집 - 각 수의 variation을 그 수 바로 다음에 처리"""
            if node.move is None:
                # 루트 노드
                for child in node.children:
                    collect_moves_linear(child, depth, move_num)
                return
            
            move_num[0] += 1
            current_move_num = move_num[0]
            moves_list.append((node, depth, current_move_num))
            
            # variations이 있으면 먼저 처리 (메인 라인 계속하기 전에)
            if len(node.children) > 1:
                main_child = node.children[0]
                for child in node.children[1:]:
                    saved_num = move_num[0]
                    move_num[0] = current_move_num  # variation은 다음 번호부터 시작
                    collect_moves_linear(child, depth + 1, move_num)
                    move_num[0] = saved_num
            
            # 메인 라인 계속
            if node.children:
                collect_moves_linear(node.children[0], depth, move_num)
        
        collect_moves_linear(game_history.root)
        
        # 수를 두 개씩 페어로 그리기
        y_offset = content_y - self.scroll_offset
        self.move_rects = {}
        mouse_pos = pygame.mouse.get_pos()
        line_height = 22
        
        i = 0
        while i < len(moves_list):
            node1, depth1, move_num1 = moves_list[i]
            
            # 들여쓰기
            indent = depth1 * 25
            x_base = panel_x + 10 + indent
            
            # 턴 번호
            turn_num = (move_num1 + 1) // 2
            turn_str = f"{turn_num}."
            turn_surf = self.ui.font_tiny.render(turn_str, True, Colors.TEXT_COLOR)
            self.ui.screen.blit(turn_surf, (x_base, y_offset))
            
            # Player 1 수 (홀수 번째)
            x_move1 = x_base + 35
            if move_num1 % 2 == 1:
                color1 = Colors.PLAYER1_COLOR if node1 == game_history.current_node else Colors.TEXT_COLOR
                move1_surf = self.ui.font_tiny.render(str(node1.move), True, color1)
                
                move1_rect = pygame.Rect(x_move1, y_offset, move1_surf.get_width(), move1_surf.get_height())
                self.move_rects[node1] = move1_rect
                
                if move1_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(self.ui.screen, (220, 235, 255), move1_rect.inflate(4, 2))
                
                self.ui.screen.blit(move1_surf, (x_move1, y_offset))
                
                # 다음 수가 Player 2 수이고 같은 depth면 같은 줄에
                x_move2 = x_move1 + move1_surf.get_width() + 10
                if i + 1 < len(moves_list):
                    node2, depth2, move_num2 = moves_list[i + 1]
                    if depth2 == depth1 and move_num2 % 2 == 0:
                        color2 = Colors.PLAYER1_COLOR if node2 == game_history.current_node else Colors.TEXT_COLOR
                        move2_surf = self.ui.font_tiny.render(str(node2.move), True, color2)
                        
                        move2_rect = pygame.Rect(x_move2, y_offset, move2_surf.get_width(), move2_surf.get_height())
                        self.move_rects[node2] = move2_rect
                        
                        if move2_rect.collidepoint(mouse_pos):
                            pygame.draw.rect(self.ui.screen, (220, 235, 255), move2_rect.inflate(4, 2))
                        
                        self.ui.screen.blit(move2_surf, (x_move2, y_offset))
                        i += 2  # 두 수 모두 처리했으므로
                    else:
                        i += 1
                else:
                    i += 1
            else:
                # Player 2 수가 홀로 시작하는 경우 (variation 등)
                # "... (move)" 형식으로 표시
                dots_surf = self.ui.font_tiny.render("...", True, Colors.TEXT_COLOR)
                self.ui.screen.blit(dots_surf, (x_move1, y_offset))
                
                x_move2 = x_move1 + dots_surf.get_width() + 10
                color2 = Colors.PLAYER1_COLOR if node1 == game_history.current_node else Colors.TEXT_COLOR
                move2_surf = self.ui.font_tiny.render(str(node1.move), True, color2)
                
                move2_rect = pygame.Rect(x_move2, y_offset, move2_surf.get_width(), move2_surf.get_height())
                self.move_rects[node1] = move2_rect
                
                if move2_rect.collidepoint(mouse_pos):
                    pygame.draw.rect(self.ui.screen, (220, 235, 255), move2_rect.inflate(4, 2))
                
                self.ui.screen.blit(move2_surf, (x_move2, y_offset))
                i += 1
            
            y_offset += line_height
        
        final_y = y_offset
        
        # 스크롤 가능 영역 계산
        total_content_height = final_y - content_y + self.scroll_offset
        self.max_scroll = max(0, total_content_height - content_height)
        
        # 클리핑 해제
        self.ui.screen.set_clip(None)
        
        # 스크롤바 그리기
        if self.max_scroll > 0:
            scrollbar_x = panel_x + self.panel_width - 15
            scrollbar_y = content_y
            scrollbar_height = content_height
            
            # 스크롤바 배경
            pygame.draw.rect(self.ui.screen, (200, 200, 200), 
                           (scrollbar_x, scrollbar_y, 10, scrollbar_height))
            
            # 스크롤바 핸들
            handle_height = max(20, int(scrollbar_height * (content_height / total_content_height)))
            handle_y = scrollbar_y + int((self.scroll_offset / self.max_scroll) * (scrollbar_height - handle_height))
            
            pygame.draw.rect(self.ui.screen, Colors.PLAYER1_COLOR,
                           (scrollbar_x, handle_y, 10, handle_height), border_radius=5)
        
    def handle_click(self, pos, game_history):
        """클릭 처리 - 수를 클릭하면 그 위치로 이동"""
        for node, rect in self.move_rects.items():
            if rect.collidepoint(pos):
                game_history.go_to_node(node)
                return True
        return False
    
    def handle_scroll(self, y_delta):
        """스크롤 처리"""
        scroll_speed = 30
        self.scroll_offset = max(0, min(self.max_scroll, self.scroll_offset - y_delta * scroll_speed))
    
    def is_over_panel(self, pos):
        """마우스가 패널 위에 있는지 확인"""
        from ui.game_constants import UIConstants
        
        panel_x = self.get_panel_x()
        panel_y = UIConstants.MARGIN + UIConstants.INFO_HEIGHT
        panel_height = UIConstants.BOARD_SIZE
        return (panel_x <= pos[0] <= panel_x + self.panel_width and 
                panel_y <= pos[1] <= panel_y + panel_height)
