"""게임 상수 및 설정"""

class GameMode:
    MENU = 'menu'
    SINGLE = 'single'
    AI = 'ai'
    ANALYSIS = 'analysis'
    REVIEW = 'review'
    COMPARE = 'compare'

class UIConstants:
    """UI 관련 상수"""
    CELL_SIZE = 60
    BOARD_SIZE = CELL_SIZE * 9
    MARGIN = 100
    INFO_HEIGHT = 120
    EVAL_BAR_WIDTH = 40
    ANALYSIS_WIDTH = 300
    
class Colors:
    """색상 정의"""
    BG_COLOR = (250, 250, 250)
    GRID_COLOR = (200, 200, 200)
    THICK_GRID_COLOR = (50, 50, 50)
    PLAYER1_COLOR = (52, 152, 219)  # Blue
    PLAYER2_COLOR = (231, 76, 60)   # Red
    HIGHLIGHT_COLOR = (46, 204, 113, 100)
    COMPLETED_COLOR = (200, 200, 200, 150)
    LAST_MOVE_COLOR = (255, 255, 150, 100)  # 연한 노란색
    TEXT_COLOR = (50, 50, 50)
    BUTTON_COLOR = (100, 100, 100)
    BUTTON_HOVER_COLOR = (150, 150, 150)
    AI_MOVE_COLOR = (255, 215, 0, 150)  # Gold
