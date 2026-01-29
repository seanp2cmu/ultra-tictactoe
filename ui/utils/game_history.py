"""게임 기록 및 분석을 위한 히스토리 시스템"""
import json
import pickle
from datetime import datetime
from typing import List, Optional, Tuple

class GameNode:
    """게임 트리의 노드 (하나의 수)"""
    def __init__(self, move: Optional[Tuple[int, int]], board_state: dict, parent=None):
        self.move = move  # (row, col) or None for root
        self.board_state = board_state  # 보드 상태
        self.parent = parent  # 부모 노드
        self.children = []  # 변화 수순들
        self.comment = ""  # 수에 대한 코멘트
        self.is_main_line = True if parent is None else False  # 메인 라인 여부
        
    def add_child(self, move: Tuple[int, int], board_state: dict):
        """자식 노드 추가"""
        child = GameNode(move, board_state, parent=self)
        self.children.append(child)
        
        # 첫 번째 자식만 메인 라인
        if len(self.children) == 1:
            child.is_main_line = True
        
        return child
    
    def get_main_line_child(self):
        """메인 라인 자식 반환"""
        for child in self.children:
            if child.is_main_line:
                return child
        return None
    
    def promote_to_main_line(self):
        """이 노드를 메인 라인으로 승격"""
        if self.parent:
            # 형제들의 메인 라인 해제
            for sibling in self.parent.children:
                sibling.is_main_line = False
            # 이 노드를 메인 라인으로
            self.is_main_line = True

class GameHistory:
    """게임 히스토리 관리 (트리 구조)"""
    def __init__(self, initial_board_state: dict):
        self.root = GameNode(None, initial_board_state)
        self.current_node = self.root
        self.game_info = {
            'date': datetime.now().isoformat(),
            'mode': 'Single',
            'result': None,
            'player1': 'Player 1',
            'player2': 'Player 2'
        }
    
    def add_move(self, move: Tuple[int, int], board_state: dict):
        """현재 위치에서 수 추가"""
        # 같은 수가 이미 있는지 확인
        for child in self.current_node.children:
            if child.move == move:
                # 이미 있으면 그 노드로 이동
                self.current_node = child
                return child
        
        # 새로운 수 추가
        child = self.current_node.add_child(move, board_state)
        self.current_node = child
        return child
    
    def go_to_parent(self) -> bool:
        """이전 수로 이동"""
        if self.current_node.parent:
            self.current_node = self.current_node.parent
            return True
        return False
    
    def go_to_child(self, index: int = 0) -> bool:
        """다음 수로 이동 (기본: 메인 라인)"""
        if index < len(self.current_node.children):
            self.current_node = self.current_node.children[index]
            return True
        return False
    
    def go_to_main_line_child(self) -> bool:
        """메인 라인 다음 수로 이동"""
        child = self.current_node.get_main_line_child()
        if child:
            self.current_node = child
            return True
        return False
    
    def go_to_start(self):
        """시작 위치로 이동"""
        self.current_node = self.root
    
    def go_to_end(self):
        """메인 라인 끝으로 이동"""
        while self.go_to_main_line_child():
            pass
    
    def go_to_node(self, node):
        """특정 노드로 이동"""
        self.current_node = node
    
    def get_main_line_moves(self) -> List[GameNode]:
        """메인 라인 수순 반환"""
        moves = []
        node = self.root
        while True:
            child = node.get_main_line_child()
            if not child:
                break
            moves.append(child)
            node = child
        return moves
    
    def get_all_moves_from_root(self) -> List[GameNode]:
        """루트부터 현재 위치까지의 모든 수"""
        moves = []
        node = self.current_node
        while node.parent:
            moves.append(node)
            node = node.parent
        moves.reverse()
        return moves
    
    def get_move_number(self) -> int:
        """현재 수 번호"""
        return len(self.get_all_moves_from_root())
    
    def save_to_file(self, filepath: str):
        """게임을 파일로 저장"""
        data = {
            'game_info': self.game_info,
            'root': self._serialize_node(self.root),
            'version': '1.0'
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _serialize_node(self, node: GameNode) -> dict:
        """노드를 직렬화"""
        return {
            'move': node.move,
            'board_state': node.board_state,
            'comment': node.comment,
            'is_main_line': node.is_main_line,
            'children': [self._serialize_node(child) for child in node.children]
        }
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """파일에서 게임 로드"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # 루트 노드 복원
        root = cls._deserialize_node(data['root'], None)
        
        # GameHistory 객체 생성
        history = cls.__new__(cls)
        history.root = root
        history.current_node = root
        history.game_info = data.get('game_info', {})
        
        return history
    
    @classmethod
    def _deserialize_node(cls, data: dict, parent) -> GameNode:
        """노드를 역직렬화"""
        node = GameNode(
            move=tuple(data['move']) if data['move'] else None,
            board_state=data['board_state'],
            parent=parent
        )
        node.comment = data.get('comment', '')
        node.is_main_line = data.get('is_main_line', False)
        
        # 자식 노드들 재귀적으로 복원
        for child_data in data.get('children', []):
            child = cls._deserialize_node(child_data, node)
            node.children.append(child)
        
        return node
