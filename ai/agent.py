import math
import random
import copy
from typing import Dict, List, Tuple
import numpy as np
from game import Board

from .dtw_calculator import DTWCalculator
from .network import AlphaZeroNet

class AlphaZeroNode:
    def __init__(self, board, parent=None, action=None, prior_prob=0):
        self.board: Board = copy.deepcopy(board)
        self.parent: AlphaZeroNode = parent
        self.action = action
        self.prior_prob = prior_prob
        self.children: Dict[int, AlphaZeroNode] = {}
        self.visits = 0
        self.value_sum = 0
        
    def is_expanded(self):
        return len(self.children) > 0
    
    def is_terminal(self):
        return self.board.winner is not None or len(self.board.get_legal_moves()) == 0
    
    def value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits
    
    def select_child(self, c_puct=1.0):
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            q_value = child.value()
            u_value = c_puct * child.prior_prob * math.sqrt(self.visits) / (1 + child.visits)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(self, action_probs):
        legal_moves = self.board.get_legal_moves()
        
        for move in legal_moves:
            action = move[0] * 9 + move[1]
            if action not in self.children:
                next_board = copy.deepcopy(self.board)
                next_board.make_move(move[0], move[1])
                prior = action_probs[action] if action in action_probs else 1e-8
                self.children[action] = AlphaZeroNode(next_board, parent=self, action=action, prior_prob=prior)
    
    def update(self, value):
        self.visits += 1
        self.value_sum += value
    
    def update_recursive(self, value):
        if self.parent:
            self.parent.update_recursive(-value)
        self.update(value)


class AlphaZeroAgent:
    def __init__(self, network, num_simulations=100, c_puct=1.0, temperature=1.0, batch_size=8, 
                 use_dtw=False, dtw_max_depth=18):
        self.network: AlphaZeroNet = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.batch_size = batch_size
        self.use_dtw = use_dtw
        
        # DTW 초기화
        if use_dtw:
            self.dtw_calculator = DTWCalculator(
                max_depth=dtw_max_depth, 
                use_cache=True,
                hot_size=500000,
                cold_size=5000000,
                use_symmetry=True
            )
        else:
            self.dtw_calculator = None  # 배치 크기
    
    def search(self, board: Board):
        root = AlphaZeroNode(board)
        
        # Root 확장
        policy_probs, _ = self.network.predict(board)
        action_probs = {i: policy_probs[i] for i in range(81)}
        root.expand(action_probs)
        
        # 배치 MCTS
        num_batches = (self.num_simulations + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(num_batches):
            batch_size = min(self.batch_size, self.num_simulations - batch_idx * self.batch_size)
            
            # 1. Selection - 배치 크기만큼 노드 선택
            search_paths: List[List[AlphaZeroNode]] = []
            leaf_nodes: List[AlphaZeroNode] = []
            leaf_boards: List[Board] = []
            tablebase_results: List[Tuple[int, float, Dict[int, float]]] = []  # (node_idx, value, expand_probs)
            
            for _ in range(batch_size):
                node = root
                search_path = [node]
                
                # Virtual loss 적용하며 leaf까지 이동
                while node.is_expanded() and not node.is_terminal():
                    action, node = node.select_child(self.c_puct)
                    node.visits += 1  # Virtual loss
                    search_path.append(node)
                
                search_paths.append(search_path)
                node_idx = len(leaf_nodes)
                leaf_nodes.append(node)
                
                # Tablebase 체크 (25칸 이하, DTW 사용 시)
                tablebase_hit = False
                if not node.is_terminal() and self.use_dtw and self.dtw_calculator:
                    empty_count = sum(1 for row in node.board.boards for cell in row if cell == 0)
                    
                    if empty_count <= 25:
                        result_data = self.dtw_calculator.calculate_dtw(node.board)
                        
                        if result_data is not None:
                            result, dtw, _ = result_data
                            value = float(result)  # -1, 0, 1
                            
                            # 관점 조정
                            if node.board.current_player != board.current_player:
                                value = -value
                            
                            # Expand용 uniform policy (Tablebase는 정확하므로 policy 불필요)
                            legal_moves = node.board.get_legal_moves()
                            uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
                            expand_probs = {move[0] * 9 + move[1]: uniform_prob for move in legal_moves}
                            
                            tablebase_results.append((node_idx, value, expand_probs))
                            tablebase_hit = True
                
                # Neural Net으로 평가할 노드만 추가
                if not node.is_terminal() and not tablebase_hit:
                    leaf_boards.append(node.board)
            
            # 2. Evaluation
            # 2-1. Tablebase 노드 처리
            tablebase_indices = {idx for idx, _, _ in tablebase_results}
            
            # 2-2. Neural Net 평가
            if leaf_boards:
                policy_probs_batch, values_batch = self.network.predict_batch(leaf_boards)
            
            # 3. 결과 적용 및 Backpropagation
            leaf_idx = 0
            for i, node in enumerate(leaf_nodes):
                if node.is_terminal():
                    # 터미널 노드는 직접 계산
                    if node.board.winner is None or node.board.winner == 3:
                        value = 0
                    elif node.board.winner == board.current_player:
                        value = 1
                    else:
                        value = -1
                elif i in tablebase_indices:
                    # Tablebase 결과 사용
                    _, value, expand_probs = next((r for r in tablebase_results if r[0] == i), (None, 0, {}))
                    if expand_probs:
                        node.expand(expand_probs)
                else:
                    # Neural Net 평가 결과 사용
                    policy_probs = policy_probs_batch[leaf_idx]
                    value = float(values_batch[leaf_idx])
                    leaf_idx += 1
                    
                    action_probs = {i: policy_probs[i] for i in range(81)}
                    node.expand(action_probs)
                    
                    if node.board.current_player != board.current_player:
                        value = -value
                
                # Backpropagation - Virtual loss 제거하며 업데이트
                for path_node in reversed(search_paths[i]):
                    path_node.visits -= 1  # Virtual loss 제거
                    path_node.update(value)
                    value = -value
        
        return root
    
    def select_action(self, board: Board, temperature=None):
        if temperature is None:
            temperature = self.temperature
        
        # DTW로 엔드게임 확정 승리 체크
        if self.use_dtw and self.dtw_calculator:
            if self.dtw_calculator.is_endgame(board):
                best_move, dtw = self.dtw_calculator.get_best_winning_move(board)
                if best_move and dtw < float('inf'):
                    return best_move[0] * 9 + best_move[1]
        
        root = self.search(board)
        
        action_visits = [(action, child.visits) for action, child in root.children.items()]
        
        if not action_visits:
            legal_moves = board.get_legal_moves()
            if legal_moves:
                move = random.choice(legal_moves)
                return move[0] * 9 + move[1]
            return 0
        
        actions, visits = zip(*action_visits)
        visits = np.array(visits, dtype=np.float32)
        
        if temperature == 0:
            action_idx = np.argmax(visits)
            return actions[action_idx]
        
        visits = visits ** (1.0 / temperature)
        probs = visits / np.sum(visits)
        action_idx = np.random.choice(len(actions), p=probs)
        
        return actions[action_idx]
    
    def get_action_probs(self, board: Board, temperature=None):
        if temperature is None:
            temperature = self.temperature
        
        root = self.search(board)
        
        action_visits = [(action, child.visits) for action, child in root.children.items()]
        
        if not action_visits:
            return {}
        
        actions, visits = zip(*action_visits)
        visits = np.array(visits, dtype=np.float32)
        
        if temperature == 0:
            action_probs = np.zeros(len(actions))
            action_probs[np.argmax(visits)] = 1.0
        else:
            visits = visits ** (1.0 / temperature)
            action_probs = visits / np.sum(visits)
        
        return {actions[i]: action_probs[i] for i in range(len(actions))}
