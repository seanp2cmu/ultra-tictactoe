import math
import random
import copy
import numpy as np
from game import Board

class Node:
    def __init__(self, board: Board, parent=None, action=None):
        self.board = copy.deepcopy(board)
        self.parent = parent
        self.action = action
        self.children: [Node] = []
        self.visits = 0
        self.wins = 0
        self.untried_actions = self.board.get_legal_moves()
        
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def is_terminal(self):
        return self.board.winner is not None or len(self.board.get_legal_moves()) == 0
    
    def best_child(self, c_param=1.41):
        choices_weights = [
            (child.wins / child.visits) + c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[choices_weights.index(max(choices_weights))]
    
    def expand(self):
        action = self.untried_actions.pop()
        next_board = copy.deepcopy(self.board)
        next_board.make_move(action[0], action[1])
        child_node = Node(next_board, parent=self, action=action)
        self.children.append(child_node)
        return child_node
    
    def rollout(self):
        current_board = copy.deepcopy(self.board)
        
        while current_board.winner is None:
            legal_moves = current_board.get_legal_moves()
            if not legal_moves:
                break
            move = random.choice(legal_moves)
            current_board.make_move(move[0], move[1])
        
        return current_board.winner
    
    def backpropagate(self, result, player):
        self.visits += 1
        
        if result == player:
            self.wins += 1
        elif result == 3:
            self.wins += 0.5
        
        if self.parent:
            self.parent.backpropagate(result, player)


class Agent:
    def __init__(self, num_simulations=1000, exploration_param=1.41):
        self.num_simulations = num_simulations
        self.exploration_param = exploration_param
    
    def select_action(self, board: Board, player=None):
        if player is None:
            player = board.current_player
        
        root = Node(board)
        
        for _ in range(self.num_simulations):
            node = root
            
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_param)
            
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            result = node.rollout()
            
            node.backpropagate(result, player)
        
        if not root.children:
            legal_moves = board.get_legal_moves()
            if legal_moves:
                move = random.choice(legal_moves)
                return move[0] * 9 + move[1]
            return 0
        
        best_child = max(root.children, key=lambda c: c.visits)
        action = best_child.action
        return action[0] * 9 + action[1]
    
    def get_action_probs(self, board: Board, player=None):
        if player is None:
            player = board.current_player
        
        root = Node(board)
        
        for _ in range(self.num_simulations):
            node = root
            
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_param)
            
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            result = node.rollout()
            
            node.backpropagate(result, player)
        
        action_visits = {}
        for child in root.children:
            action = child.action[0] * 9 + child.action[1]
            action_visits[action] = child.visits
        
        total_visits = sum(action_visits.values())
        if total_visits == 0:
            return {}
        
        action_probs = {action: visits / total_visits for action, visits in action_visits.items()}
        return action_probs


class AlphaZeroNode:
    def __init__(self, board, parent=None, action=None, prior_prob=0):
        self.board = copy.deepcopy(board)
        self.parent = parent
        self.action = action
        self.prior_prob = prior_prob
        self.children = {}
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
    def __init__(self, network, num_simulations=100, c_puct=1.0, temperature=1.0, batch_size=8):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.batch_size = batch_size  # 배치 크기
    
    def search(self, board):
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
            search_paths = []
            leaf_nodes = []
            leaf_boards = []
            
            for _ in range(batch_size):
                node = root
                search_path = [node]
                
                # Virtual loss 적용하며 leaf까지 이동
                while node.is_expanded() and not node.is_terminal():
                    action, node = node.select_child(self.c_puct)
                    node.visits += 1  # Virtual loss
                    search_path.append(node)
                
                search_paths.append(search_path)
                leaf_nodes.append(node)
                
                if not node.is_terminal():
                    leaf_boards.append(node.board)
            
            # 2. Evaluation - 배치로 한 번에 평가
            if leaf_boards:
                policy_probs_batch, values_batch = self.network.predict_batch(leaf_boards)
                
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
                    else:
                        # 배치 평가 결과 사용
                        policy_probs = policy_probs_batch[leaf_idx]
                        value = float(values_batch[leaf_idx])
                        leaf_idx += 1
                        
                        action_probs = {i: policy_probs[i] for i in range(81)}
                        node.expand(action_probs)
                        
                        if node.board.current_player != board.current_player:
                            value = -value
                    
                    # 3. Backpropagation - Virtual loss 제거하며 업데이트
                    for path_node in reversed(search_paths[i]):
                        path_node.visits -= 1  # Virtual loss 제거
                        path_node.update(value)
                        value = -value
            else:
                # 모든 노드가 터미널인 경우
                for i, node in enumerate(leaf_nodes):
                    if node.board.winner is None or node.board.winner == 3:
                        value = 0
                    elif node.board.winner == board.current_player:
                        value = 1
                    else:
                        value = -1
                    
                    for path_node in reversed(search_paths[i]):
                        path_node.visits -= 1  # Virtual loss 제거
                        path_node.update(value)
                        value = -value
        
        return root
    
    def select_action(self, board, temperature=None):
        if temperature is None:
            temperature = self.temperature
        
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
    
    def get_action_probs(self, board, temperature=None):
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
