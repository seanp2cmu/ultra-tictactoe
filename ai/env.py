import gymnasium as gym
import numpy as np
from gymnasium import spaces
from game import Board

class Env(gym.Env):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = gym.spaces.Discrete(81) 

        self.observation_space = spaces.Box(
            low=0, high=2, shape=(9, 9), dtype=np.int8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = Board()
        self.active_board = None
        return self._get_obs(), {}
    
    def step(self, action):
        row = action // 9
        col = action % 9
        
        legal_moves = self.board.get_legal_moves()
        if (row, col) not in legal_moves:
            return self._get_obs(), -1.0, True, False, {"illegal_move": True}
        
        current_player = self.board.current_player
        
        self.board.make_move(row, col)
        
        terminated = False
        reward = 0.0
        
        if self.board.winner is not None:
            terminated = True
            if self.board.winner == current_player:
                reward = 1.0
            elif self.board.winner == 3:
                reward = 0.0
            else:
                reward = -1.0
        
        return self._get_obs(), reward, terminated, False, {}
    
    def _get_obs(self):
        return np.array(self.board.boards, dtype=np.int8)