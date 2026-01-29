import numpy as np
import torch
from collections import deque
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
from game import Board
from .agent import AlphaZeroAgent
from .network import AlphaZeroNet
from .batch_predictor import BatchPredictor


class SelfPlayData:
    def __init__(self, max_size=10000):
        self.data = deque(maxlen=max_size)
    
    def add(self, state, policy, value):
        self.data.append((state, policy, value))
    
    def sample(self, batch_size):
        if len(self.data) < batch_size:
            batch = list(self.data)
        else:
            batch = random.sample(self.data, batch_size)
        
        states, policies, values = zip(*batch)
        return np.array(states), np.array(policies), np.array(values)
    
    def __len__(self):
        return len(self.data)


class SelfPlayWorker:
    def __init__(self, network, batch_predictor=None, num_simulations=100, temperature=1.0):
        # batch_predictor가 있으면 그걸 사용, 없으면 직접 network 사용
        if batch_predictor:
            self.agent = AlphaZeroAgent(batch_predictor, num_simulations=num_simulations, temperature=temperature)
        else:
            self.agent = AlphaZeroAgent(network, num_simulations=num_simulations, temperature=temperature)
    
    def play_game(self, verbose=False):
        board = Board()
        game_data = []
        step = 0
        
        while board.winner is None:
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
            
            current_player = board.current_player
            
            root = self.agent.search(board)
            
            action_probs = np.zeros(81, dtype=np.float32)
            for action, child in root.children.items():
                action_probs[action] = child.visits
            
            action_probs = action_probs / np.sum(action_probs)
            
            state = self._board_to_input(board)
            game_data.append((state, action_probs, current_player))
            
            action = self.agent.select_action(board, temperature=self.agent.temperature)
            row, col = action // 9, action % 9
            
            if (row, col) not in legal_moves:
                if verbose:
                    print(f"Illegal move: {action}")
                break
            
            board.make_move(row, col)
            step += 1
            
            if verbose and step % 10 == 0:
                print(f"Step {step}")
        
        if board.winner is None or board.winner == 3:
            final_value = 0
        else:
            final_value = board.winner
        
        training_data = []
        for state, policy, player in game_data:
            if final_value == 0:
                value = 0
            elif final_value == player:
                value = 1
            else:
                value = -1
            
            training_data.append((state, policy, value))
        
        if verbose:
            print(f"Game finished in {step} steps. Winner: {board.winner}")
        
        return training_data
    
    def _board_to_input(self, board):
        boards = np.array(board.boards, dtype=np.float32)
        
        player1_plane = (boards == 1).astype(np.float32)
        player2_plane = (boards == 2).astype(np.float32)
        
        if board.current_player == 1:
            current_player_plane = np.ones((9, 9), dtype=np.float32)
        else:
            current_player_plane = np.zeros((9, 9), dtype=np.float32)
        
        completed_p1_plane = np.zeros((9, 9), dtype=np.float32)
        completed_p2_plane = np.zeros((9, 9), dtype=np.float32)
        completed_draw_plane = np.zeros((9, 9), dtype=np.float32)
        
        for br in range(3):
            for bc in range(3):
                if board.completed_boards[br][bc] == 1:
                    completed_p1_plane[br*3:(br+1)*3, bc*3:(bc+1)*3] = 1
                elif board.completed_boards[br][bc] == 2:
                    completed_p2_plane[br*3:(br+1)*3, bc*3:(bc+1)*3] = 1
                elif board.completed_boards[br][bc] == 3:
                    completed_draw_plane[br*3:(br+1)*3, bc*3:(bc+1)*3] = 1
        
        state = np.stack([
            player1_plane, player2_plane, current_player_plane,
            completed_p1_plane, completed_p2_plane, completed_draw_plane
        ], axis=0)
        return state


class AlphaZeroTrainer:
    def __init__(self, network=None, lr=0.001, weight_decay=1e-4, batch_size=32, 
                 num_simulations=100, replay_buffer_size=10000, device=None, use_amp=True,
                 num_res_blocks=10, num_channels=256, num_parallel_games=1):
        if network is None:
            from .network import Model
            model = Model(num_res_blocks=num_res_blocks, num_channels=num_channels)
            self.network = AlphaZeroNet(model=model, lr=lr, weight_decay=weight_decay, 
                                       device=device, use_amp=use_amp)
        else:
            self.network = network
        
        self.batch_size = batch_size
        self.num_simulations = num_simulations
        self.replay_buffer = SelfPlayData(max_size=replay_buffer_size)
        self.num_parallel_games = num_parallel_games
        
    def _play_single_game(self, game_idx, num_games, temperature, verbose, batch_predictor=None, num_simulations=None):
        """단일 게임 실행 (병렬 실행용)"""
        sims = num_simulations if num_simulations is not None else self.num_simulations
        worker = SelfPlayWorker(self.network, batch_predictor=batch_predictor,
                               num_simulations=sims, temperature=temperature)
        
        if verbose:
            print(f"\n=== Game {game_idx + 1}/{num_games} ===")
        
        game_data = worker.play_game(verbose=verbose)
        
        if verbose:
            print(f"Collected {len(game_data)} training samples")
        
        return game_data
    
    def generate_self_play_data(self, num_games=10, temperature=1.0, verbose=False, disable_tqdm=False, num_simulations=None):
        all_data = []
        
        if self.num_parallel_games > 1:
            # BatchPredictor를 사용하여 병렬 게임의 prediction을 batch로 처리
            with BatchPredictor(self.network, batch_size=self.num_parallel_games, 
                               wait_time=0.005, verbose=verbose) as batch_predictor:
                with ThreadPoolExecutor(max_workers=self.num_parallel_games) as executor:
                    futures = [
                        executor.submit(self._play_single_game, game_idx, num_games, 
                                      temperature, verbose, batch_predictor, num_simulations)
                        for game_idx in range(num_games)
                    ]
                    
                    # tqdm으로 게임 진행 상황 표시
                    game_pbar = tqdm(total=num_games, desc="Self-play", 
                                    leave=False, disable=disable_tqdm, ncols=100)
                    
                    for future in as_completed(futures):
                        game_data = future.result()
                        all_data.extend(game_data)
                        game_pbar.update(1)
                        game_pbar.set_postfix({"samples": len(all_data)})
                    
                    game_pbar.close()
        else:
            game_pbar = tqdm(range(num_games), desc="Self-play", 
                           leave=False, disable=disable_tqdm, ncols=100)
            for game_idx in game_pbar:
                game_data = self._play_single_game(game_idx, num_games, temperature, verbose, None, num_simulations)
                all_data.extend(game_data)
                game_pbar.set_postfix({"samples": len(all_data)})
        
        for state, policy, value in all_data:
            self.replay_buffer.add(state, policy, value)
        
        if verbose:
            print(f"\nTotal samples in replay buffer: {len(self.replay_buffer)}")
        
        return len(all_data)
    
    def train(self, num_epochs=10, verbose=False, disable_tqdm=False):
        if len(self.replay_buffer) < self.batch_size:
            if verbose:
                print(f"Not enough data in replay buffer: {len(self.replay_buffer)} < {self.batch_size}")
            return {}
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        # tqdm으로 학습 진행 상황 표시
        epoch_pbar = tqdm(range(num_epochs), desc="Training", 
                         leave=False, disable=disable_tqdm, ncols=100)
        
        for epoch in epoch_pbar:
            states, policies, values = self.replay_buffer.sample(self.batch_size)
            
            loss_dict = self.network.train_step(states, policies, values)
            
            total_loss += loss_dict['total_loss']
            total_policy_loss += loss_dict['policy_loss']
            total_value_loss += loss_dict['value_loss']
            num_batches += 1
            
            # 현재 loss 표시
            epoch_pbar.set_postfix({
                "loss": f"{loss_dict['total_loss']:.4f}",
                "p_loss": f"{loss_dict['policy_loss']:.4f}",
                "v_loss": f"{loss_dict['value_loss']:.4f}"
            })
        
        return {
            'total_loss': total_loss / num_batches,
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches
        }
    
    def train_iteration(self, num_self_play_games=10, num_train_epochs=10, 
                       temperature=1.0, verbose=False, disable_tqdm=False, num_simulations=None):
        if verbose:
            print("=" * 60)
            print("Generating self-play data...")
            print("=" * 60)
        
        num_samples = self.generate_self_play_data(
            num_games=num_self_play_games, 
            temperature=temperature,
            verbose=verbose,
            disable_tqdm=disable_tqdm,
            num_simulations=num_simulations
        )
        
        if verbose:
            print("\n" + "=" * 60)
            print("Training network...")
            print("=" * 60)
        
        avg_loss = self.train(num_epochs=num_train_epochs, verbose=verbose, disable_tqdm=disable_tqdm)
        
        return {
            'num_samples': num_samples,
            'avg_loss': avg_loss
        }
    
    def save(self, filepath):
        self.network.save(filepath)
        if hasattr(self, 'replay_buffer'):
            buffer_path = filepath.replace('.pth', '_buffer.pkl')
            import pickle
            with open(buffer_path, 'wb') as f:
                pickle.dump(self.replay_buffer.data, f)
    
    def load(self, filepath):
        self.network.load(filepath)
        buffer_path = filepath.replace('.pth', '_buffer.pkl')
        try:
            import pickle
            with open(buffer_path, 'rb') as f:
                self.replay_buffer.data = pickle.load(f)
        except FileNotFoundError:
            pass
