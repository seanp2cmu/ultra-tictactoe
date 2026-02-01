"""
DTW (Distance to Win) + 압축 TT를 사용하는 개선된 AlphaZero 트레이너
엔드게임 성능 향상 및 훈련 데이터 품질 개선
"""
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
from .dtw_calculator import DTWCalculator


class SelfPlayData:
    """
    포지션별 가중치 기반 샘플 버퍼
    
    가중치 전략:
    - 50칸+: 1.0 (미드게임)
    - 40-49칸: 1.0 (중반)
    - 30-39칸: 1.0 (중요)
    - 26-29칸: 1.2 (전환 구간, 가장 중요!)
    - 20-25칸: 0.8 (Tablebase 근처)
    - 10-19칸: 0.5 (Tablebase)
    - 0-9칸: 0.3 (Deep Tablebase)
    """
    
    def __init__(self, max_size=10000):
        self.data = deque(maxlen=max_size)
        self.weights = deque(maxlen=max_size)
        self.categories = deque(maxlen=max_size)
        
        # 캐시된 확률 배열 (샘플링 최적화용)
        self._probs_cache = None
        self._cache_dirty = True
    
    def _get_weight(self, state):
        """
        보드 state에서 가중치 계산
        state shape: (7, 9, 9)
        Channels: my_plane, opponent_plane, my_completed, opponent_completed, 
                  draw_completed, last_move, valid_board_mask
        """
        # 빈 칸 개수 계산 (my_plane + opponent_plane = 0인 곳)
        my_plane = state[0]
        opponent_plane = state[1]
        empty_count = np.sum((my_plane == 0) & (opponent_plane == 0))
        
        if empty_count >= 50:
            weight = 1.0
            category = "opening"
        elif empty_count >= 40:
            weight = 1.0
            category = "early_mid"
        elif empty_count >= 30:
            weight = 1.0
            category = "mid"
        elif empty_count >= 26:
            weight = 1.2  # 전환 구간 - 가장 중요!
            category = "transition"
        elif empty_count >= 20:
            weight = 0.8
            category = "near_tablebase"
        elif empty_count >= 10:
            weight = 0.5
            category = "tablebase"
        else:
            weight = 0.3
            category = "deep_tablebase"
        
        return weight, category
    
    def add(self, state, policy, value, dtw=None):
        """DTW 정보 + 가중치 포함"""
        weight, category = self._get_weight(state)
        
        self.data.append((state, policy, value, dtw))
        self.weights.append(weight)
        self.categories.append(category)
        
        # 캐시 무효화
        self._cache_dirty = True
    
    def sample(self, batch_size):
        """가중치 기반 샘플링 (캐싱 최적화)"""
        if len(self.data) < batch_size:
            batch_indices = list(range(len(self.data)))
        else:
            # 캐시된 확률 배열 사용
            if self._cache_dirty:
                weights_array = np.array(self.weights)
                total_weight = np.sum(weights_array)
                self._probs_cache = weights_array / total_weight
                self._cache_dirty = False
            
            batch_indices = np.random.choice(
                len(self.data),
                size=batch_size,
                replace=False,
                p=self._probs_cache
            )
        
        batch = [self.data[i] for i in batch_indices]
        states, policies, values, dtws = zip(*batch)
        return np.array(states), np.array(policies), np.array(values), list(dtws)
    
    def get_stats(self):
        """통계 반환 (정확한 실시간 계산)"""
        total = len(self.data)
        if total == 0:
            return {}
        
        # 카테고리 분포 정확히 계산
        from collections import Counter
        counter = Counter(self.categories)
        
        return {
            "total": total,
            "avg_weight": np.mean(self.weights) if self.weights else 0,
            "distribution": {
                cat: f"{count} ({100*count/total:.1f}%)"
                for cat, count in counter.items()
            }
        }
    
    def __len__(self):
        return len(self.data)


class SelfPlayWorkerWithDTW:
    def __init__(self, network, dtw_calculator=None, batch_predictor=None, 
                 num_simulations=100, temperature=1.0, use_dtw_endgame=True,
                 dtw_max_depth=12, hot_cache_size=50000, cold_cache_size=500000, use_symmetry=True):
        """
        Args:
            dtw_calculator: DTW 계산기 (None이면 생성)
            use_dtw_endgame: 엔드게임에서 DTW 사용 여부
            dtw_max_depth: DTW 최대 깊이
            hot_cache_size: Hot cache 크기
            cold_cache_size: Cold cache 크기
            use_symmetry: 보드 대칭 사용 (8배 메모리 절약)
        """
        # Agent는 항상 network를 받음 (batch_predictor는 별도로 처리 안 함)
        self.agent = AlphaZeroAgent(network, num_simulations=num_simulations, temperature=temperature)
        
        self.use_dtw_endgame = use_dtw_endgame
        
        if use_dtw_endgame:
            if dtw_calculator is None:
                self.dtw_calculator = DTWCalculator(
                    max_depth=dtw_max_depth, 
                    use_cache=True,
                    hot_size=hot_cache_size,
                    cold_size=cold_cache_size,
                    use_symmetry=use_symmetry
                )
            else:
                self.dtw_calculator = dtw_calculator
        else:
            self.dtw_calculator = None
    
    def play_game(self, verbose=False):
        board = Board()
        game_data = []
        step = 0
        
        while board.winner is None:
            legal_moves = board.get_legal_moves()
            if not legal_moves:
                break
            
            current_player = board.current_player
            
            # 엔드게임이면 DTW 계산
            dtw = None
            is_endgame = False
            if self.use_dtw_endgame and self.dtw_calculator:
                is_endgame = self.dtw_calculator.is_endgame(board)
                if is_endgame:
                    result_data = self.dtw_calculator.calculate_dtw(board)
                    if result_data is not None:
                        result, dtw, best_move = result_data
                        is_winning = (result == 1)
                        
                        # 확정 승리 수가 있으면 그것을 선택
                        if is_winning and dtw <= 5 and best_move:
                            if verbose:
                                print(f"Step {step}: DTW={dtw}, using winning move {best_move}")
                            
                            # Policy는 해당 수에만 확률 부여
                            action_probs = np.zeros(81, dtype=np.float32)
                            action = best_move[0] * 9 + best_move[1]
                            action_probs[action] = 1.0
                            
                            state = self._board_to_input(board)
                            game_data.append((state, action_probs, current_player, dtw))
                            
                            board.make_move(best_move[0], best_move[1])
                            step += 1
                            continue
            
            # 일반적인 MCTS
            root = self.agent.search(board)
            
            action_probs = np.zeros(81, dtype=np.float32)
            for action, child in root.children.items():
                action_probs[action] = child.visits
            
            action_probs = action_probs / np.sum(action_probs)
            
            state = self._board_to_input(board)
            game_data.append((state, action_probs, current_player, dtw))
            
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
        
        # 게임 결과 (ground truth)
        if board.winner is None or board.winner == 3:
            winner = None  # 무승부
        else:
            winner = board.winner  # 1 또는 2
        
        # 훈련 데이터 생성 (게임 결과가 ground truth)
        training_data = []
        for state, policy, player, dtw in game_data:
            # Value 계산: 실제 게임 결과 기준
            if winner is None:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            
            # DTW는 value를 오버라이드하지 않음!
            # DTW는 해당 시점의 이론적 평가일 뿐, 실제 게임 결과가 ground truth
            
            training_data.append((state, policy, value, dtw))
        
        if verbose:
            print(f"Game finished in {step} steps. Winner: {board.winner}")
            if self.dtw_calculator:
                stats = self.dtw_calculator.get_stats()
                if stats:
                    print(f"DTW Cache: {stats}")
        
        return training_data
    
    def _board_to_input(self, board):
        """
        Convert board to network input format (7 channels)
        Uses network's _board_to_tensor method for consistency
        """
        tensor = self.network.model._board_to_tensor(board)
        # Remove batch dimension and convert to numpy
        state = tensor.squeeze(0).cpu().numpy()
        return state


class AlphaZeroTrainerWithDTW:
    def __init__(self, network=None, lr=0.001, weight_decay=1e-4, batch_size=32, 
                 num_simulations=100, replay_buffer_size=10000, device=None, use_amp=True,
                 num_res_blocks=10, num_channels=256, num_parallel_games=1,
                 use_dtw=True, dtw_max_depth=12, hot_cache_size=50000, cold_cache_size=500000, use_symmetry=True,
                 total_iterations=300):
        """
        Args:
            use_dtw: DTW 사용 여부
            dtw_max_depth: DTW 최대 탐색 깊이
            hot_cache_size: Hot cache 크기
            cold_cache_size: Cold cache 크기
            use_symmetry: 보드 대칭 사용 (8배 메모리 절약)
        """
        if network is None:
            from .network import Model
            model = Model(num_res_blocks=num_res_blocks, num_channels=num_channels)
            self.network = AlphaZeroNet(model=model, lr=lr, weight_decay=weight_decay, 
                                       device=device, use_amp=use_amp, total_iterations=total_iterations)
        else:
            self.network = network
        
        self.batch_size = batch_size
        self.num_simulations = num_simulations
        self.replay_buffer = SelfPlayData(max_size=replay_buffer_size)
        self.num_parallel_games = num_parallel_games
        
        # DTW 설정
        self.use_dtw = use_dtw
        self.dtw_max_depth = dtw_max_depth
        self.hot_cache_size = hot_cache_size
        self.cold_cache_size = cold_cache_size
        self.use_symmetry = use_symmetry
        
        if use_dtw:
            self.dtw_calculator = DTWCalculator(
                max_depth=dtw_max_depth, 
                use_cache=True,
                hot_size=hot_cache_size,
                cold_size=cold_cache_size,
                use_symmetry=use_symmetry
            )
        else:
            self.dtw_calculator = None
        
        # 통계
        self.total_dtw_positions = 0
        self.total_dtw_wins = 0
    
    def _play_single_game(self, game_idx, num_games, temperature, verbose, batch_predictor=None, num_simulations=None):
        """단일 게임 실행 (병렬 실행용)"""
        sims = num_simulations if num_simulations is not None else self.num_simulations
        # 각 worker가 자신의 DTW calculator를 생성 (thread-safe)
        # dtw_calculator=None으로 전달하면 worker 내부에서 새로 생성됨
        worker = SelfPlayWorkerWithDTW(
            self.network, 
            dtw_calculator=None,  # None으로 전달하여 각 worker가 자신의 calculator 생성
            batch_predictor=batch_predictor,
            num_simulations=sims, 
            temperature=temperature,
            use_dtw_endgame=self.use_dtw,
            dtw_max_depth=self.dtw_max_depth,
            hot_cache_size=self.hot_cache_size,
            cold_cache_size=self.cold_cache_size,
            use_symmetry=self.use_symmetry
        )
        
        if verbose:
            print(f"\n=== Game {game_idx + 1}/{num_games} ===")
        
        game_data = worker.play_game(verbose=verbose)
        
        if verbose:
            print(f"Collected {len(game_data)} training samples")
        
        return game_data
    
    def generate_self_play_data(self, num_games=10, temperature=1.0, verbose=False, disable_tqdm=False, num_simulations=None):
        all_data = []
        
        if self.num_parallel_games > 1:
            with BatchPredictor(self.network, batch_size=self.num_parallel_games, 
                               wait_time=0.005, verbose=verbose) as batch_predictor:
                with ThreadPoolExecutor(max_workers=self.num_parallel_games) as executor:
                    futures = [
                        executor.submit(self._play_single_game, game_idx, num_games, 
                                      temperature, verbose, batch_predictor, num_simulations)
                        for game_idx in range(num_games)
                    ]
                    
                    game_pbar = tqdm(total=num_games, desc="Self-play", 
                                    leave=False, disable=disable_tqdm, ncols=100)
                    
                    completed_games = 0
                    total_positions = 0
                    dtw_positions = 0
                    
                    for future in as_completed(futures):
                        game_data = future.result()
                        all_data.extend(game_data)
                        completed_games += 1
                        
                        # DTW 통계 수집
                        for _, _, _, dtw in game_data:
                            total_positions += 1
                            if dtw is not None:
                                dtw_positions += 1
                        
                        avg_length = len(all_data) / completed_games if completed_games > 0 else 0
                        dtw_rate = dtw_positions / total_positions if total_positions > 0 else 0
                        
                        game_pbar.update(1)
                        game_pbar.set_postfix({
                            "samples": len(all_data),
                            "avg_len": f"{avg_length:.1f}",
                            "dtw%": f"{dtw_rate:.1%}"
                        })
                    
                    game_pbar.close()
        else:
            game_pbar = tqdm(range(num_games), desc="Self-play", 
                           leave=False, disable=disable_tqdm, ncols=100)
            for game_idx in game_pbar:
                game_data = self._play_single_game(game_idx, num_games, temperature, verbose, None, num_simulations)
                all_data.extend(game_data)
                
                # DTW 통계 수집
                dtw_count = sum(1 for _, _, _, dtw in game_data if dtw is not None)
                dtw_rate = dtw_count / len(game_data) if game_data else 0
                avg_length = len(all_data) / (game_idx + 1) if game_idx >= 0 else 0
                
                game_pbar.set_postfix({
                    "samples": len(all_data),
                    "avg_len": f"{avg_length:.1f}",
                    "dtw%": f"{dtw_rate:.1%}"
                })
        
        # Replay buffer에 추가 및 통계
        for state, policy, value, dtw in all_data:
            self.replay_buffer.add(state, policy, value, dtw)
            
            if dtw is not None:
                self.total_dtw_positions += 1
                if dtw < float('inf'):
                    self.total_dtw_wins += 1
        
        if verbose:
            print(f"\nTotal samples in replay buffer: {len(self.replay_buffer)}")
            if self.use_dtw and self.total_dtw_positions > 0:
                win_rate = self.total_dtw_wins / self.total_dtw_positions
                print(f"DTW positions: {self.total_dtw_positions}, Win rate: {win_rate:.2%}")
                if self.dtw_calculator:
                    cache_stats = self.dtw_calculator.get_stats()
                    print(f"DTW Cache: {cache_stats}")
        
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
        
        epoch_pbar = tqdm(range(num_epochs), desc="Training", 
                         leave=False, disable=disable_tqdm, ncols=100)
        
        for epoch in epoch_pbar:
            states, policies, values, dtws = self.replay_buffer.sample(self.batch_size)
            
            loss_dict = self.network.train_step(states, policies, values)
            
            total_loss += loss_dict['total_loss']
            total_policy_loss += loss_dict['policy_loss']
            total_value_loss += loss_dict['value_loss']
            num_batches += 1
            
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
            if self.use_dtw:
                print("DTW enabled for endgame improvement")
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
        
        # Learning rate scheduler step (after each iteration)
        current_lr = self.network.step_scheduler()
        
        result = {
            'num_samples': num_samples,
            'avg_loss': avg_loss,
            'learning_rate': current_lr
        }
        
        # DTW 통계 추가
        if self.use_dtw and self.dtw_calculator:
            result['dtw_stats'] = self.dtw_calculator.get_stats()
        
        if verbose:
            print(f"Learning rate: {current_lr:.6f}")
        
        return result
    
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
    
    def clear_dtw_cache(self, clear_cold_only=True):
        """
        DTW 캐시 초기화 (메모리 절약)
        
        Args:
            clear_cold_only: True면 Cold cache만 비우고 Hot은 유지 (기본값)
                           False면 전체 캐시 초기화
        """
        if self.use_dtw and self.dtw_calculator:
            if clear_cold_only:
                # Cold만 비워서 메모리 절약하되 Hot은 유지 (성능 유지)
                self.dtw_calculator.tt.cold.clear()
                print("DTW cold cache cleared (hot cache preserved)")
            else:
                # 전체 캐시 초기화
                self.dtw_calculator.clear_cache()
                print("DTW cache fully cleared")
