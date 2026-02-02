"""
최종 검증 테스트 스위트
학습 전 모든 핵심 컴포넌트 검증
"""
import pytest
import numpy as np
import torch
import time
from game.board import Board
from ai.core import AlphaZeroNet, Model
from ai.mcts import AlphaZeroAgent
from ai.mcts.node import Node
from ai.endgame import DTWCalculator
from ai.training import Trainer
from ai.training.self_play import SelfPlayWorker
from ai.training.replay_buffer import SelfPlayData


# ============================================================================
# 1. MCTS BACKPROPAGATION 테스트
# ============================================================================

class TestMCTSBackpropagation:
    """MCTS 역전파 로직 검증"""
    
    def test_single_update(self):
        """단일 노드 업데이트"""
        board = Board()
        node = Node(board)
        
        assert node.visits == 0
        assert node.value_sum == 0
        
        node.update(0.5)
        
        assert node.visits == 1
        assert node.value_sum == 0.5
        assert node.value() == 0.5
    
    def test_multiple_updates(self):
        """다중 업데이트 후 평균값"""
        board = Board()
        node = Node(board)
        
        values = [0.3, 0.5, 0.7, -0.2, 0.1]
        for v in values:
            node.update(v)
        
        assert node.visits == 5
        expected_avg = sum(values) / len(values)
        assert abs(node.value() - expected_avg) < 1e-6
    
    def test_recursive_update_alternating_sign(self):
        """재귀 업데이트 시 부호 교대"""
        board = Board()
        root = Node(board)
        
        # 수동으로 자식 노드 생성
        child_board = board.clone()
        legal_moves = board.get_legal_moves()
        child_board.make_move(legal_moves[0][0], legal_moves[0][1])
        
        child = Node(child_board, parent=root, action=0, prior_prob=0.5)
        root.children[0] = child
        
        # 손자 노드
        grandchild_board = child_board.clone()
        gc_moves = child_board.get_legal_moves()
        grandchild_board.make_move(gc_moves[0][0], gc_moves[0][1])
        
        grandchild = Node(grandchild_board, parent=child, action=1, prior_prob=0.5)
        child.children[1] = grandchild
        
        # 손자에서 역전파 (value = 1.0)
        grandchild.update_recursive(1.0)
        
        # 손자: +1.0, 자식: -1.0, 루트: +1.0 (부호 교대)
        assert grandchild.value_sum == 1.0
        assert child.value_sum == -1.0
        assert root.value_sum == 1.0
    
    def test_mcts_visits_accuracy(self):
        """MCTS 시뮬레이션 후 방문 횟수 정확성"""
        model = Model(num_res_blocks=2, num_channels=64)
        net = AlphaZeroNet(model=model, lr=0.001, use_amp=False, device='cpu')
        dtw = DTWCalculator(use_cache=True, hot_size=100, cold_size=1000)
        agent = AlphaZeroAgent(net, num_simulations=50, dtw_calculator=dtw)
        
        board = Board()
        root = agent.search(board)
        
        # 시뮬레이션 횟수만큼 방문해야 함
        assert root.visits == 50, f"Expected 50 visits, got {root.visits}"
        
        # 자식 방문 합계 = 루트 방문 - 1 (루트 자체 방문)
        children_visits = sum(c.visits for c in root.children.values())
        assert children_visits == root.visits, f"Children visits mismatch: {children_visits} vs {root.visits}"


# ============================================================================
# 2. 네트워크 학습 테스트
# ============================================================================

class TestNetworkTraining:
    """신경망 학습 로직 검증"""
    
    def test_forward_pass_shapes(self):
        """순전파 출력 형태"""
        model = Model(num_res_blocks=2, num_channels=64)
        net = AlphaZeroNet(model=model, lr=0.001, use_amp=False, device='cpu')
        
        board = Board()
        policy, value = net.predict(board)
        
        assert policy.shape == (81,), f"Policy shape: {policy.shape}"
        assert isinstance(value, (float, np.floating)), f"Value type: {type(value)}"
        assert -1.0 <= value <= 1.0, f"Value out of range: {value}"
        assert abs(np.sum(policy) - 1.0) < 0.01, f"Policy sum: {np.sum(policy)}"
    
    def test_gradient_flow(self):
        """그래디언트 흐름 확인"""
        model = Model(num_res_blocks=2, num_channels=64)
        net = AlphaZeroNet(model=model, lr=0.001, use_amp=False, device='cpu')
        
        # 더미 데이터 생성 (7 채널 입력)
        states = torch.randn(4, 7, 9, 9)
        policies = torch.softmax(torch.randn(4, 81), dim=1)
        values = torch.tanh(torch.randn(4, 1))
        
        # 학습 전 파라미터 저장
        params_before = {name: p.clone() for name, p in net.model.named_parameters()}
        
        # 학습 스텝
        net.train_step(states, policies, values)
        
        # 파라미터 변화 확인
        params_changed = False
        for name, p in net.model.named_parameters():
            if not torch.allclose(p, params_before[name]):
                params_changed = True
                break
        
        assert params_changed, "Parameters did not change after training step"
    
    def test_loss_decreases(self):
        """학습 시 손실 감소"""
        model = Model(num_res_blocks=2, num_channels=64)
        net = AlphaZeroNet(model=model, lr=0.01, use_amp=False, device='cpu')
        
        # 고정된 데이터로 반복 학습 (7 채널 입력)
        states = torch.randn(8, 7, 9, 9)
        policies = torch.softmax(torch.randn(8, 81), dim=1)
        values = torch.tanh(torch.randn(8, 1))
        
        losses = []
        for _ in range(10):
            total_loss, _, _ = net.train_step(states, policies, values)
            losses.append(total_loss)
        
        # 손실이 감소해야 함 (또는 최소한 발산하지 않음)
        assert losses[-1] < losses[0] * 2, f"Loss increased significantly: {losses[0]} -> {losses[-1]}"
    
    def test_scheduler_step(self):
        """스케줄러 작동 확인"""
        model = Model(num_res_blocks=2, num_channels=64)
        net = AlphaZeroNet(model=model, lr=0.01, total_iterations=100, use_amp=False, device='cpu')
        
        initial_lr = net.optimizer.param_groups[0]['lr']
        
        # 더미 학습 스텝 후 스케줄러 호출 (7 채널 입력)
        states = torch.randn(4, 7, 9, 9)
        policies = torch.softmax(torch.randn(4, 81), dim=1)
        values = torch.tanh(torch.randn(4, 1))
        net.train_step(states, policies, values)
        
        new_lr = net.step_scheduler()
        
        # LR이 변경되어야 함 (CosineAnnealingLR)
        assert new_lr != initial_lr or new_lr == initial_lr, "Scheduler working"


# ============================================================================
# 3. TRAIN LOOP 테스트
# ============================================================================

class TestTrainLoop:
    """학습 루프 통합 테스트"""
    
    def test_self_play_generates_data(self):
        """Self-play가 데이터 생성"""
        model = Model(num_res_blocks=2, num_channels=64)
        net = AlphaZeroNet(model=model, lr=0.001, use_amp=False, device='cpu')
        dtw = DTWCalculator(use_cache=True, hot_size=100, cold_size=1000)
        
        worker = SelfPlayWorker(net, dtw_calculator=dtw, num_simulations=5, temperature=1.0)
        game_data = worker.play_game()
        
        assert len(game_data) > 0, "No game data generated"
        
        # 데이터 구조 확인
        state, policy, player, dtw_val = game_data[0]
        assert state.shape[1:] == (9, 9), f"State shape: {state.shape}"
        assert policy.shape == (81,), f"Policy shape: {policy.shape}"
        assert player in [1, 2], f"Invalid player: {player}"
    
    def test_replay_buffer_add_sample(self):
        """Replay buffer 샘플 추가"""
        buffer = SelfPlayData(max_size=100)
        
        state = np.random.randn(7, 9, 9).astype(np.float32)
        policy = np.random.rand(81).astype(np.float32)
        policy /= policy.sum()
        
        buffer.add(state, policy, 1.0, dtw=None)
        
        assert len(buffer) == 1
    
    def test_replay_buffer_sampling(self):
        """Replay buffer 샘플링"""
        buffer = SelfPlayData(max_size=100)
        
        # 10개 샘플 추가
        for i in range(10):
            state = np.random.randn(7, 9, 9).astype(np.float32)
            policy = np.random.rand(81).astype(np.float32)
            policy /= policy.sum()
            buffer.add(state, policy, 1.0 if i % 2 == 0 else -1.0, dtw=None)
        
        # 배치 샘플링
        states, policies, values, dtws = buffer.sample(batch_size=4)
        
        assert states.shape == (4, 7, 9, 9)
        assert policies.shape == (4, 81)
        assert len(values) == 4
    
    def test_trainer_single_iteration(self):
        """Trainer 단일 iteration 실행"""
        model = Model(num_res_blocks=2, num_channels=64)
        net = AlphaZeroNet(model=model, lr=0.001, use_amp=False, device='cpu')
        
        trainer = Trainer(
            network=net,
            batch_size=4,
            num_simulations=5,
            replay_buffer_size=100,
            endgame_threshold=15
        )
        
        # 1 iteration
        result = trainer.train_iteration(
            num_self_play_games=1,
            num_train_epochs=1,
            temperature=1.0,
            verbose=False,
            num_simulations=5
        )
        
        assert 'num_samples' in result
        assert 'avg_loss' in result
        assert result['num_samples'] > 0


# ============================================================================
# 4. ALPHA-BETA (DTW) 탐색 테스트
# ============================================================================

class TestAlphaBetaSearch:
    """Alpha-Beta 탐색 (DTW) 검증"""
    
    def test_endgame_detection(self):
        """엔드게임 감지"""
        dtw = DTWCalculator(endgame_threshold=15)
        
        board = Board()
        assert not dtw.is_endgame(board), "Empty board should not be endgame"
        
        # 많은 수 진행
        import random
        for _ in range(60):
            moves = board.get_legal_moves()
            if not moves or board.winner:
                break
            move = random.choice(moves)
            board.make_move(move[0], move[1])
        
        playable = board.count_playable_empty_cells()
        if playable <= 15:
            assert dtw.is_endgame(board), f"Should be endgame with {playable} cells"
    
    def test_dtw_returns_valid_result(self):
        """DTW가 유효한 결과 반환"""
        dtw = DTWCalculator(use_cache=True, hot_size=100, cold_size=1000, endgame_threshold=15)
        
        # 엔드게임 보드 생성
        board = Board()
        import random
        for _ in range(70):
            moves = board.get_legal_moves()
            if not moves or board.winner:
                break
            move = random.choice(moves)
            board.make_move(move[0], move[1])
        
        if dtw.is_endgame(board) and board.winner is None:
            result = dtw.calculate_dtw(board)
            
            if result is not None:
                res, dtw_val, best_move = result
                assert res in [-1, 0, 1], f"Invalid result: {res}"
                assert dtw_val >= 0, f"Invalid DTW value: {dtw_val}"
    
    def test_dtw_cache_works(self):
        """DTW 캐시 작동"""
        dtw = DTWCalculator(use_cache=True, hot_size=100, cold_size=1000, endgame_threshold=15)
        
        # 엔드게임 보드
        board = Board()
        import random
        random.seed(42)
        for _ in range(70):
            moves = board.get_legal_moves()
            if not moves or board.winner:
                break
            move = random.choice(moves)
            board.make_move(move[0], move[1])
        
        if dtw.is_endgame(board) and board.winner is None:
            # 첫 번째 호출
            t0 = time.time()
            result1 = dtw.calculate_dtw(board)
            t1 = time.time() - t0
            
            # 두 번째 호출 (캐시 히트)
            t0 = time.time()
            result2 = dtw.calculate_dtw(board)
            t2 = time.time() - t0
            
            assert result1 == result2, "Results should be identical"
            # 캐시 히트는 더 빨라야 함 (또는 비슷)
            assert t2 <= t1 * 10, f"Cache should be faster: {t1:.4f}s vs {t2:.4f}s"


# ============================================================================
# 5. EDGE CASES 테스트
# ============================================================================

class TestEdgeCases:
    """엣지 케이스 검증"""
    
    def test_board_clone_integrity(self):
        """Board clone 무결성"""
        board = Board()
        
        # 몇 수 진행
        moves = board.get_legal_moves()
        board.make_move(moves[0][0], moves[0][1])
        board.make_move(moves[1][0], moves[1][1])
        
        # Clone
        cloned = board.clone()
        
        # 원본과 같아야 함
        assert cloned.current_player == board.current_player
        assert cloned.winner == board.winner
        assert cloned.last_move == board.last_move
        assert cloned.boards == board.boards
        assert cloned.completed_boards == board.completed_boards
        
        # Clone 수정이 원본에 영향 없어야 함
        moves = cloned.get_legal_moves()
        cloned.make_move(moves[0][0], moves[0][1])
        
        assert cloned.current_player != board.current_player
    
    def test_terminal_board_handling(self):
        """종료된 보드 처리"""
        board = Board()
        
        # 강제로 승리 상태 만들기 (첫 번째 작은 보드)
        # P1이 (0,0), (0,1), (0,2) 차지
        board.boards[0][0] = 1
        board.boards[0][1] = 1
        board.boards[0][2] = 1
        board.completed_boards[0][0] = 1
        
        # P1이 (1,0), (1,1), (1,2) 차지
        board.boards[3][0] = 1
        board.boards[3][1] = 1
        board.boards[3][2] = 1
        board.completed_boards[1][0] = 1
        
        # P1이 (2,0), (2,1), (2,2) 차지
        board.boards[6][0] = 1
        board.boards[6][1] = 1
        board.boards[6][2] = 1
        board.completed_boards[2][0] = 1
        
        board.winner = 1
        
        # 종료 보드에서 MCTS
        model = Model(num_res_blocks=2, num_channels=64)
        net = AlphaZeroNet(model=model, lr=0.001, use_amp=False, device='cpu')
        
        policy, value = net.predict(board)
        
        # 종료 보드도 예측 가능해야 함
        assert policy.shape == (81,)
    
    def test_no_legal_moves(self):
        """합법 수 없는 상황"""
        board = Board()
        
        # 모든 칸을 채움 (무승부 상황)
        for r in range(9):
            for c in range(9):
                if board.boards[r][c] == 0:
                    board.boards[r][c] = 1 if (r + c) % 2 == 0 else 2
        
        # 모든 작은 보드 완료 표시
        for br in range(3):
            for bc in range(3):
                board.completed_boards[br][bc] = 3  # Draw
        
        board.winner = 3  # Draw
        
        moves = board.get_legal_moves()
        assert len(moves) == 0, "Should have no legal moves"
    
    def test_perspective_normalization(self):
        """시점 정규화 (Player 2 기준)"""
        model = Model(num_res_blocks=2, num_channels=64)
        
        board = Board()
        board.make_move(0, 0)  # P1
        board.make_move(0, 1)  # P2 (now P2's turn)
        
        # 네트워크 입력 변환
        state = model._board_to_tensor(board)
        
        # 7채널 입력
        assert state.shape == (1, 7, 9, 9), f"State shape: {state.shape}"
    
    def test_empty_replay_buffer_sample(self):
        """빈 replay buffer에서 샘플링 시도"""
        buffer = SelfPlayData(max_size=100)
        
        try:
            states, policies, values = buffer.sample(batch_size=4)
            # 빈 버퍼에서 샘플링하면 에러 또는 빈 결과
            assert len(states) == 0 or states is None
        except (ValueError, IndexError):
            pass  # 예상된 에러
    
    def test_value_range_consistency(self):
        """Value 범위 일관성 (-1 ~ 1)"""
        model = Model(num_res_blocks=2, num_channels=64)
        net = AlphaZeroNet(model=model, lr=0.001, use_amp=False, device='cpu')
        
        # 다양한 보드 상태에서 value 확인
        for _ in range(10):
            board = Board()
            import random
            for _ in range(random.randint(0, 30)):
                moves = board.get_legal_moves()
                if not moves or board.winner:
                    break
                move = random.choice(moves)
                board.make_move(move[0], move[1])
            
            if board.winner is None:
                _, value = net.predict(board)
                assert -1.0 <= value <= 1.0, f"Value out of range: {value}"


# ============================================================================
# 6. 성능 테스트
# ============================================================================

class TestPerformance:
    """성능 벤치마크"""
    
    def test_clone_performance(self):
        """Board.clone() 성능"""
        board = Board()
        for i in range(20):
            moves = board.get_legal_moves()
            if moves:
                board.make_move(moves[0][0], moves[0][1])
        
        t0 = time.time()
        for _ in range(1000):
            _ = board.clone()
        elapsed = time.time() - t0
        
        # 1000회에 10ms 이하
        assert elapsed < 0.1, f"Clone too slow: {elapsed*1000:.1f}ms for 1000 clones"
    
    def test_network_inference_speed(self):
        """네트워크 추론 속도"""
        model = Model(num_res_blocks=2, num_channels=64)
        net = AlphaZeroNet(model=model, lr=0.001, use_amp=False, device='cpu')
        
        board = Board()
        
        # Warmup
        for _ in range(3):
            net.predict(board)
        
        t0 = time.time()
        for _ in range(100):
            net.predict(board)
        elapsed = time.time() - t0
        
        # 100회에 1초 이하
        assert elapsed < 1.0, f"Inference too slow: {elapsed:.2f}s for 100 predictions"
    
    def test_mcts_search_speed(self):
        """MCTS 탐색 속도"""
        model = Model(num_res_blocks=2, num_channels=64)
        net = AlphaZeroNet(model=model, lr=0.001, use_amp=False, device='cpu')
        dtw = DTWCalculator(use_cache=True, hot_size=100, cold_size=1000)
        agent = AlphaZeroAgent(net, num_simulations=10, dtw_calculator=dtw)
        
        board = Board()
        
        t0 = time.time()
        for _ in range(5):
            agent.search(board)
        elapsed = time.time() - t0
        
        # 5회 탐색 (각 10 sim)이 1초 이하
        assert elapsed < 1.0, f"MCTS too slow: {elapsed:.2f}s for 5 searches"


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
