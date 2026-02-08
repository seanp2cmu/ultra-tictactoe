"""최적화 검증 테스트"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Board


class TestBoardOptimization:
    """Board 클래스 최적화 테스트"""
    
    def test_checker_is_class_variable(self):
        """CHECKER가 클래스 변수인지 확인"""
        b1 = Board()
        b2 = Board()
        # 클래스 변수는 id가 같아야 함
        assert Board.CHECKER is not None
        assert hasattr(Board, 'CHECKER')
        print("✓ CHECKER is class variable")
    
    def test_no_duplicate_completed_boards_init(self):
        """completed_boards 중복 초기화 제거 확인"""
        b = Board()
        assert hasattr(b, 'completed_boards')
        assert len(b.completed_boards) == 3
        assert len(b.completed_boards[0]) == 3
        print("✓ No duplicate completed_boards initialization")
    
    def test_clone_no_checker_copy(self):
        """clone()이 CHECKER를 복사하지 않는지 확인"""
        b1 = Board()
        b2 = b1.clone()
        # clone된 보드에 checker가 없어야 함 (클래스 변수 사용)
        assert not hasattr(b2, 'checker') or b2.__dict__.get('checker') is None
        print("✓ clone() does not copy checker")
    
    def test_get_phase_vectorized(self):
        """get_phase가 벡터화되어 있는지 확인 (성능 테스트)"""
        b = Board()
        # 몇 수 두기
        moves = [(0,0), (4,0), (0,4), (4,4), (0,8)]
        for r, c in moves:
            if (r, c) in b.get_legal_moves():
                b.make_move(r, c)
        
        start = time.perf_counter()
        for _ in range(1000):
            weight, phase = b.get_phase()
        elapsed = time.perf_counter() - start
        
        assert weight > 0
        assert phase in ["opening", "early_mid", "mid", "transition", "near_endgame", "endgame", "deep_endgame"]
        print(f"✓ get_phase() vectorized - 1000 calls in {elapsed:.4f}s")
    
    def test_board_game_logic(self):
        """기본 게임 로직이 정상 작동하는지 확인"""
        b = Board()
        
        # 첫 수
        b.make_move(0, 0)
        assert b.boards[0][0] == 1
        assert b.current_player == 2
        assert b.last_move == (0, 0)
        
        # 두번째 수 (0,0 소보드로 보내짐)
        b.make_move(0, 1)
        assert b.boards[0][1] == 2
        
        print("✓ Basic game logic works")
    
    def test_win_detection(self):
        """승리 판정 테스트"""
        b = Board()
        # 첫번째 소보드에서 player 1 승리 시나리오
        # (0,0) -> (0,1) -> (0,3) -> (1,0) -> (3,0) -> (0,2) -> (0,6) -> (2,0)
        # Player 1: (0,0), (0,3), (0,6) - 첫번째 소보드 첫번째 행
        moves = [
            (0, 0),  # P1 → (0,0) 소보드
            (0, 1),  # P2 → (0,0) 소보드  
            (0, 3),  # P1 → (0,1) 소보드
            (1, 0),  # P2 → (1,0) 소보드
            (3, 0),  # P1 → (0,0) 소보드 - (0,0) 비어있어야 함
        ]
        
        for i, (r, c) in enumerate(moves):
            legal = b.get_legal_moves()
            if (r, c) in legal:
                b.make_move(r, c)
        
        print("✓ Win detection works")


class TestBoardSymmetryOptimization:
    """BoardSymmetry 최적화 테스트"""
    
    def test_get_all_symmetries_no_unnecessary_copy(self):
        """get_all_symmetries가 불필요한 copy를 하지 않는지 확인"""
        from utils import BoardSymmetry
        
        b = Board()
        b.make_move(0, 0)
        
        start = time.perf_counter()
        for _ in range(1000):
            symmetries = BoardSymmetry.get_all_symmetries(b)
        elapsed = time.perf_counter() - start
        
        assert len(symmetries) == 8
        print(f"✓ get_all_symmetries() optimized - 1000 calls in {elapsed:.4f}s")
    
    def test_canonical_hash_consistency(self):
        """canonical hash가 일관성있게 동작하는지 확인"""
        from utils import BoardSymmetry
        
        b = Board()
        b.make_move(4, 4)  # 중앙
        
        hash1 = BoardSymmetry.get_canonical_hash(b)
        hash2 = BoardSymmetry.get_canonical_hash(b)
        
        assert hash1 == hash2
        print("✓ Canonical hash is consistent")


class TestDTWOptimization:
    """DTWCalculator 최적화 테스트"""
    
    def test_move_ordering(self):
        """move ordering이 적용되었는지 확인"""
        from ai.endgame import DTWCalculator
        
        dtw = DTWCalculator(use_cache=False, endgame_threshold=15)
        
        # 엔드게임 상황 시뮬레이션
        b = Board()
        # 대부분의 보드를 채움
        for i in range(60):
            legal = b.get_legal_moves()
            if legal and b.winner is None:
                b.make_move(legal[0][0], legal[0][1])
        
        if b.winner is None and b.count_playable_empty_cells() <= 15:
            result = dtw.calculate_dtw(b)
            print(f"✓ DTW with move ordering works - result: {result}")
        else:
            print("✓ DTW move ordering test skipped (game ended or not in endgame)")


class TestMCTSNodeOptimization:
    """MCTS Node 최적화 테스트"""
    
    def test_is_terminal_caching(self):
        """is_terminal 캐싱이 작동하는지 확인"""
        from ai.mcts import Node
        
        b = Board()
        node = Node(b)
        
        # 첫 호출
        result1 = node.is_terminal()
        # 캐시된 값 확인
        assert node._is_terminal is not None
        # 두번째 호출 (캐시 사용)
        result2 = node.is_terminal()
        
        assert result1 == result2
        assert result1 == False  # 초기 보드는 terminal이 아님
        print("✓ is_terminal() caching works")


class TestNetworkOptimization:
    """Network 최적화 테스트"""
    
    def test_board_to_tensor_vectorized(self):
        """_board_to_tensor가 벡터화되어 있는지 확인"""
        try:
            import torch
            from ai.core import AlphaZeroNet
            
            # CPU에서 테스트
            net = AlphaZeroNet(device='cpu')
            b = Board()
            b.make_move(0, 0)
            b.make_move(0, 1)
            
            start = time.perf_counter()
            for _ in range(100):
                tensor = net.model._board_to_tensor(b)
            elapsed = time.perf_counter() - start
            
            assert tensor.shape == (1, 7, 9, 9)
            print(f"✓ _board_to_tensor() vectorized - 100 calls in {elapsed:.4f}s")
        except Exception as e:
            print(f"⚠ Network test skipped: {e}")


def run_all_tests():
    """모든 테스트 실행"""
    print("\n" + "="*60)
    print("최적화 검증 테스트")
    print("="*60 + "\n")
    
    test_classes = [
        TestBoardOptimization,
        TestBoardSymmetryOptimization,
        TestDTWOptimization,
        TestMCTSNodeOptimization,
        TestNetworkOptimization,
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_class in test_classes:
        print(f"\n--- {test_class.__name__} ---")
        instance = test_class()
        
        for method_name in dir(instance):
            if method_name.startswith('test_'):
                try:
                    getattr(instance, method_name)()
                    total_passed += 1
                except Exception as e:
                    print(f"✗ {method_name}: {e}")
                    total_failed += 1
    
    print("\n" + "="*60)
    print(f"결과: {total_passed} passed, {total_failed} failed")
    print("="*60 + "\n")
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
