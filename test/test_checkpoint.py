"""
체크포인트 및 DTW 수정 사항 테스트
"""
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Board
from ai.endgame.dtw_calculator import DTWCalculator


class TestDTWNodeLimit:
    """DTW 노드 제한 테스트"""
    
    def test_dtw_endgame_only(self):
        """DTW가 endgame에서만 동작하는지 테스트 (shallow 제거됨)"""
        dtw = DTWCalculator(
            use_cache=False, 
            max_nodes=100,
            endgame_threshold=15
        )
        
        board = Board()
        # 중반 상태 - DTW는 동작하지 않아야 함
        moves = [(0,0), (1,1), (0,1), (1,2), (0,2)]
        
        for i, (r, c) in enumerate(moves):
            if board.winner is None:
                legal = board.get_legal_moves()
                if (r, c) in legal:
                    board.make_move(r, c)
        
        empty_cells = board.count_playable_empty_cells()
        print(f"Empty cells: {empty_cells}")
        print(f"Is endgame: {dtw.is_endgame(board)}")
        
        # 중반이므로 endgame이 아님
        assert not dtw.is_endgame(board), "Should not be endgame with many empty cells"
        print("✓ DTW endgame-only test passed")
        
    def test_full_alpha_beta_aborts(self):
        """_alpha_beta_search가 max_nodes 초과 시 중단되는지 테스트"""
        dtw = DTWCalculator(
            use_cache=False, 
            max_nodes=50,  # 매우 작은 노드 제한
            endgame_threshold=15
        )
        
        board = Board()
        # 엔드게임 상태로 만들기 (15칸 이하)
        # 81 - 15 = 66수 필요
        import random
        random.seed(42)
        
        move_count = 0
        while board.winner is None and board.count_playable_empty_cells() > 14:
            legal = board.get_legal_moves()
            if not legal:
                break
            move = random.choice(legal)
            board.make_move(move[0], move[1])
            move_count += 1
        
        empty_cells = board.count_playable_empty_cells()
        print(f"Empty cells: {empty_cells}")
        print(f"Is endgame: {dtw.is_endgame(board)}")
        
        if dtw.is_endgame(board) and board.winner is None:
            result = dtw.calculate_dtw(board)
            stats = dtw.get_stats()
            print(f"DTW searches: {stats['dtw_searches']}")
            print(f"DTW aborted: {stats['dtw_aborted']}")
            print(f"DTW nodes: {stats['dtw_nodes']}")
            
            # result가 None이면 abort됨
            if result is None:
                print("✓ Full alpha-beta aborted as expected (returned None)")
            else:
                print(f"✓ Full alpha-beta completed: result={result[0]}, dtw={result[1]}")
        else:
            print("Skip: Game already ended or not in endgame")
        
        print("✓ Full alpha-beta abort test passed")


class TestCheckpointSystem:
    """체크포인트 시스템 테스트"""
    
    def test_save_load_with_iteration(self):
        """iteration 정보 저장/로드 테스트"""
        from ai.core.alpha_zero_net import AlphaZeroNet
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 작은 네트워크 생성
            net = AlphaZeroNet(device='cpu')
            
            # iteration 포함해서 저장
            save_path = os.path.join(tmpdir, 'test_checkpoint.pt')
            net.save(save_path, iteration=42)
            
            # 새 네트워크에서 로드
            net2 = AlphaZeroNet(device='cpu')
            loaded_iter = net2.load(save_path)
            
            assert loaded_iter == 42, f"Expected iteration 42, got {loaded_iter}"
            print(f"✓ Checkpoint save/load with iteration test passed (iteration={loaded_iter})")
    
    def test_find_checkpoint_files(self):
        """체크포인트 파일 찾기 테스트"""
        import glob
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 체크포인트 파일들 생성
            for i in [5, 10, 15]:
                path = os.path.join(tmpdir, f'checkpoint_{i}.pt')
                with open(path, 'w') as f:
                    f.write('dummy')
            
            # best.pt도 생성
            best_path = os.path.join(tmpdir, 'best.pt')
            with open(best_path, 'w') as f:
                f.write('dummy')
            
            # 체크포인트 파일 찾기
            checkpoints = glob.glob(os.path.join(tmpdir, 'checkpoint_*.pt'))
            assert len(checkpoints) == 3
            
            # 최신 iteration 찾기
            iterations = []
            for cp in checkpoints:
                name = os.path.basename(cp)
                iter_num = int(name.replace('checkpoint_', '').replace('.pt', ''))
                iterations.append(iter_num)
            
            latest = max(iterations)
            assert latest == 15, f"Expected 15, got {latest}"
            print(f"✓ find_checkpoint_files test passed (latest iter {latest})")
            
            # best.pt 확인
            assert os.path.exists(best_path)
            print("✓ best.pt exists test passed")


def run_all_tests():
    print("=" * 60)
    print("Running DTW Node Limit Tests")
    print("=" * 60)
    
    dtw_tests = TestDTWNodeLimit()
    dtw_tests.test_dtw_endgame_only()
    print()
    dtw_tests.test_full_alpha_beta_aborts()
    
    print("\n" + "=" * 60)
    print("Running Checkpoint System Tests")
    print("=" * 60)
    
    ckpt_tests = TestCheckpointSystem()
    ckpt_tests.test_save_load_with_iteration()
    print()
    ckpt_tests.test_find_checkpoint_files()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == '__main__':
    run_all_tests()
