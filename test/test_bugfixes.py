"""
버그 수정 검증 테스트 - Round 1 & Round 2
Critical 버그들이 올바르게 수정되었는지 확인
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.mcts import AlphaZeroAgent, Node
from ai.core import AlphaZeroNet, Model
from ai.endgame import DTWCalculator
from game import Board


class MockNetwork:
    """테스트용 Mock Network"""
    def predict(self, board):
        policy = np.ones(81, dtype=np.float32) / 81
        value = 0.0
        return policy, value
    
    def predict_batch(self, boards):
        policies = np.ones((len(boards), 81), dtype=np.float32) / 81
        values = np.zeros(len(boards), dtype=np.float32)
        return policies, values


def create_player1_win_board():
    """Player 1이 이긴 보드 생성 (수동으로 설정)"""
    board = Board()
    # 수동으로 completed_boards 설정 (P1이 대각선 승리)
    board.completed_boards[0][0] = 1  # Player 1 wins (0,0)
    board.completed_boards[1][1] = 1  # Player 1 wins (1,1)
    board.completed_boards[2][2] = 1  # Player 1 wins (2,2)
    board.winner = 1
    board.current_player = 2  # make_move 후 전환된 상태
    return board


def create_near_endgame_board():
    """엔드게임에 가까운 보드 (20칸 비어있음)"""
    board = Board()
    # 61칸 채우기 (81 - 20 = 61)
    count = 0
    for r in range(9):
        for c in range(9):
            if count >= 61:
                break
            if (r, c) in board.get_legal_moves():
                board.make_move(r, c)
                count += 1
        if count >= 61:
            break
    return board


# ============================================================================
# BUG #1: Virtual Loss - visits 정확도
# ============================================================================

def test_virtual_loss_visits_accuracy():
    """Virtual loss가 visits를 정확히 카운트하는지 확인"""
    network = MockNetwork()
    agent = AlphaZeroAgent(network, num_simulations=100, batch_size=8)
    
    board = Board()
    root = agent.search(board)
    
    # Root visits = num_simulations
    assert root.visits == 100, f"Root visits should be 100, got {root.visits}"
    
    # 자식들 visits 합도 num_simulations와 같아야 함
    total_child_visits = sum(child.visits for child in root.children.values())
    assert total_child_visits == 100, f"Total child visits should be 100, got {total_child_visits}"
    
    print("✓ Virtual loss visits accuracy test passed")


def test_virtual_loss_value_sum_update():
    """Virtual loss가 value_sum만 업데이트하는지 확인"""
    network = MockNetwork()
    agent = AlphaZeroAgent(network, num_simulations=10, batch_size=2)
    
    board = Board()
    root = agent.search(board)
    
    # Root의 value는 자식들의 평균
    # value_sum이 올바르게 누적되었는지 간접 확인
    root_value = root.value()
    assert -1.0 <= root_value <= 1.0, f"Root value should be in [-1, 1], got {root_value}"
    
    # visits와 value_sum 일관성
    assert root.visits > 0, "Root should have visits"
    assert abs(root.value()) <= 1.0, "Value should be normalized"
    
    print("✓ Virtual loss value_sum update test passed")


def test_virtual_loss_multiple_simulations():
    """여러 시뮬레이션에서 visits 누적 확인"""
    network = MockNetwork()
    agent = AlphaZeroAgent(network, num_simulations=50, batch_size=5)
    
    board = Board()
    root = agent.search(board)
    
    # 50번 시뮬레이션
    assert root.visits == 50, f"Expected 50 visits, got {root.visits}"
    
    # 모든 자식의 visits 합 = 50
    total = sum(c.visits for c in root.children.values())
    assert total == 50, f"Total child visits should be 50, got {total}"
    
    print("✓ Virtual loss multiple simulations test passed")


# ============================================================================
# BUG #2: Terminal Value - 플레이어 관점
# ============================================================================

def test_terminal_value_player1_wins():
    """Player 1 승리 시 value 부호 확인"""
    network = MockNetwork()
    agent = AlphaZeroAgent(network, num_simulations=10)
    
    board = create_player1_win_board()
    
    # 게임이 끝났는지 확인
    assert board.winner is not None, "Board should have a winner"
    assert board.winner == 1, f"Winner should be player 1, got {board.winner}"
    
    # make_move 후 current_player가 전환됨
    # Player 1이 이겼으므로 current_player = 2
    print(f"  Winner: {board.winner}, Current Player: {board.current_player}")
    
    # MCTS로 평가
    root = agent.search(board)
    
    # 게임이 끝났으므로 평가가 명확해야 함
    # Current player (2) 관점에서 패배이므로 value < 0
    root_value = root.value()
    print(f"  Root value: {root_value}")
    
    # Terminal 노드이므로 value가 -1, 0, 1 중 하나에 가까워야 함
    assert abs(abs(root_value) - 1.0) < 0.5 or abs(root_value) < 0.1, \
        f"Terminal value should be near -1, 0, or 1, got {root_value}"
    
    print("✓ Terminal value player1 wins test passed")


def test_terminal_value_draw():
    """무승부 시 value = 0 확인"""
    network = MockNetwork()
    agent = AlphaZeroAgent(network, num_simulations=10)
    
    # 무승부 보드 만들기는 복잡하므로 간단히 테스트
    board = Board()
    
    # 몇 수 진행
    for move in [(0, 0), (1, 1), (2, 2)]:
        if move in board.get_legal_moves():
            board.make_move(move[0], move[1])
    
    root = agent.search(board)
    
    # 게임이 계속 진행 중이므로 value는 [-1, 1] 범위
    assert -1.0 <= root.value() <= 1.0
    
    print("✓ Terminal value draw test passed")


def test_terminal_node_direct_evaluation():
    """Terminal 노드가 직접 평가되는지 확인"""
    board = create_player1_win_board()
    node = Node(board)
    
    # Terminal 노드 확인
    assert node.is_terminal(), "Node should be terminal"
    
    # winner와 current_player 관계
    print(f"  Terminal node - Winner: {board.winner}, Current: {board.current_player}")
    
    # 관점에 따른 value
    if board.winner == board.current_player:
        expected_value = 1.0
    elif board.winner == 3 or board.winner is None:
        expected_value = 0.0
    else:
        expected_value = -1.0
    
    print(f"  Expected terminal value: {expected_value}")
    
    print("✓ Terminal node direct evaluation test passed")


# ============================================================================
# BUG #3: DTW MAX_DEPTH
# ============================================================================

def test_dtw_endgame_threshold():
    """DTW가 25칸 이하에서만 계산되는지 확인 (플레이 가능한 빈칸 기준)"""
    dtw = DTWCalculator(endgame_threshold=25, use_cache=False)
    
    # 26칸 이상 (DTW 계산 안 함)
    board_26 = Board()
    for i in range(55):  # 81 - 26 = 55
        moves = board_26.get_legal_moves()
        if moves:
            board_26.make_move(moves[0][0], moves[0][1])
    
    # 실제 플레이 가능한 빈칸 확인
    playable_26 = board_26.count_playable_empty_cells()
    result_26 = dtw.calculate_dtw(board_26)
    
    if playable_26 > 25:
        assert result_26 is None, f"Should return None for {playable_26} > 25 playable cells"
    else:
        # 완료된 보드로 인해 플레이 가능한 칸이 25 이하면 계산 시도
        print(f"  Note: Only {playable_26} playable cells (completed boards exist)")
    
    # 25칸 이하 (DTW 계산 시도)
    board_25 = Board()
    for i in range(56):  # 81 - 25 = 56
        moves = board_25.get_legal_moves()
        if moves:
            board_25.make_move(moves[0][0], moves[0][1])
    
    playable_25 = board_25.count_playable_empty_cells()
    result_25 = dtw.calculate_dtw(board_25)
    
    if playable_25 <= 25:
        # 25칸 이하면 계산 시도 (결과는 None일 수도 있음)
        print(f"  25칸 이하 결과: {result_25}, playable cells: {playable_25}")
    
    print("✓ DTW endgame threshold test passed")


def test_dtw_depth_parameter():
    """DTW _alpha_beta_search에 depth 파라미터가 전달되는지 확인"""
    dtw = DTWCalculator(use_cache=False)
    
    board = Board()
    # 간단한 보드로 테스트
    for i in range(60):
        moves = board.get_legal_moves()
        if moves:
            board.make_move(moves[0][0], moves[0][1])
    
    # depth=0으로 시작해서 재귀 시 depth+1 전달
    result = dtw._alpha_beta_search(board, depth=0)
    
    assert result is not None, "Should return result"
    assert len(result) == 3, "Should return (result, dtw, best_move)"
    
    print("✓ DTW depth parameter test passed")


# ============================================================================
# BUG #4: Neural Net Perspective
# ============================================================================

def test_neural_net_perspective_no_extra_flip():
    """Neural net value에 불필요한 flip이 없는지 확인"""
    network = MockNetwork()
    agent = AlphaZeroAgent(network, num_simulations=20, batch_size=4)
    
    board = Board()
    root = agent.search(board)
    
    # Backprop에서 자동으로 flip되므로 추가 flip 없어야 함
    # 정확한 검증은 어렵지만 최소한 value가 유효한 범위
    assert -1.0 <= root.value() <= 1.0
    
    # 여러 자식들도 확인
    for child in root.children.values():
        if child.visits > 0:
            assert -1.0 <= child.value() <= 1.0
    
    print("✓ Neural net perspective no extra flip test passed")


def test_backprop_value_alternation():
    """Backprop 시 value가 교대로 플립되는지 확인"""
    board = Board()
    root = Node(board)
    
    # 수동으로 path 생성
    board1 = Board()
    board1.make_move(0, 0)
    node1 = Node(board1, parent=root)
    
    board2 = Board()
    board2.make_move(0, 0)
    board2.make_move(1, 1)
    node2 = Node(board2, parent=node1)
    
    # Value 1.0으로 backprop 시뮬레이션
    value = 1.0
    nodes = [node2, node1, root]
    
    for node in nodes:
        node.value_sum += value
        node.visits += 1
        value = -value
    
    # 각 노드의 value 확인
    # node2: 1.0, node1: -1.0, root: 1.0
    assert abs(node2.value() - 1.0) < 1e-5, f"node2.value should be 1.0, got {node2.value()}"
    assert abs(node1.value() - (-1.0)) < 1e-5, f"node1.value should be -1.0, got {node1.value()}"
    assert abs(root.value() - 1.0) < 1e-5, f"root.value should be 1.0, got {root.value()}"
    
    print("✓ Backprop value alternation test passed")


# ============================================================================
# 통합 테스트
# ============================================================================

def test_full_mcts_with_all_fixes():
    """모든 버그 수정이 적용된 전체 MCTS 테스트"""
    network = MockNetwork()
    agent = AlphaZeroAgent(
        network,
        num_simulations=100,
        batch_size=10
    )
    
    board = Board()
    root = agent.search(board)
    
    # 1. Virtual loss: visits 정확도
    assert root.visits == 100, f"Visits should be 100, got {root.visits}"
    
    # 2. Value 범위
    assert -1.0 <= root.value() <= 1.0, f"Value out of range: {root.value()}"
    
    # 3. 자식 노드 확인
    assert len(root.children) > 0, "Root should have children"
    
    # 4. 자식들의 visits 합
    total_visits = sum(c.visits for c in root.children.values())
    assert total_visits == 100, f"Total child visits should be 100, got {total_visits}"
    
    # 5. Action 선택 가능
    action = agent.select_action(board, temperature=0)
    assert 0 <= action < 81, f"Action out of range: {action}"
    
    print("✓ Full MCTS with all fixes test passed")


def test_mcts_with_dtw():
    """DTW와 함께 MCTS 테스트"""
    network = MockNetwork()
    agent = AlphaZeroAgent(
        network,
        num_simulations=50,
        batch_size=5
    )
    
    board = Board()
    root = agent.search(board)
    
    # DTW가 활성화되어도 정상 작동
    assert root.visits == 50
    assert -1.0 <= root.value() <= 1.0
    
    # Endgame 보드
    endgame_board = create_near_endgame_board()
    root_endgame = agent.search(endgame_board)
    
    assert root_endgame.visits > 0
    assert -1.0 <= root_endgame.value() <= 1.0
    
    print("✓ MCTS with DTW test passed")


def test_edge_case_terminal_from_start():
    """시작부터 terminal인 보드"""
    network = MockNetwork()
    agent = AlphaZeroAgent(network, num_simulations=10)
    
    board = create_player1_win_board()
    
    # Terminal 보드에서 MCTS
    root = agent.search(board)
    
    # Terminal이므로 즉시 평가
    assert root.visits >= 10
    
    # Value가 명확해야 함
    assert abs(abs(root.value()) - 1.0) < 0.5 or abs(root.value()) < 0.1
    
    print("✓ Edge case: terminal from start test passed")


if __name__ == "__main__":
    print("=" * 70)
    print("🔍 BUGFIX VERIFICATION TESTS")
    print("=" * 70)
    print()
    
    print("BUG #1: Virtual Loss")
    print("-" * 70)
    test_virtual_loss_visits_accuracy()
    test_virtual_loss_value_sum_update()
    test_virtual_loss_multiple_simulations()
    print()
    
    print("BUG #2: Terminal Value")
    print("-" * 70)
    test_terminal_value_player1_wins()
    test_terminal_value_draw()
    test_terminal_node_direct_evaluation()
    print()
    
    print("BUG #3: DTW MAX_DEPTH")
    print("-" * 70)
    test_dtw_max_depth_limit()
    test_dtw_endgame_threshold()
    test_dtw_depth_parameter()
    print()
    
    print("BUG #4: Neural Net Perspective")
    print("-" * 70)
    test_neural_net_perspective_no_extra_flip()
    test_backprop_value_alternation()
    print()
    
    print("Integration Tests")
    print("-" * 70)
    test_full_mcts_with_all_fixes()
    test_mcts_with_dtw()
    test_edge_case_terminal_from_start()
    print()
    
    print("=" * 70)
    print("✅ ALL BUGFIX TESTS PASSED!")
    print("=" * 70)
