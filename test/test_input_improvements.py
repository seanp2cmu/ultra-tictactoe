"""
Input 개선사항 검증 테스트
- Completed board masking
- Perspective normalization
- Last move plane
- Valid board mask
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from game import Board
from ai.network import Model

def test_input_channels():
    """입력 채널이 7개로 업데이트되었는지 확인"""
    print("\n=== Test 1: Input Channels (6→7) ===")
    
    model = Model(num_res_blocks=5, num_channels=64)
    
    # Check input layer
    input_conv = model.input[0]
    assert input_conv.in_channels == 7, f"Expected 7 input channels, got {input_conv.in_channels}"
    
    # Test forward pass
    x = torch.randn(2, 7, 9, 9)
    policy, value = model(x)
    
    assert policy.shape == (2, 81), f"Policy shape: {policy.shape}"
    assert value.shape == (2, 1), f"Value shape: {value.shape}"
    
    print(f"✓ Input channels: 7")
    print(f"✓ Forward pass successful")
    print(f"✓ Policy output: {policy.shape}")
    print(f"✓ Value output: {value.shape}")


def test_completed_board_masking():
    """완료된 보드의 칸들이 마스킹되는지 확인"""
    print("\n=== Test 2: Completed Board Masking ===")
    
    model = Model(num_res_blocks=5, num_channels=64)
    board = Board()
    
    # 임의로 칸에 돌 배치
    board.boards[0][0] = 1  # 작은 보드 (0,0)의 (0,0)칸
    board.boards[0][1] = 2
    board.boards[1][0] = 1
    
    # 작은 보드 (0,0) 완료 처리
    board.completed_boards[0][0] = 1  # Player 1 승리
    
    # Tensor 변환
    tensor = model._board_to_tensor(board)
    
    # Shape 확인
    assert tensor.shape == (1, 7, 9, 9), f"Tensor shape: {tensor.shape}"
    
    # Channel 0, 1 (my_plane, opponent_plane) 확인
    # 완료된 보드 영역 (0:3, 0:3)은 모두 0이어야 함
    my_plane = tensor[0, 0].cpu().numpy()
    opp_plane = tensor[0, 1].cpu().numpy()
    
    completed_region_my = my_plane[0:3, 0:3]
    completed_region_opp = opp_plane[0:3, 0:3]
    
    assert np.all(completed_region_my == 0), "Completed board should be masked (my_plane)"
    assert np.all(completed_region_opp == 0), "Completed board should be masked (opponent_plane)"
    
    # Channel 2 (my_completed_plane)는 1이어야 함
    my_completed = tensor[0, 2].cpu().numpy()
    assert np.all(my_completed[0:3, 0:3] == 1), "my_completed should be 1 for completed board"
    
    print(f"✓ Completed board cells masked to 0")
    print(f"✓ Completed board indicator is 1")


def test_perspective_normalization():
    """관점 정규화가 올바르게 작동하는지 확인"""
    print("\n=== Test 3: Perspective Normalization ===")
    
    model = Model(num_res_blocks=5, num_channels=64)
    board = Board()
    
    # Player 1 돌 배치
    board.boards[0][0] = 1
    board.boards[0][1] = 2
    board.current_player = 1
    
    # Player 1 관점
    tensor_p1 = model._board_to_tensor(board)
    my_plane_p1 = tensor_p1[0, 0].cpu().numpy()
    opp_plane_p1 = tensor_p1[0, 1].cpu().numpy()
    
    print(f"P1 turn - my_plane[0,0]: {my_plane_p1[0,0]}, opp_plane[0,1]: {opp_plane_p1[0,1]}")
    assert my_plane_p1[0, 0] == 1.0, "P1 turn: my_plane should have P1 stones"
    assert opp_plane_p1[0, 1] == 1.0, "P1 turn: opp_plane should have P2 stones"
    
    # Player 2 관점 (같은 보드, 차례만 바뀜)
    board.current_player = 2
    tensor_p2 = model._board_to_tensor(board)
    my_plane_p2 = tensor_p2[0, 0].cpu().numpy()
    opp_plane_p2 = tensor_p2[0, 1].cpu().numpy()
    
    print(f"P2 turn - my_plane[0,1]: {my_plane_p2[0,1]}, opp_plane[0,0]: {opp_plane_p2[0,0]}")
    assert my_plane_p2[0, 1] == 1.0, "P2 turn: my_plane should have P2 stones"
    assert opp_plane_p2[0, 0] == 1.0, "P2 turn: opp_plane should have P1 stones"
    
    print(f"✓ Perspective normalization working correctly")
    print(f"✓ Always 'my' vs 'opponent' representation")


def test_last_move_plane():
    """Last move plane이 올바르게 생성되는지 확인"""
    print("\n=== Test 4: Last Move Plane ===")
    
    model = Model(num_res_blocks=5, num_channels=64)
    board = Board()
    
    # Last move 설정
    board.last_move = (2, 5)
    
    tensor = model._board_to_tensor(board)
    last_move_plane = tensor[0, 5].cpu().numpy()  # Channel 5
    
    # (2, 5) 위치만 1이어야 함
    assert last_move_plane[2, 5] == 1.0, f"Last move position should be 1.0"
    assert last_move_plane.sum() == 1.0, f"Only one position should be 1.0, got {last_move_plane.sum()}"
    
    print(f"✓ Last move plane correct at position (2, 5)")
    print(f"✓ Only one cell marked as last move")


def test_valid_board_mask():
    """Valid board mask가 올바르게 생성되는지 확인"""
    print("\n=== Test 5: Valid Board Mask ===")
    
    model = Model(num_res_blocks=5, num_channels=64)
    board = Board()
    
    # Last move: (2, 5) → 다음 수는 작은 보드 (2, 2)에 둬야 함
    # (2 % 3, 5 % 3) = (2, 2)
    board.last_move = (2, 5)
    board.completed_boards[2][2] = 0  # 완료되지 않음
    
    tensor = model._board_to_tensor(board)
    valid_mask = tensor[0, 6].cpu().numpy()  # Channel 6
    
    # 작은 보드 (2, 2) 영역만 1이어야 함
    # 보드 (2, 2) = rows 6:9, cols 6:9
    target_region = valid_mask[6:9, 6:9]
    other_regions = valid_mask.copy()
    other_regions[6:9, 6:9] = 0
    
    assert np.all(target_region == 1.0), "Target board should be valid (all 1s)"
    assert np.all(other_regions == 0.0), "Other boards should not be valid (all 0s)"
    
    print(f"✓ Valid board mask correct for target board (2, 2)")
    print(f"✓ Only target small board marked as valid")
    
    # Test case 2: Target board completed → any board 가능
    board.completed_boards[2][2] = 1  # 완료됨
    tensor2 = model._board_to_tensor(board)
    valid_mask2 = tensor2[0, 6].cpu().numpy()
    
    # 완료되지 않은 모든 보드가 valid해야 함
    valid_count = valid_mask2.sum()
    assert valid_count > 9, f"Should have multiple valid boards, got {valid_count} cells"
    
    print(f"✓ Any board mode: multiple boards valid ({int(valid_count)} cells)")


def test_full_integration():
    """전체 통합 테스트 - 실제 게임 시나리오"""
    print("\n=== Test 6: Full Integration ===")
    
    model = Model(num_res_blocks=5, num_channels=64)
    board = Board()
    
    # 간단한 게임 진행 (valid moves)
    # First move can be anywhere
    board.make_move(0, 0)  # P1: (0,0) → next must be in board (0,0)
    # Now must play in small board (0,0), i.e., rows 0-2, cols 0-2
    board.make_move(0, 1)  # P2: (0,1) → next must be in board (0,1)
    # Now must play in small board (0,1), i.e., rows 0-2, cols 3-5
    board.make_move(1, 3)  # P1: (1,3) → next must be in board (1,0)
    
    # Tensor 생성
    tensor = model._board_to_tensor(board)
    
    # Shape 확인
    assert tensor.shape == (1, 7, 9, 9)
    
    # Forward pass
    tensor_batch = tensor.expand(4, -1, -1, -1)  # Batch of 4
    policy, value = model(tensor_batch)
    
    assert policy.shape == (4, 81)
    assert value.shape == (4, 1)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)
    
    print(f"✓ Full integration successful")
    print(f"✓ Tensor shape: {tensor.shape}")
    print(f"✓ Forward pass output shapes correct")


if __name__ == "__main__":
    print("=" * 70)
    print("Input 개선사항 검증 테스트")
    print("=" * 70)
    
    try:
        test_input_channels()
        test_completed_board_masking()
        test_perspective_normalization()
        test_last_move_plane()
        test_valid_board_mask()
        test_full_integration()
        
        print("\n" + "=" * 70)
        print("✅ 모든 테스트 통과!")
        print("=" * 70)
        print("\n개선 사항 요약:")
        print("1. ✅ Completed board masking (무의미한 칸 제거)")
        print("2. ✅ Perspective normalization (my vs opponent)")
        print("3. ✅ Last move plane (규칙 인식)")
        print("4. ✅ Valid board mask (합법 보드 표시)")
        print("5. ✅ Input channels: 6 → 7")
        print("\n예상 효과: 학습 효율 30-50% 향상")
        
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
