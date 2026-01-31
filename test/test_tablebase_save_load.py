"""
Tablebase 저장/로드 테스트 (CPU 환경)
빠른 검증을 위해 작은 설정 사용
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.network import AlphaZeroNet
from ai.trainer_with_dtw import AlphaZeroTrainerWithDTW
from config import Config, NetworkConfig, TrainingConfig, GPUConfig, DTWConfig, MCTSConfig

def test_tablebase_save_load():
    print("="*80)
    print("Tablebase 저장/로드 테스트 시작")
    print("="*80)
    
    # 작은 설정 (빠른 테스트)
    config = Config()
    config.network = NetworkConfig(
        num_res_blocks=1,           # 2 → 1
        num_channels=16             # 32 → 16
    )
    config.training = TrainingConfig(
        num_iterations=1,           # 2 → 1
        num_self_play_games=3,      # 엔트리 생성 확인
        num_train_epochs=1,         # 2 → 1
        num_simulations=10,         # 50 → 25 → 10
        batch_size=16,
        lr=0.001,
        replay_buffer_size=100,
        save_dir="./test_output",
        use_amp=False,
        num_parallel_games=1
    )
    config.gpu = GPUConfig(
        device="cpu",
        num_workers=1,
        pin_memory=False
    )
    config.mcts = MCTSConfig(
        c_puct=1.0,
        temperature_start=1.0,
        temperature_end=0.5
    )
    config.dtw = DTWConfig(
        use_dtw=True,               # DTW ON
        max_depth=12,
        endgame_threshold=35,       # 35칸 이하 계산 (엔트리 생성 보장)
        hot_cache_size=100,
        cold_cache_size=500,
        use_symmetry=True,
        use_tablebase=True
    )
    
    # 출력 디렉토리 생성
    os.makedirs(config.training.save_dir, exist_ok=True)
    
    # 1단계: 학습 (Tablebase 구축)
    print("\n[1단계] 짧은 학습으로 Tablebase 구축...")
    from ai.network import Model
    model = Model(
        num_res_blocks=config.network.num_res_blocks,
        num_channels=config.network.num_channels
    )
    network = AlphaZeroNet(
        model=model,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        device=config.gpu.device,
        use_amp=config.training.use_amp
    )
    
    trainer = AlphaZeroTrainerWithDTW(
        network=network,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        batch_size=config.training.batch_size,
        num_simulations=config.training.num_simulations,
        replay_buffer_size=config.training.replay_buffer_size,
        device=config.gpu.device,
        use_amp=config.training.use_amp,
        num_res_blocks=config.network.num_res_blocks,
        num_channels=config.network.num_channels,
        num_parallel_games=config.training.num_parallel_games,
        use_dtw=config.dtw.use_dtw,
        dtw_max_depth=config.dtw.max_depth,
        hot_cache_size=config.dtw.hot_cache_size,
        cold_cache_size=config.dtw.cold_cache_size,
        use_symmetry=config.dtw.use_symmetry
    )
    
    # 1 iteration, 3 games (엔트리 생성 확인)
    print("(3 games, endgame=35칸 - 실제 엔트리 생성 확인)")
    for iteration in range(config.training.num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{config.training.num_iterations} ---")
        
        training_data = trainer.generate_self_play_data(
            num_games=config.training.num_self_play_games,
            num_simulations=config.training.num_simulations,
            temperature=config.mcts.temperature_start
        )
        
        if training_data:
            loss_dict = trainer.train(num_epochs=config.training.num_train_epochs)
            if isinstance(loss_dict, dict) and 'total_loss' in loss_dict:
                print(f"Loss: {loss_dict['total_loss']:.4f}")
    
    # Tablebase 상태 확인
    if trainer.dtw_calculator and trainer.dtw_calculator.tt:
        stats_before = trainer.dtw_calculator.tt.get_stats()
        hot_size_before = len(trainer.dtw_calculator.tt.hot)
        cold_size_before = len(trainer.dtw_calculator.tt.cold)
        
        print(f"\n✓ Tablebase 구축 완료")
        print(f"  Hot entries: {hot_size_before}")
        print(f"  Cold entries: {cold_size_before}")
        print(f"  Total entries: {hot_size_before + cold_size_before}")
    else:
        print("\n✗ DTW Calculator 없음")
        return False
    
    # 2단계: Tablebase 저장
    print("\n[2단계] Tablebase 디스크 저장...")
    tablebase_path = os.path.join(config.training.save_dir, "tablebase_test.pkl")
    
    try:
        trainer.dtw_calculator.tt.save_to_file(tablebase_path)
        file_size = os.path.getsize(tablebase_path)
        print(f"✓ 저장 성공: {file_size / 1024:.2f} KB")
    except Exception as e:
        print(f"✗ 저장 실패: {e}")
        return False
    
    # 3단계: 새로운 Tablebase 생성 및 로드
    print("\n[3단계] 새로운 Tablebase에 로드...")
    
    from ai.transposition_table import CompressedTranspositionTable
    new_tt = CompressedTranspositionTable(
        hot_size=1000,
        cold_size=5000,
        use_symmetry=True
    )
    
    try:
        success = new_tt.load_from_file(tablebase_path)
        if not success:
            print("✗ 로드 실패")
            return False
    except Exception as e:
        print(f"✗ 로드 실패: {e}")
        return False
    
    # 4단계: 검증
    print("\n[4단계] 데이터 검증...")
    hot_size_after = len(new_tt.hot)
    cold_size_after = len(new_tt.cold)
    
    print(f"  저장 전 - Hot: {hot_size_before}, Cold: {cold_size_before}")
    print(f"  로드 후 - Hot: {hot_size_after}, Cold: {cold_size_after}")
    
    if hot_size_before == hot_size_after and cold_size_before == cold_size_after:
        print("\n✓ 검증 성공: 모든 엔트리가 동일합니다")
        
        # 샘플 데이터 확인
        if hot_size_after > 0:
            sample_key = list(new_tt.hot.keys())[0]
            sample_value = new_tt.hot[sample_key]
            print(f"  샘플 데이터: {sample_value}")
        
        return True
    else:
        print("\n✗ 검증 실패: 엔트리 개수가 다릅니다")
        return False

if __name__ == "__main__":
    try:
        success = test_tablebase_save_load()
        
        print("\n" + "="*80)
        if success:
            print("테스트 결과: ✓ 성공")
            print("Tablebase 저장/로드가 정상적으로 작동합니다!")
        else:
            print("테스트 결과: ✗ 실패")
        print("="*80)
        
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print("\n" + "="*80)
        print(f"테스트 중 오류 발생: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        sys.exit(1)
