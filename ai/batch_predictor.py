import threading
import queue
import time
import numpy as np
from typing import Dict, Tuple, Any

class PredictionRequest:
    """개별 prediction 요청"""
    def __init__(self, board_state, request_id):
        self.board_state = board_state
        self.request_id = request_id
        self.result = None
        self.ready_event = threading.Event()
    
    def wait_for_result(self, timeout=10):
        """결과를 기다림"""
        if self.ready_event.wait(timeout):
            return self.result
        else:
            raise TimeoutError("Prediction timeout")

class BatchPredictor:
    """여러 게임의 prediction을 batch로 처리"""
    
    def __init__(self, network, batch_size=8, wait_time=0.002, verbose=False):
        self.network = network
        self.batch_size = batch_size
        self.wait_time = wait_time  # 초 단위 (기본 2ms)
        self.verbose = verbose
        
        self.request_queue = queue.Queue()
        self.running = True
        self.worker_thread = None
        
        # 통계
        self.total_batches = 0
        self.total_predictions = 0
        self.batch_sizes = []  # 각 batch 크기 기록
        self.wait_times = []  # 각 batch 대기 시간 기록
        
    def start(self):
        """배치 처리 워커 시작"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
        self.worker_thread.start()
    
    def stop(self):
        """배치 처리 워커 중지"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        
        # 통계 출력
        if self.total_batches > 0:
            avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes)
            avg_wait_time = sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0
            print(f"\n[BatchPredictor Stats]")
            print(f"  Total batches: {self.total_batches}")
            print(f"  Total predictions: {self.total_predictions}")
            print(f"  Avg batch size: {avg_batch_size:.2f}")
            print(f"  Avg wait time: {avg_wait_time*1000:.1f}ms")
            print(f"  Batch efficiency: {avg_batch_size/self.batch_size*100:.1f}%")
    
    def predict(self, board_state, request_id=None):
        """Prediction 요청 (blocking)"""
        if request_id is None:
            request_id = id(threading.current_thread())
        
        request = PredictionRequest(board_state, request_id)
        self.request_queue.put(request)
        
        # 결과 대기
        return request.wait_for_result()
    
    def predict_batch(self, board_states):
        """여러 board states를 한 번에 처리 (배치 MCTS용)"""
        # BatchPredictor를 통해 각 board를 개별적으로 요청
        # BatchPredictor가 알아서 이들을 모아서 배치 처리함
        results = []
        for board_state in board_states:
            policy_probs, value = self.predict(board_state)
            results.append((policy_probs, value))
        
        # 결과를 numpy 배열로 변환
        policy_probs_batch = np.array([r[0] for r in results])
        values_batch = np.array([r[1] for r in results])
        
        return policy_probs_batch, values_batch
    
    def _batch_worker(self):
        """배치 처리 워커 스레드"""
        pending_requests = []
        last_process_time = time.time()
        
        while self.running:
            try:
                # 새로운 요청 수집 (non-blocking)
                try:
                    while len(pending_requests) < self.batch_size:
                        request = self.request_queue.get(timeout=0.001)
                        pending_requests.append(request)
                except queue.Empty:
                    pass
                
                # 배치가 충분히 찼거나 충분한 시간이 지났으면 처리
                current_time = time.time()
                time_since_last = current_time - last_process_time
                should_process = (
                    len(pending_requests) >= self.batch_size or
                    (len(pending_requests) > 0 and time_since_last >= self.wait_time)
                )
                
                if should_process and pending_requests:
                    self.wait_times.append(time_since_last)
                
                if should_process and pending_requests:
                    self._process_batch(pending_requests)
                    pending_requests = []
                    last_process_time = current_time
                
                # CPU 과부하 방지
                if not pending_requests:
                    time.sleep(0.001)
                    
            except Exception as e:
                print(f"Batch worker error: {e}")
                # 오류 발생 시 pending requests에 None 반환
                for req in pending_requests:
                    req.result = None
                    req.ready_event.set()
                pending_requests = []
        
        # 종료 시 남은 요청 처리
        if pending_requests:
            self._process_batch(pending_requests)
    
    def _process_batch(self, requests):
        """배치로 prediction 처리"""
        if not requests:
            return
        
        batch_size = len(requests)
        self.total_batches += 1
        self.total_predictions += batch_size
        self.batch_sizes.append(batch_size)
        
        if self.verbose and batch_size > 1:
            print(f"[Batch] Processing {batch_size} predictions")
        
        try:
            # 모든 board states를 batch로 변환
            board_states = [req.board_state for req in requests]
            
            # Batch prediction
            policy_probs_batch, values_batch = self.network.predict_batch(board_states)
            
            # 결과 분배
            for i, request in enumerate(requests):
                request.result = (policy_probs_batch[i], values_batch[i])
                request.ready_event.set()
                
        except Exception as e:
            print(f"Batch prediction error: {e}")
            import traceback
            traceback.print_exc()
            # 오류 시 None 반환
            for request in requests:
                request.result = None
                request.ready_event.set()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
