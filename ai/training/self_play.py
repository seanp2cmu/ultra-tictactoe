"""Multi-process self-play orchestrator with centralised GPU inference.

Architecture:
  - N CPU worker processes run MCTS (select / expand / backprop / game mgmt).
  - Workers do NOT touch the GPU.  When they need inference they write an
    encoded tensor into per-worker SharedMemory, signal the main process,
    and block until the result is ready.
  - The main process collects pending requests, runs a single batched TRT
    (or PyTorch) inference call, writes results back into SharedMemory, and
    signals the workers.

Communication per worker:
  shm_in   (max_leaves x 7 x 9 x 9 float32)  — worker writes encoded tensor
  shm_pol  (max_leaves x 81 float32)          — main writes policy result
  shm_val  (max_leaves x 1  float32)          — main writes value result
  shm_inv  (max_leaves x 81 int64)            — worker writes inverse indices
  resp_event — main sets when result is ready
  req_queue  — shared: worker → main messages
"""

import os
import time
import numpy as np
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from typing import List, Tuple, Optional

from .self_play_worker import _cpu_worker

_mp = get_context('spawn')

# Max leaves any single worker can submit in one inference request
_MAX_LEAVES_PER_WORKER = 4096


def run_multiprocess_self_play(
    network,
    num_games: int,
    num_simulations: int,
    parallel_games: int,
    temperature: float,
    game_id_start: int,
    num_workers: int = 4,
    endgame_threshold: int = 15,
    c_puct: float = 1.0,
    dtw_cache_path: str = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], dict]:
    """Run multi-process self-play with centralised GPU inference.

    Returns:
        (states, policies, values, game_ids), timing_dict
    """
    max_leaves = _MAX_LEAVES_PER_WORKER
    pid = os.getpid()

    # ── Shared request queue (all workers → main) ──
    req_queue = _mp.Queue()

    # ── Allocate per-worker SharedMemory ──
    worker_ctx = []
    all_shms = []
    for wid in range(num_workers):
        names = {
            'in':  f'mpsp_in_{pid}_{wid}',
            'pol': f'mpsp_pol_{pid}_{wid}',
            'val': f'mpsp_val_{pid}_{wid}',
            'inv': f'mpsp_inv_{pid}_{wid}',
        }
        sizes = {
            'in':  max_leaves * 7 * 9 * 9 * 4,
            'pol': max_leaves * 81 * 4,
            'val': max_leaves * 1 * 4,
            'inv': max_leaves * 81 * 8,
        }
        for n in names.values():
            try:
                s = SharedMemory(name=n, create=False)
                s.close(); s.unlink()
            except FileNotFoundError:
                pass

        shms = {k: SharedMemory(name=names[k], create=True, size=sizes[k])
                for k in names}
        bufs = {
            'in':  np.ndarray((max_leaves, 7, 9, 9), np.float32, buffer=shms['in'].buf),
            'pol': np.ndarray((max_leaves, 81),       np.float32, buffer=shms['pol'].buf),
            'val': np.ndarray((max_leaves, 1),        np.float32, buffer=shms['val'].buf),
            'inv': np.ndarray((max_leaves, 81),       np.int64,   buffer=shms['inv'].buf),
        }
        resp_event = _mp.Event()

        worker_ctx.append({
            'wid': wid, 'names': names, 'shms': shms, 'bufs': bufs,
            'resp_event': resp_event, 'alive': True,
        })
        all_shms.extend(shms.values())

    # ── Divide games among workers ──
    base = num_games // num_workers
    remainder = num_games % num_workers
    games_per_worker = [base + (1 if i < remainder else 0) for i in range(num_workers)]
    worker_parallel = max(32, parallel_games // num_workers)

    # ── Spawn workers ──
    processes = []
    gid = game_id_start
    for wid in range(num_workers):
        ng = games_per_worker[wid]
        ctx = worker_ctx[wid]
        p = _mp.Process(
            target=_cpu_worker,
            args=(
                wid,
                ctx['names'],
                max_leaves,
                req_queue, ctx['resp_event'],
                ng, num_simulations, worker_parallel, temperature, gid,
                endgame_threshold, c_puct, dtw_cache_path,
            ),
            daemon=True,
        )
        p.start()
        processes.append(p)
        gid += ng

    _mp_info = (f'[MP-SelfPlay] {num_workers} workers, {num_games} games, '
                f'{num_simulations} sims, {worker_parallel} parallel/worker')

    # ── Main inference loop ──
    # Strategy: block for first message, drain queue, then run inference.
    # Workers are synchronous (submit → wait → CPU work → submit → wait),
    # so multiple workers naturally stagger their requests.
    t0 = time.perf_counter()
    total_network_time = 0.0
    total_batches = 0
    all_results = {}
    all_dtw_stats = []  # collect DTW stats from workers
    all_new_dtw_entries = {}  # collect new DTW cache entries from workers
    errors = []
    alive_count = num_workers
    last_infer_time = 0.005  # initial estimate: 5ms

    _games_done = 0

    from tqdm import tqdm as _tqdm
    sp_pbar = _tqdm(total=num_games, desc="Self-Play", ncols=100, leave=False, position=1,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    while alive_count > 0:
        try:
            msg = req_queue.get(timeout=300)
        except Exception:
            print(f'[MP-SelfPlay] Timeout, alive={alive_count}', flush=True)
            break

        # pending: list of (wid, batch_size)
        pending = []
        pending_wids = set()

        def _process(m):
            nonlocal alive_count, _games_done
            cmd = m[0]
            if cmd == 'infer':
                pending.append((m[1], m[2]))
                pending_wids.add(m[1])
            elif cmd == 'done':
                wid = m[1]
                st, po, va, gi = m[2], m[3], m[4], m[5]
                all_results[wid] = (st, po, va, gi)
                if len(m) > 6 and m[6]:
                    all_dtw_stats.append(m[6])
                if len(m) > 7 and m[7]:
                    all_new_dtw_entries.update(m[7])
                worker_ctx[wid]['alive'] = False
                alive_count -= 1
            elif cmd == 'error':
                errors.append(m[2])
                worker_ctx[m[1]]['alive'] = False
                alive_count -= 1
            elif cmd == 'progress':
                _games_done += m[2]
                sp_pbar.update(m[2])

        _process(msg)

        # Drain all immediately available messages
        while True:
            try:
                _process(req_queue.get_nowait())
            except Exception:
                break

        # Adaptive batching: if we have infer requests but not all workers
        # have reported in, wait up to last_infer_time for stragglers.
        # This costs nothing because inference would take that long anyway.
        if pending and len(pending_wids) < alive_count:
            deadline = time.perf_counter() + last_infer_time
            while len(pending_wids) < alive_count:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    _process(req_queue.get(timeout=remaining))
                except Exception:
                    break

        if not pending:
            continue

        # Batch all pending requests into one inference call
        total_bs = sum(bs for _, bs in pending)
        if total_bs == 0:
            for wid, _ in pending:
                worker_ctx[wid]['resp_event'].set()
            continue

        # Gather tensors
        batch_tensor = np.empty((total_bs, 7, 9, 9), dtype=np.float32)
        offsets = []
        offset = 0
        for wid, bs in pending:
            ctx = worker_ctx[wid]
            batch_tensor[offset:offset+bs] = ctx['bufs']['in'][:bs]
            offsets.append((wid, offset, bs))
            offset += bs

        # Run inference
        t_net = time.perf_counter()
        policy_probs, values = network._infer_chunked(batch_tensor)
        dt = time.perf_counter() - t_net
        total_network_time += dt
        total_batches += 1
        # Update adaptive wait time (EMA)
        last_infer_time = 0.7 * last_infer_time + 0.3 * dt

        # Scatter results back
        for wid, off, bs in offsets:
            ctx = worker_ctx[wid]
            ctx['bufs']['pol'][:bs] = policy_probs[off:off+bs]
            ctx['bufs']['val'][:bs] = values[off:off+bs]
            ctx['resp_event'].set()

    sp_pbar.close()

    # Wait for processes
    for p in processes:
        p.join(timeout=10)

    wall_time = time.perf_counter() - t0

    # Cleanup SharedMemory
    for shm in all_shms:
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass

    if errors:
        try:
            from tqdm import tqdm as _tq
            _tq.write(f'[MP-SelfPlay] {len(errors)} worker(s) failed: {errors}')
        except ImportError:
            print(f'[MP-SelfPlay] {len(errors)} worker(s) failed: {errors}')

    # Merge results
    all_s, all_p, all_v, all_g = [], [], [], []
    for wid in sorted(all_results.keys()):
        s, p, v, g = all_results[wid]
        if len(s) > 0:
            all_s.append(s); all_p.append(p); all_v.append(v); all_g.append(g)

    if all_s:
        states = np.concatenate(all_s)
        policies = np.concatenate(all_p)
        values = np.concatenate(all_v)
        game_ids = np.concatenate(all_g)
    else:
        states = np.empty((0, 7, 9, 9), dtype=np.float32)
        policies = np.empty((0, 81), dtype=np.float32)
        values = np.empty(0, dtype=np.float32)
        game_ids = np.empty(0, dtype=np.int64)

    # Aggregate DTW stats from all workers
    merged_dtw = {}
    if all_dtw_stats:
        # Keys to sum across workers (counters)
        sum_keys = {'hot_hits', 'cold_hits', 'misses', 'evictions', 'symmetry_saves', 'dtw_searches'}
        # Keys to take max (cache sizes - workers share same base cache)
        max_keys = {'hot_entries', 'cold_entries'}
        for ds in all_dtw_stats:
            for k, v in ds.items():
                if not isinstance(v, (int, float)):
                    continue
                if k in max_keys:
                    merged_dtw[k] = max(merged_dtw.get(k, 0), v)
                elif k in sum_keys:
                    merged_dtw[k] = merged_dtw.get(k, 0) + v
                else:
                    merged_dtw[k] = merged_dtw.get(k, 0) + v
        # Recompute hit_rate and total_queries for logger compatibility
        total_q = merged_dtw.get('hot_hits', 0) + merged_dtw.get('cold_hits', 0) + merged_dtw.get('misses', 0)
        merged_dtw['total_queries'] = total_q
        if total_q > 0:
            hr = (merged_dtw.get('hot_hits', 0) + merged_dtw.get('cold_hits', 0)) / total_q
            merged_dtw['hit_rate'] = f"{hr:.2%}"
        else:
            merged_dtw['hit_rate'] = "0.00%"

    timing = {
        'total_time': wall_time,
        'network_time': total_network_time,
        'mcts_overhead': wall_time - total_network_time,
        'batches': total_batches,
        'games': num_games,
        'moves': len(states),
        'dtw_stats': merged_dtw,
        'new_dtw_entries': all_new_dtw_entries,
    }
    return (states, policies, values, game_ids), timing
