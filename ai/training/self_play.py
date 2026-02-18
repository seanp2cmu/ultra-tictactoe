"""Multi-process self-play with centralised GPU inference.

Architecture:
  - N CPU worker processes run MCTS (select / expand / backprop / game mgmt).
  - Workers do NOT touch the GPU.  When they need inference they write an
    encoded tensor into per-worker SharedMemory, signal the main process,
    and block until the result is ready.
  - The main process collects pending requests, runs a single batched TRT
    (or PyTorch) inference call, writes results back into SharedMemory, and
    signals the workers.

Communication per worker:
  shm_in   (max_leaves × 7 × 9 × 9 float32)  — worker writes encoded tensor
  shm_pol  (max_leaves × 81 float32)          — main writes policy result
  shm_val  (max_leaves × 1  float32)          — main writes value result
  shm_inv  (max_leaves × 81 int64)            — worker writes inverse indices
  resp_event — main sets when result is ready
  req_queue  — shared: worker → main messages
"""

import os
import time
import numpy as np
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from typing import List, Tuple, Optional

_mp = get_context('spawn')

# Max leaves any single worker can submit in one inference request
_MAX_LEAVES_PER_WORKER = 4096


# ─────────────────────────────────────────────────────────────────────────────
# Worker process  (CPU only — no GPU, no PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def _cpu_worker(
    worker_id: int,
    shm_names: dict,      # {'in':..,'pol':..,'val':..,'inv':..}
    max_leaves: int,
    req_queue,             # shared queue: worker → main
    resp_event,            # per-worker Event: main sets when result ready
    num_games: int,
    num_simulations: int,
    parallel_games: int,
    temperature: float,
    game_id_start: int,
    endgame_threshold: int,
    c_puct: float,
):
    """CPU-only self-play worker. Calls main for every inference."""
    try:
        import numpy as _np
        from multiprocessing.shared_memory import SharedMemory as _SHM
        from game import Board
        from ai.mcts.node_cy import (
            NodeCy as Node,
            select_multi_leaves_cy,
            expand_backprop_batch_cy,
            revert_vl_batch_cy,
        )
        from utils import BoardEncoder
        from ai.endgame import DTWCalculator

        # Attach to shared memory
        _shm_in  = _SHM(name=shm_names['in'],  create=False)
        _shm_pol = _SHM(name=shm_names['pol'], create=False)
        _shm_val = _SHM(name=shm_names['val'], create=False)
        _shm_inv = _SHM(name=shm_names['inv'], create=False)

        buf_in  = _np.ndarray((max_leaves, 7, 9, 9), _np.float32, buffer=_shm_in.buf)
        buf_pol = _np.ndarray((max_leaves, 81),       _np.float32, buffer=_shm_pol.buf)
        buf_val = _np.ndarray((max_leaves, 1),        _np.float32, buffer=_shm_val.buf)
        buf_inv = _np.ndarray((max_leaves, 81),       _np.int64,   buffer=_shm_inv.buf)

        dtw = DTWCalculator(use_cache=True, endgame_threshold=endgame_threshold)

        def _request_inference(boards):
            """Encode boards, write to shm, signal main, wait for result."""
            if not boards:
                return _np.empty((0, 81), _np.float32), _np.empty((0, 1), _np.float32)
            batch_tensor, inv_idx = BoardEncoder.to_inference_tensor_batch(boards)
            bs = batch_tensor.shape[0]
            buf_in[:bs] = batch_tensor
            buf_inv[:bs] = _np.asarray(inv_idx, dtype=_np.int64)[:bs]
            req_queue.put(('infer', worker_id, bs))
            resp_event.wait()
            resp_event.clear()
            policies = buf_pol[:bs].copy()
            values = buf_val[:bs].copy()
            inv = buf_inv[:bs]
            row_idx = _np.arange(bs)[:, None]
            return policies[row_idx, inv], values

        # ── Self-play loop ──
        all_data = []
        games_completed = 0
        games_started = 0
        current_game_id = game_id_start

        def _new_game():
            nonlocal current_game_id, games_started
            g = {'board': Board(), 'history': [], 'done': False,
                 'game_id': current_game_id}
            current_game_id += 1
            games_started += 1
            return g

        pool_size = min(parallel_games, num_games)
        pool = [_new_game() for _ in range(pool_size)]

        while pool:
            # Separate endgame vs MCTS
            endgame_games = []
            mcts_games = []
            for g in pool:
                if dtw and dtw.is_endgame(g['board']):
                    endgame_games.append(g)
                else:
                    mcts_games.append(g)

            finished_ids = set()
            for game in endgame_games:
                dtw_result = dtw.calculate_dtw(game['board'], need_best_move=False)
                if dtw_result is not None:
                    result_code, _, _ = dtw_result
                    game['done'] = True
                    board = game['board']
                    fv = 1.0 if result_code == 1 else (-1.0 if result_code == -1 else 0.0)
                    for step in game['history']:
                        value = fv if step['player'] == board.current_player else -fv
                        all_data.append((step['state'], step['policy'], value, game['game_id']))
                    finished_ids.add(id(game))
                else:
                    mcts_games.append(game)

            if mcts_games:
                game_temps = []
                for g in mcts_games:
                    game_temps.append(temperature if len(g['history']) < 16 else 0)
                batch_temp = sum(game_temps) / len(game_temps) if game_temps else 0
                add_noise = (batch_temp > 0)

                roots = [Node(g['board'], _clone=False) for g in mcts_games]

                # Initial expansion
                boards_init = [g['board'] for g in mcts_games]
                policies_init, _ = _request_inference(boards_init)
                if add_noise:
                    noise = _np.random.dirichlet([0.3] * 81, size=len(roots))
                    policies_init = 0.75 * policies_init + 0.25 * noise
                policies_init = policies_init.astype(_np.float32)
                for i, root in enumerate(roots):
                    root.expand_numpy(policies_init[i])

                # MCTS rounds
                n_games_mcts = len(roots)
                all_indices = list(range(n_games_mcts))
                max_bs = max_leaves
                max_lpg = max(1, max_bs // max(1, n_games_mcts))
                max_lpg_depth = max(1, num_simulations // 10)
                lpg = max(1, min(num_simulations, min(max_lpg, max_lpg_depth)))
                num_rounds = max(1, (num_simulations + lpg - 1) // lpg)
                lpg = max(1, num_simulations // num_rounds)
                leftover = num_simulations - lpg * num_rounds

                for rnd in range(num_rounds):
                    k = lpg + (1 if rnd < leftover else 0)
                    leaves, leaf_boards = select_multi_leaves_cy(
                        roots, all_indices, k, c_puct)
                    if not leaf_boards:
                        if leaves:
                            revert_vl_batch_cy(leaves)
                        break
                    policies_batch, values_batch = _request_inference(leaf_boards)
                    n_leaves = len(leaves)
                    vals_scaled = _np.ascontiguousarray(
                        values_batch[:n_leaves].ravel(), dtype=_np.float32)
                    pols_f32 = _np.ascontiguousarray(
                        policies_batch[:n_leaves], dtype=_np.float32)
                    expand_backprop_batch_cy(leaves, pols_f32, vals_scaled)

                # Select moves
                for gi, (game, root) in enumerate(zip(mcts_games, roots)):
                    visits = _np.zeros(81, dtype=_np.float32)
                    for action, child in root.children.items():
                        visits[action] = child.visits
                    if visits.sum() == 0:
                        policy = _np.zeros(81, dtype=_np.float32)
                        for action, child in root.children.items():
                            policy[action] = child.prior_prob
                        total = policy.sum()
                        if total > 0:
                            policy /= total
                        else:
                            legal = game['board'].get_legal_moves()
                            for r, c in legal:
                                policy[r * 9 + c] = 1.0 / len(legal)
                        action = int(_np.argmax(policy)) if batch_temp < 0.01 else int(_np.random.choice(81, p=policy))
                    elif batch_temp < 0.01:
                        action = int(_np.argmax(visits))
                        policy = _np.zeros(81, dtype=_np.float32)
                        policy[action] = 1.0
                    else:
                        with _np.errstate(divide='ignore'):
                            log_v = _np.where(visits > 0, _np.log(visits), -1e9)
                        log_t = log_v / batch_temp
                        log_t -= log_t.max()
                        vt = _np.exp(log_t)
                        total = vt.sum()
                        policy = vt / total if total > 0 and _np.isfinite(total) else _np.ones(81) / 81
                        policy = policy / policy.sum()
                        action = int(_np.random.choice(81, p=policy))

                    board = game['board']
                    canonical_tensor, canonical_policy = BoardEncoder.to_training_tensor(board, policy)
                    game['history'].append({
                        'state': canonical_tensor,
                        'policy': canonical_policy,
                        'player': board.current_player,
                    })
                    row, col = action // 9, action % 9
                    try:
                        board.make_move(row, col)
                    except Exception:
                        legal = board.get_legal_moves()
                        if legal:
                            board.make_move(*legal[0])
                    if board.is_game_over():
                        game['done'] = True
                        if board.winner in (None, -1, 3):
                            result = 0.0
                        else:
                            result = 1.0 if board.winner == 1 else -1.0
                        for step in game['history']:
                            value = result if step['player'] == 1 else -result
                            all_data.append((step['state'], step['policy'], value, game['game_id']))
                        finished_ids.add(id(game))

            # Replace finished games
            if finished_ids:
                new_pool = []
                batch_done = 0
                for g in pool:
                    if id(g) in finished_ids:
                        games_completed += 1
                        batch_done += 1
                        if games_started < num_games:
                            new_pool.append(_new_game())
                    else:
                        new_pool.append(g)
                pool = new_pool
                if batch_done > 0:
                    req_queue.put(('progress', worker_id, batch_done))

        # Send results back
        if all_data:
            states = _np.array([d[0] for d in all_data], dtype=_np.float32)
            policies = _np.array([d[1] for d in all_data], dtype=_np.float32)
            values = _np.array([d[2] for d in all_data], dtype=_np.float32)
            game_ids = _np.array([d[3] for d in all_data], dtype=_np.int64)
        else:
            states = _np.empty((0, 7, 9, 9), dtype=_np.float32)
            policies = _np.empty((0, 81), dtype=_np.float32)
            values = _np.empty(0, dtype=_np.float32)
            game_ids = _np.empty(0, dtype=_np.int64)

        dtw_stats = dtw.get_stats() if dtw else {}
        req_queue.put(('done', worker_id, states, policies, values, game_ids, dtw_stats))

        _shm_in.close()
        _shm_pol.close()
        _shm_val.close()
        _shm_inv.close()

    except Exception as e:
        import traceback
        traceback.print_exc()
        req_queue.put(('error', worker_id, str(e)))


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator  (main process — owns GPU / TRT)
# ─────────────────────────────────────────────────────────────────────────────

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
                endgame_threshold, c_puct,
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
    errors = []
    alive_count = num_workers
    last_infer_time = 0.005  # initial estimate: 5ms

    _games_done = 0
    _last_log_pct = 0

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
            nonlocal alive_count, _games_done, _last_log_pct
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
                worker_ctx[wid]['alive'] = False
                alive_count -= 1
            elif cmd == 'error':
                errors.append(m[2])
                worker_ctx[m[1]]['alive'] = False
                alive_count -= 1
            elif cmd == 'progress':
                _games_done += m[2]

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
        for ds in all_dtw_stats:
            for k, v in ds.items():
                if isinstance(v, (int, float)):
                    merged_dtw[k] = merged_dtw.get(k, 0) + v
        # Recompute hit_rate from totals
        total_q = merged_dtw.get('hot_hits', 0) + merged_dtw.get('cold_hits', 0) + merged_dtw.get('misses', 0)
        if total_q > 0:
            merged_dtw['hit_rate'] = (merged_dtw.get('hot_hits', 0) + merged_dtw.get('cold_hits', 0)) / total_q

    timing = {
        'total_time': wall_time,
        'network_time': total_network_time,
        'mcts_overhead': wall_time - total_network_time,
        'batches': total_batches,
        'games': num_games,
        'moves': len(states),
        'dtw_stats': merged_dtw,
    }
    return (states, policies, values, game_ids), timing
