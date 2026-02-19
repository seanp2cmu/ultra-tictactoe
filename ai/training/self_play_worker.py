"""CPU-only self-play worker process.

Runs MCTS (select / expand / backprop / game management) without touching
the GPU.  Inference requests are sent to the main process via SharedMemory
+ queue signalling.
"""


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
    dtw_cache_path: str = None,
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
        if dtw_cache_path and dtw.tt:
            try:
                dtw.tt.load_from_file(dtw_cache_path, quiet=True)
            except Exception:
                pass
        _initial_hot_size = len(dtw.tt.hot) if dtw and dtw.tt else 0
        _initial_cold_size = len(dtw.tt.cold) if dtw and dtw.tt else 0
        _initial_keys = set(dtw.tt.hot.keys()) | set(dtw.tt.cold.keys()) if dtw and dtw.tt else set()

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
                    game_temps.append(temperature if len(g['history']) < 20 else 0)
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
        # Collect new DTW entries computed by this worker
        new_dtw_entries = {}
        if dtw and dtw.tt:
            for k, v in dtw.tt.hot.items():
                if k not in _initial_keys:
                    new_dtw_entries[k] = v
            for k, v in dtw.tt.cold.items():
                if k not in _initial_keys:
                    new_dtw_entries[k] = dtw.tt.decompress_entry(v)
        req_queue.put(('done', worker_id, states, policies, values, game_ids, dtw_stats, new_dtw_entries))

        _shm_in.close()
        _shm_pol.close()
        _shm_val.close()
        _shm_inv.close()

    except Exception as e:
        import traceback
        traceback.print_exc()
        req_queue.put(('error', worker_id, str(e)))
