"""Run management: folder structure, runs.json, checkpoint selection."""
import os
import json
import glob
import datetime


RUNS_FILE = "runs.json"


def load_runs(base_dir: str) -> dict:
    path = os.path.join(base_dir, RUNS_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_runs(base_dir: str, runs: dict):
    path = os.path.join(base_dir, RUNS_FILE)
    with open(path, 'w') as f:
        json.dump(runs, f, indent=2)


def select_run_and_checkpoint(base_dir: str) -> tuple:
    """
    Interactive run/checkpoint selection.
    Returns: (run_id, run_name, checkpoint_path, is_new_run)
    """
    runs = load_runs(base_dir)
    
    existing_runs = []
    for run_id, info in runs.items():
        run_dir = os.path.join(base_dir, run_id)
        if os.path.isdir(run_dir):
            pts = sorted(glob.glob(os.path.join(run_dir, '*.pt')))
            if pts:
                existing_runs.append((run_id, info, pts))
    
    # Collect all checkpoints across all runs for fork option
    all_checkpoints = []
    for run_id, info, pts in existing_runs:
        name = info.get('name', run_id[:8])
        for pt in pts:
            all_checkpoints.append((run_id, name, pt))
    
    print("\n" + "=" * 60)
    print("Select training run")
    print("=" * 60)
    print("  0: New run (start from scratch)")
    if all_checkpoints:
        print("  F: Fork (new run from existing checkpoint)")
    for i, (run_id, info, pts) in enumerate(existing_runs, 1):
        name = info.get('name', run_id[:8])
        iters = info.get('last_iteration', '?')
        n_ckpts = len(pts)
        print(f"  {i}: [Resume] {name} (iter {iters}, {n_ckpts} checkpoints)")
    print("=" * 60)
    
    while True:
        try:
            raw = input(f"(0-{len(existing_runs)}, F=fork): ").strip()
            
            if raw.lower() == 'f' and all_checkpoints:
                # Fork: new run with weights from existing checkpoint
                print("\n  Available checkpoints:")
                for j, (rid, rname, pt) in enumerate(all_checkpoints):
                    fname = os.path.basename(pt)
                    size_mb = os.path.getsize(pt) / 1024 / 1024
                    print(f"    {j}: {rname}/{fname} ({size_mb:.1f} MB)")
                ckpt_choice = int(input(f"  checkpoint (0-{len(all_checkpoints)-1}): ").strip())
                ckpt_choice = max(0, min(ckpt_choice, len(all_checkpoints) - 1))
                _, src_name, ckpt_path = all_checkpoints[ckpt_choice]
                run_name = input(f"  New run name (fork from {src_name}): ").strip()
                if not run_name:
                    run_name = f"fork-{src_name}-{datetime.datetime.now().strftime('%m%d-%H%M')}"
                print(f"  -> Fork: {run_name} from {os.path.basename(ckpt_path)}")
                # Return is_new_run=True but with checkpoint_path set (fork mode)
                return None, run_name, ckpt_path, True
            
            choice = int(raw)
            if choice == 0:
                run_name = input("Run name: ").strip()
                if not run_name:
                    run_name = f"run-{datetime.datetime.now().strftime('%m%d-%H%M')}"
                return None, run_name, None, True
            elif 1 <= choice <= len(existing_runs):
                run_id, info, pts = existing_runs[choice - 1]
                run_name = info.get('name', run_id[:8])
                print(f"\n-> Resuming: {run_name}")
                print("  Checkpoints:")
                for j, pt in enumerate(pts):
                    fname = os.path.basename(pt)
                    size_mb = os.path.getsize(pt) / 1024 / 1024
                    print(f"    {j}: {fname} ({size_mb:.1f} MB)")
                ckpt_choice = int(input(f"  checkpoint (0-{len(pts)-1}): ").strip())
                ckpt_path = pts[max(0, min(ckpt_choice, len(pts) - 1))]
                print(f"  -> {os.path.basename(ckpt_path)}")
                return run_id, run_name, ckpt_path, False
            else:
                print("Invalid choice")
        except ValueError:
            print("Enter a number (or F to fork)")
        except KeyboardInterrupt:
            print("\nCancelled")
            exit(0)


def register_run(base_dir: str, run_id: str, run_name: str):
    """Register a new run in runs.json."""
    runs = load_runs(base_dir)
    if run_id not in runs:
        runs[run_id] = {'name': run_name, 'created': datetime.datetime.now().isoformat()}
    runs[run_id]['name'] = run_name
    save_runs(base_dir, runs)


def update_run_iteration(base_dir: str, run_id: str, iteration: int):
    """Update last_iteration in runs.json."""
    runs = load_runs(base_dir)
    if run_id in runs:
        runs[run_id]['last_iteration'] = iteration
        save_runs(base_dir, runs)


