# Distributed NNUE Training

CPU worker 서버(N대)에서 데이터를 생성하고, GPU 서버(1대)에서 학습하는 구조.

## Architecture

```
[CPU Worker 0] ──┐
[CPU Worker 1] ──┤──→ shared_dir/data/*.npz ──→ [GPU Trainer]
[CPU Worker N] ──┘                                    │
       ↑                                              │
       └──── shared_dir/model/nnue_model.nnue ←───────┘
```

## Shared Directory Structure

```
shared_dir/
├── model/
│   ├── nnue_model.nnue    # Current NNUE model (binary)
│   ├── nnue_model.pt      # PyTorch weights
│   └── version.txt        # Monotonic version counter
├── data/                  # Pending data from workers
│   ├── v0001_worker0_0001.npz
│   └── ...
├── consumed/              # Processed data (moved by trainer)
├── checkpoint.npz         # Training data checkpoint
├── config.json            # Optional shared config
└── STOP                   # Trainer creates this to signal workers to stop
```

## Setup

### Shared Storage

NFS mount, rsync, or any shared filesystem between servers.

```bash
# Example: NFS
sudo mount -t nfs gpu-server:/shared/nnue /mnt/shared/nnue

# Example: create local dir for testing
mkdir -p /tmp/nnue-shared
```

### GPU Server (Trainer)

```bash
# Bootstrap with seed data (from existing checkpoint)
python -m nnue.distributed.trainer \
    --shared-dir /mnt/shared/nnue \
    --seed-data nnue/model/fin1zkq5/checkpoint.npz \
    --run-name "distributed-v1" \
    --max-loops 50 \
    --min-new-positions 50000 \
    --max-positions 2000000 \
    --train-epochs 10

# Or without seed data (random init)
python -m nnue.distributed.trainer \
    --shared-dir /mnt/shared/nnue \
    --max-loops 50 \
    --min-new-positions 50000
```

### CPU Server(s) (Workers)

```bash
# Each CPU server runs one worker with unique ID
python -m nnue.distributed.worker \
    --shared-dir /mnt/shared/nnue \
    --worker-id worker0 \
    --threads 16 \
    --games-per-batch 1000 \
    --depth 8

# Second server
python -m nnue.distributed.worker \
    --shared-dir /mnt/shared/nnue \
    --worker-id worker1 \
    --threads 32 \
    --games-per-batch 2000 \
    --depth 8
```

## Protocol

1. **Trainer** bootstraps model → exports `nnue_model.nnue` + `version.txt`
2. **Workers** poll `version.txt`, load model, generate selfplay data
3. **Workers** save `.npz` to `data/` (atomic rename to avoid partial reads)
4. **Trainer** polls `data/`, collects files, moves to `consumed/`
5. **Trainer** merges data, trains, exports new model, bumps version
6. Workers detect new version, reload model → repeat
7. **Trainer** creates `STOP` file when done → workers exit

## Scaling

- Each 16-core CPU server: ~5000 games/hour at depth 8
- 4× CPU servers: ~20,000 games/hour → ~100K positions/hour
- Trainer retrains every 50K new positions → every ~30 min
- 10 hours → ~1M positions, ~20 training loops
