# Snake AI with Deep RL

This folder is a self-contained Deep RL follow-up to the tabular Snake AI project.
The current version supports **CPU or CUDA** training while keeping the same teaching-oriented dashboard and DQN flow.
It also supports a separate **parallel training mode** for higher-throughput bulk training with a short rendered evaluation tail.

## What this project shows

- the richer 18-feature state vector seen by the DQN
- live action selection and Q-values
- prioritized replay-memory driven training
- Double-DQN target updates
- target-network syncing
- a local tabular baseline for comparison
- a live neural-network view across configurable hidden layers
- a separate parallel trainer that batches multiple Snake environments for speed

## Files

- `snake_game.py` - Snake environment and pygame visualizer
- `dqn_agent.py` - configurable CPU/CUDA DQN, replay buffer, inspection payload, checkpointing
- `tabular_agent.py` - local tabular baseline for comparison
- `metrics_utils.py` - JSONL metric helpers
- `dashboard.py` - UI controls and dashboard payload builder
- `app_core.py` - training and viewer runtime
- `train.py` - Deep RL training entrypoint
- `visualizer.py` - checkpoint visualizer entrypoint
- `HOW_DEEP_RL_WORKS.md` - algorithm explanation
- `SESSION_HANDOFF.md` - progress tracker for future Codex sessions

## Install

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On this machine, `py -3` resolves to Python 3.14 while `python` and `py -3.12` resolve to Python 3.12.
Use `python` or `py -3.12` for this project so you stay on the interpreter that already has the project dependencies.

## Train

Default architecture:

```powershell
python train.py --episodes 300 --comparison-mode --device auto
```

Custom hidden layers:

```powershell
python train.py --episodes 150 --hidden-layers 64 --device cpu
python train.py --episodes 150 --hidden-layers 128,128 --device auto
python train.py --episodes 150 --hidden-layers 256,128,64 --comparison-mode --device cuda
```

Useful options:

```powershell
python train.py --episodes 100 --resume --device auto
python train.py --episodes 100 --resume --hidden-layers 128,128 --device cuda
python train.py --episodes 300 --fast-mode --device auto
python train.py --episodes 400 --trainer-mode parallel --parallel-envs 8 --eval-tail-episodes 3 --device auto
```

Notes:

- if you resume from a checkpoint, the checkpoint architecture is used automatically
- if you explicitly pass `--hidden-layers` while resuming, it must match the checkpoint
- older checkpoints without architecture metadata are treated as `128,128`
- `--device auto` uses CUDA when available, otherwise CPU
- `--device cuda` fails clearly if CUDA is unavailable on your machine
- fresh CUDA sessions default to `GPU + Parallel`, while CPU-only sessions default to `CPU + Single`
- `--fast-mode` is the single-snake speedup path: training runs hidden and the newest 3 runs are replayable afterward
- fast mode skips tabular-baseline generation and hides baseline series while it is active
- `--trainer-mode single` is the learning or inspection path for watching one snake train live
- `--trainer-mode parallel` is the speed or throughput path with batched workers
- `--parallel-envs` controls how many Snake environments the parallel trainer steps at once
- `--eval-tail-episodes` controls how many final rendered evaluation runs are shown and replayable in parallel mode
- if `--parallel-envs` is omitted, parallel mode now defaults to `64` on CUDA and `16` on CPU
- replay is now proportional prioritized replay, so high-TD-error transitions are revisited more often than routine ones
- replay capacity now defaults to `200000`
- parallel mode now uses a larger warmup (`10000`) and a smaller batch size (`512`) to improve sample efficiency
- the DQN now uses automatic plateau recovery: when training stalls for long enough, epsilon reheats upward temporarily before decaying again
- the default stall threshold is now `150`, and the GUI exposes it as an editable lower-dock input
- repeated spin-loop behavior now gets a small automatic penalty so the snake is nudged away from cycling in place
- non-terminal moves now get a small Manhattan-distance shaping term (`0.05 * (previous_distance - current_distance)`) so the snake gets a denser signal for moving toward food
- resume restores trainer-mode metadata from checkpoints unless you explicitly override it on the CLI
- when a fast-mode session finishes, the UI can replay the last 3 recorded runs from both the board overlay and a `Recent Replays` card in `Overview`
- the `Recent Replays` card explains when replay is unavailable and what will be captured next
- fast-mode single training runs uncapped and hidden, then hands off to the post-run results and replay screen
- parallel mode keeps bulk training speed-first, updates the graph on completed episodes, and renders only the final evaluation tail runs
- during parallel bulk mode, the left board switches to an aggregate status panel so it does not pretend to show one specific run
- in parallel mode, only the newest 3 evaluation-tail replays are kept in memory
- the parallel trainer now reuses preallocated state buffers instead of rebuilding NumPy state batches every step

## Visualizer

```powershell
python visualizer.py --checkpoint dqn_checkpoint.pt --metrics-log training_metrics.jsonl --device auto
```

## Views

- `Overview` - KPI cards, decision summary, controls, comparison graph, and top active features
- `Network` - layer-by-layer heatmaps plus compact connection-strength blocks
- `Algorithm` - short cards for the RL loop, current transition, Double DQN update, and why prioritized replay/target networks help

## Controls

- `Enter` click or trigger `Start` to begin a rendered training session
- `Space` pause or resume
- `Tab` cycle through `Overview`, `Network`, and `Algorithm`
- `N` open the network page
- `E` open the algorithm page
- `C` select CPU training or viewing
- `U` select GPU or CUDA training or viewing when CUDA is available
- `J` select the single learn or inspection trainer before starting
- `P` select the parallel speed or throughput trainer before starting
- `Q` finish the current training session early and jump to the results or replay screen
- `X` toggle Fast Mode for training sessions
- `A` toggle action arrows
- `D` toggle danger overlays
- `G` toggle the graph
- `T` toggle turbo mode
- `K` keep the window open after training
- `S` toggle score-series visibility
- `M` toggle moving-average visibility
- `B` toggle best-score visibility
- `1`, `2`, `3`, `4` speed presets
- `F` fit the graph view to the full history

Rendered training sessions now wait for an on-screen `Start` button, so you can choose the runtime device first.
Rendered training sessions also let you choose `Single [J]` or `Parallel [P]` before start.
`Fast Mode [X]` is the single-mode shortcut for hidden training plus post-run replay of the newest 3 runs.
After a fast-mode session completes, the finish overlay and `Overview` sidebar can replay the last 3 recorded runs without keeping older episode replays in memory.
`Q` is the quickest way to stop a flat run and still land in the same results screen with the latest 3 captured replays.
The lower dock now includes a `Stall threshold` input that controls when epsilon reheats during plateau recovery; fresh sessions default it to `150`.
The overview also shows lightweight throughput feedback during training, including episode steps, environment steps per second, and optimizer updates per second.
In parallel mode, bulk training uses multiple Snake environments and the `Network` / `Algorithm` pages switch to placeholders until the final rendered evaluation tail begins.
The score-series toggles now live in the lower dock near the loss chart, which keeps them visible on shorter windows.
Post-run tab switching now rebuilds fresh finished-state views instead of relying on one stale snapshot, so `Overview`, `Network`, `Algorithm`, and `Results` can all be opened safely after training ends.
The post-run `Results` tab now includes an episode slider so you can scrub across the run and inspect score and loss values at different points in training.

## Current implementation note

The configurable-architecture Deep RL version is in place, with CPU/CUDA device selection and an optional fast training mode.
Open `SESSION_HANDOFF.md` for a clear done / next / blocked record before continuing in another Codex session.
