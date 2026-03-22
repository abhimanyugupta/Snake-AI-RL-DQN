# Snake AI with Deep RL

This folder is a self-contained Deep RL follow-up to the tabular Snake AI project.
The current version supports **CPU or CUDA** training, while keeping the same teaching-oriented dashboard and DQN flow.
It now also supports a separate **parallel training mode** for higher-throughput headless bulk training with a short rendered evaluation tail.

## What this project shows

- the richer 18-feature state vector seen by the DQN
- live action selection and Q-values
- replay-memory driven training
- Bellman-target updates
- target-network syncing
- a local tabular baseline for comparison
- a live neural-network view across configurable hidden layers
- a separate parallel trainer that batches multiple headless Snake environments for speed

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
python train.py --episodes 100 --no-render --device auto
python train.py --episodes 100 --resume --hidden-layers 128,128 --device cuda
python train.py --episodes 300 --fast-mode --fast-tail-episodes 5 --device auto
python train.py --episodes 400 --trainer-mode parallel --parallel-envs 8 --eval-tail-episodes 3 --device auto
```

Notes:

- if you resume from a checkpoint, the checkpoint architecture is used automatically
- if you explicitly pass `--hidden-layers` while resuming, it must match the checkpoint
- older checkpoints without architecture metadata are treated as `128,128`
- `--device auto` uses CUDA when available, otherwise CPU
- `--device cuda` fails clearly if CUDA is unavailable on your machine
- `--fast-mode` strips most per-step visualization work and animates only the final tail episodes
- `--fast-tail-episodes` controls how many ending episodes stay fully animated in fast mode
- fast mode skips tabular-baseline generation and hides baseline series while it is active
- `--trainer-mode parallel` switches to a separate speed-first training path with batched headless workers
- `--parallel-envs` controls how many Snake environments the parallel trainer steps at once
- `--eval-tail-episodes` controls how many final rendered evaluation runs are shown and replayable in parallel mode
- resume restores trainer-mode metadata from checkpoints unless you explicitly override it on the CLI
- when a rendered fast-mode session finishes, the UI can replay the last 3 recorded fast episodes from both the board overlay and a `Recent Replays` card in `Overview`
- the `Recent Replays` card also explains why replay is unavailable when `Fast Mode`, `No Render`, or `Keep open` prevent capture
- stripped fast episodes now run uncapped, while the final animated tail keeps the full teaching dashboard
- parallel mode keeps bulk training headless for speed, updates the graph on completed episodes, and renders only the final evaluation tail runs
- in parallel mode, only the newest 3 evaluation-tail replays are kept in memory

## Visualizer

```powershell
python visualizer.py --checkpoint dqn_checkpoint.pt --metrics-log training_metrics.jsonl --device auto
```

## Views

- `Overview` - KPI cards, decision summary, controls, comparison graph, and top active features
- `Network` - layer-by-layer heatmaps plus compact connection-strength blocks
- `Algorithm` - short cards for the RL loop, current transition, Bellman update, and why replay/target networks help

## Controls

- `Enter` click or trigger `Start` to begin a rendered training session
- `Space` pause or resume
- `Tab` cycle through `Overview`, `Network`, and `Algorithm`
- `N` open the network page
- `E` open the algorithm page
- `C` select CPU training or viewing
- `U` select GPU or CUDA training or viewing when CUDA is available
- `J` select the single teaching trainer before starting
- `P` select the parallel throughput trainer before starting
- `X` toggle Fast Mode for training sessions
- `A` toggle action arrows
- `D` toggle danger overlays
- `G` toggle the graph
- `T` toggle turbo mode
- `K` keep the window open after training
- `H` toggle no-render mode from the UI
- `S` toggle score-series visibility
- `M` toggle moving-average visibility
- `B` toggle best-score visibility
- `1`, `2`, `3`, `4` speed presets
- `F` fit the graph view to the full history

Rendered training sessions now wait for an on-screen `Start` button, so you can choose the runtime device first.
Rendered training sessions also let you choose `Single [J]` or `Parallel [P]` before start.
If `No Render` is enabled, training auto-starts and keeps updating the graph in the open window after each episode.
`Fast Mode [X]` is available in the training UI and applies from the next episode boundary, not in the middle of the current run.
After a rendered fast-mode session completes, the finish overlay and `Overview` sidebar can replay the last 3 recorded runs without keeping older episode replays in memory.
The overview also shows lightweight throughput feedback during training, including episode steps, environment steps per second, and optimizer updates per second.
In parallel mode, bulk training uses multiple headless Snake environments and the `Network` / `Algorithm` pages switch to placeholders until the final rendered evaluation tail begins.

## Current implementation note

The configurable-architecture Deep RL version is in place, with CPU/CUDA device selection and an optional fast training mode.
Open `SESSION_HANDOFF.md` for a clear done / next / blocked record before continuing in another Codex session.
