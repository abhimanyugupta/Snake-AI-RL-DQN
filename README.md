# Snake AI with Deep RL

This folder is a self-contained Deep RL follow-up to the tabular Snake AI project.
The current version is intentionally **CPU only** so the DQN training loop stays easier to inspect and understand.

## What this project shows

- the richer 18-feature state vector seen by the DQN
- live action selection and Q-values
- replay-memory driven training
- Bellman-target updates
- target-network syncing
- a local tabular baseline for comparison
- a live neural-network view across configurable hidden layers

## Files

- `snake_game.py` - Snake environment and pygame visualizer
- `dqn_agent.py` - configurable CPU DQN, replay buffer, inspection payload, checkpointing
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
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
py -3 -m pip install -r requirements.txt
```

If `py` is not available on your machine, use your Python executable directly instead.

## Train

Default architecture:

```powershell
py -3 train.py --episodes 300 --comparison-mode
```

Custom hidden layers:

```powershell
py -3 train.py --episodes 150 --hidden-layers 64
py -3 train.py --episodes 150 --hidden-layers 128,128
py -3 train.py --episodes 150 --hidden-layers 256,128,64 --comparison-mode
```

Useful options:

```powershell
py -3 train.py --episodes 100 --resume
py -3 train.py --episodes 100 --no-render
py -3 train.py --episodes 100 --resume --hidden-layers 128,128
```

Notes:

- if you resume from a checkpoint, the checkpoint architecture is used automatically
- if you explicitly pass `--hidden-layers` while resuming, it must match the checkpoint
- older checkpoints without architecture metadata are treated as `128,128`

## Visualizer

```powershell
py -3 visualizer.py --checkpoint dqn_checkpoint.pt --metrics-log training_metrics.jsonl
```

## Views

- `Overview` - KPI cards, decision summary, controls, comparison graph, and top active features
- `Network` - layer-by-layer heatmaps plus compact connection-strength blocks
- `Algorithm` - short cards for the RL loop, current transition, Bellman update, and why replay/target networks help

## Controls

- `Space` pause or resume
- `Tab` cycle through `Overview`, `Network`, and `Algorithm`
- `N` open the network page
- `E` open the algorithm page
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

## Current implementation note

The configurable-architecture CPU Deep RL version is in place.
Open `SESSION_HANDOFF.md` for a clear done / next / blocked record before continuing in another Codex session.
