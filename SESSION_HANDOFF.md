# Snake AI Deep RL Session Handoff

## Goal

Build a self-contained Deep RL Snake training lab in this folder, with:

- CPU-only DQN training
- live pygame training visualizer
- local tabular comparison baseline
- teaching-oriented views for `Overview`, `Network`, and `Algorithm`
- clear docs so another Codex session can continue safely

## Done

- kept this folder independent from `4)Snake AI (with RL)`
- kept the first implementation target **CPU only**
- added project docs:
  - `README.md`
  - `HOW_DEEP_RL_WORKS.md`
  - `requirements.txt`
- added support modules:
  - `dqn_agent.py`
  - `metrics_utils.py`
  - `dashboard.py`
  - `app_core.py`
- kept `tabular_agent.py` as a local comparison baseline
- kept `train.py` and `visualizer.py` as thin entrypoints into the new runtime
- added `IMPLEMENTATION_SUMMARY.md` to the original tabular project folder

## Deep RL features implemented

- richer 18-value Deep RL state vector
- epsilon-greedy action selection
- replay buffer
- target network
- checkpoint save/load
- JSONL metric logging
- headless mode support
- local tabular baseline generation for comparison mode

## Architecture work implemented

- DQN hidden layers are now configurable from the CLI with:
  - `--hidden-layers 64`
  - `--hidden-layers 128,128`
  - `--hidden-layers 256,128,64`
- the DQN model builder now supports an arbitrary number of hidden layers
- checkpoints now persist `network_config.hidden_layers`
- resume logic now:
  - loads checkpoint architecture automatically
  - rejects explicit `--hidden-layers` values that do not match the checkpoint
- legacy checkpoints without architecture metadata fall back to `[128, 128]`

## UI / teaching improvements implemented

- renamed the views to:
  - `Overview`
  - `Network`
  - `Algorithm`
- `Tab` now cycles through all three views
- the `Overview` layout is clearer:
  - full-width KPI card at the top
  - decision + Q-value card on the right
  - grouped controls on the left
  - comparison graph below the decision card
  - top active features below the graph
- the `Network` view now uses:
  - one heatmap card per layer
  - dynamic support for 1..N hidden layers
  - compact connection-strength blocks between adjacent layers
  - summary cards for top inputs, hidden-layer highlights, and outputs
- the `Algorithm` view is now a short card-based explanation instead of a long dense text stack

## Validation completed

- ran `py_compile` successfully on:
  - `dqn_agent.py`
  - `dashboard.py`
  - `app_core.py`
  - `snake_game.py`
  - `train.py`
  - `visualizer.py`

## Blocked / not yet verified

- real runtime testing is still blocked in this sandbox by missing third-party packages
- the following have **not** been visually or behaviorally verified here:
  - `train.py --episodes 2 --no-render`
  - `train.py --episodes 2 --comparison-mode --no-render`
  - `train.py --episodes 2 --hidden-layers 64 --no-render`
  - `train.py --episodes 2 --hidden-layers 256,128,64 --no-render`
  - `visualizer.py --checkpoint ... --metrics-log ...`
  - actual pygame layout behavior with different window updates and longer runs

## Next recommended steps

1. Install dependencies locally:

```powershell
py -3 -m pip install -r requirements.txt
```

2. Run short headless smoke tests:

```powershell
py -3 train.py --episodes 2 --no-render
py -3 train.py --episodes 2 --comparison-mode --no-render
py -3 train.py --episodes 2 --hidden-layers 64 --no-render
py -3 train.py --episodes 2 --hidden-layers 256,128,64 --no-render
```

3. Run a resume mismatch check manually:

```powershell
py -3 train.py --episodes 2 --hidden-layers 128,128
py -3 train.py --episodes 2 --resume --hidden-layers 64
```

The second command should fail clearly if the checkpoint uses a different architecture.

4. Open the live UI and visually inspect:

```powershell
py -3 train.py --episodes 20 --comparison-mode --hidden-layers 128,128
```

Check for:

- no overlap in the `Overview` page
- readable legends in the graph
- readable layer heatmaps in the `Network` page
- correct `Tab` / `N` / `E` view switching
- clear teaching flow in the `Algorithm` page

## Good continuation targets for another Codex session

- do the real runtime smoke tests and fix any behavioral bugs found there
- polish the `Network` page further if a very deep architecture makes columns too narrow
- add screenshots or a visual verification checklist to `README.md`
- optionally add richer checkpoint browsing or per-layer statistics if you want a more research-lab feel

## Environment note from this sandbox

- Python is available here through:
  - `E:\Vibe Coding\Codex\.uv-python\cpython-3.11.15-windows-x86_64-none\python.exe`
- `py_compile` worked through that interpreter
- dependency installation and real runtime verification were not completed in this sandbox
