# Snake AI Deep RL Session Handoff

## Goal

Build a self-contained Deep RL Snake training lab in this folder, with:

- device-selectable DQN training
- live pygame training visualizer
- local tabular comparison baseline
- teaching-oriented views for `Overview`, `Network`, and `Algorithm`
- clear docs so another Codex session can continue safely

## Done

- kept this folder independent from `4)Snake AI (with RL)`
- started with CPU-first training, then added optional CUDA support
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
- CPU/CUDA device selection through `auto`, `cpu`, and `cuda`

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
- Deep RL runtime can now run on:
  - `--device auto`
  - `--device cpu`
  - `--device cuda`
- requesting `cuda` now fails clearly if CUDA is unavailable

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
- fixed the `Training Snapshot` cards so metric values sit cleanly inside their boxes
- improved the headless or `No Render` behavior when the window is open:
  - the toggle now repaints immediately
  - the graph now refreshes after each completed run in headless mode
- added on-screen runtime controls:
  - `Start [Enter]` appears for rendered training sessions before training begins
  - `CPU [C]` and `GPU [U]` buttons allow switching the active runtime target from the dashboard
  - GPU selection is disabled in the UI when CUDA is unavailable
  - viewer mode exposes the CPU/GPU buttons without a start gate
  - enabling `No Render` before starting now bypasses the start gate and begins training automatically
- added score-axis labels to the training graph for easier reading
- made the pygame window fit the usable desktop height more safely so it does not drop below the taskbar on typical Windows layouts
- made the `Overview` page compress the right-side Q-value and graph cards a bit when the window is shorter
- reduced the requested training/viewer window height to `940` and added a larger desktop-height safety margin, because the earlier clamp was still too optimistic on some Windows setups
- tightened that again after a follow-up report:
  - requested window height is now `880`
  - desktop-height safety margin is now larger
  - the `Overview` sidebar uses a more compact metric/control layout so the options fit on shorter windows
- improved the network page further with:
  - a dominant-path summary
  - a color legend
  - stronger layer accenting
  - clearer top-node highlighting

## Validation completed

- ran `py_compile` successfully on:
  - `dqn_agent.py`
  - `dashboard.py`
  - `app_core.py`
  - `snake_game.py`
  - `train.py`
  - `visualizer.py`
- verified in code that:
  - rendered training waits for `Start` before stepping the environment
  - `No Render` still auto-starts
  - dashboard device selection can switch the agent between CPU and CUDA at runtime

## Blocked / not yet verified

- real runtime testing is still blocked in this sandbox by missing third-party packages
- the following have **not** been visually or behaviorally verified here:
  - `train.py --episodes 2 --no-render`
  - `train.py --episodes 2 --comparison-mode --no-render`
  - `train.py --episodes 2 --hidden-layers 64 --no-render`
  - `train.py --episodes 2 --hidden-layers 256,128,64 --no-render`
  - `train.py --episodes 2 --device auto`
  - `train.py --episodes 2 --device cuda`
  - `visualizer.py --checkpoint ... --metrics-log ...`
  - the new on-screen `Start`, `CPU`, and `GPU` buttons in a real pygame session
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
py -3 train.py --episodes 2 --device auto
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
- correct `Start [Enter]` behavior in rendered training
- correct `CPU [C]` / `GPU [U]` button states and runtime switching
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
