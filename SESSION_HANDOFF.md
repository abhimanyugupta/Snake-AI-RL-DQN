# Snake AI Deep RL Session Handoff

## Goal

Build a self-contained Deep RL Snake training lab in this folder, with:

- device-selectable DQN training
- live pygame training visualizer
- local tabular comparison baseline
- teaching-oriented views for `Overview`, `Network`, and `Algorithm`
- clear docs so another Codex session can continue safely

## Latest status

- standalone `No Render` has now been removed from both the GUI and CLI
- `Fast Mode` is now the only hidden-training path for `Single` mode
- `Single` is the learn or inspection path
- `Parallel` is the speed or throughput path
- fresh CUDA sessions now default to `GPU + Parallel`
- fresh CPU-only sessions still default to `CPU + Single`
- training now supports `Q` as an early-finish hotkey that still lands in results and replays
- the stall threshold now defaults to `150` and is intended to be editable from the lower dock
- repeated spin-loop behavior now gets a small soft penalty instead of only timing out
- the post-run `Results` tab now has an interactive episode slider or scrubber for inspecting values across the full run
- the finished-window loop has extra defensive rebuilding so post-run tab clicks do not immediately tear down the window on view refresh errors
- non-terminal moves now get a small Manhattan-distance shaping term so the reward is denser when the snake moves toward food
- the trainer now uses Double DQN target selection instead of the older plain DQN target
- the trainer now uses hybrid replay instead of pure prioritized replay
- older bullets below that mention `No Render` describe superseded behavior and should not be treated as current

## Current smoke-test commands

Use these current commands instead of the older `--no-render` examples lower in this file:

- `python train.py --episodes 20`
- `python train.py --episodes 20 --fast-mode`
- `python train.py --episodes 20 --trainer-mode parallel --parallel-envs 8 --eval-tail-episodes 3 --device auto`

## Latest verification

- `py_compile` passes for the Deep RL runtime files
- dry-run post-run rendering now succeeds even when `last_transition` is `None`:
  - `Results`
  - `Algorithm`
  - `Network`
  - `Overview`
- verified the obsolete floating `control_panel` is no longer present in dashboard payloads
- verified the hybrid plateau recovery path can trigger epsilon reheats after prolonged stagnation
- ran a short real runtime smoke through `app_core.train_session(...)` after the new changes:
  - `render=False + trainer_mode=\"single\" + hidden_layers=[64] + episodes=2` completed successfully
- verified with direct runtime smoke tests through `app_core.train_session(...)`:
  - `render=False + trainer_mode=\"single\" + fast_mode=True` completed successfully
  - `render=False + trainer_mode=\"parallel\" + parallel_envs=4` completed successfully
- verified helper defaults:
  - `default_trainer_mode_for_device(\"cuda\") -> \"parallel\"`
  - `default_trainer_mode_for_device(\"cpu\") -> \"single\"`
- verified CLI help no longer shows `--no-render`
- verified the new loop-penalty helper smoke: repeated identical movement signatures trigger the extra penalty
- verified the new stall-threshold input default resolves to `150`
- verified checkpoint save/load now persists the stall threshold in `trainer_config`
- verified a dummy-display post-run smoke:
  - `Results`, `Overview`, `Network`, `Algorithm`, then back to `Results` all redraw cleanly
  - the new results slider updates the selected episode index
- verified in-sandbox that `py_compile` still passes after the Manhattan-shaping patch
- verified by direct code inspection that both snake environments now:
  - keep death and food rewards unchanged
  - add Manhattan shaping only on ordinary non-terminal moves
  - still apply the soft anti-loop penalty on top of those ordinary moves
- verified in-sandbox that the Double DQN trainer patch still compiles cleanly
- verified by direct code inspection that training now:
  - selects next actions with the policy net
  - evaluates those chosen actions with the target net
  - leaves the rest of the training loop unchanged
- verified in-sandbox that the hybrid replay patch compiles cleanly
- verified by direct code inspection that replay now:
  - stores per-transition priorities
  - mixes `70%` uniform samples with `30%` prioritized samples
  - updates priorities from absolute TD error
  - falls back safely when older checkpoints do not contain priority data

## Latest UI fixes

- removed the obsolete floating `Controls` card draw path that was overlapping the snake board
- post-run views now rebuild from fresh finished-state data instead of a stale frozen snapshot
  - this hardens tab switching across `Overview`, `Network`, `Algorithm`, and `Results`
  - missing transition data no longer crashes the finished-state algorithm page
- added hybrid plateau recovery to the DQN:
  - baseline epsilon still decays episode by episode
  - fresh sessions now default the stall threshold to `150`, and the lower dock can change it live
  - when training stalls for the current threshold, epsilon reheats to at least `0.12`
  - reheats decay back down with the existing decay factor
  - reheats respect a `150`-episode cooldown
  - reheat state is checkpointed and restored on resume
  - the snapshot and results views now show exploration mode, stall progress, reheats, and cooldown
- single-mode sessions can now be finished early with `Q`:
  - the current partial run is discarded
  - the session still lands in post-run results
  - the newest 3 completed replays remain available
- added a soft anti-spin penalty in both snake environments:
  - repeated `(head, direction, food)` signatures inside a short window subtract a small extra reward
  - food pickups reset the loop detector
- added Manhattan-distance reward shaping in both snake environments:
  - `FOOD_PROGRESS_SHAPING_SCALE = 0.05`
  - ordinary moves now add `0.05 * (previous_distance - current_distance)`
  - food and death steps still use their original rewards only
- upgraded the trainer from plain DQN targets to Double DQN targets:
  - the policy net now chooses the next action
  - the target net evaluates that chosen next action
  - the teaching text now describes Double DQN instead of plain Bellman max-over-target updates
- upgraded replay from pure prioritized sampling to hybrid replay:
  - batches now mix `70%` uniform and `30%` prioritized samples
  - new transitions enter at `max(median_priority, 0.5 * max_priority)` instead of raw max priority
  - only the prioritized portion carries importance-weighted emphasis into the Huber loss
  - replay capacity now defaults to `200000`
  - parallel mode now uses `warmup_size = 10000` and `batch_size = 512`
  - old checkpoints without replay priorities still load with safe default priorities
- the lower dock now carries a `Stall threshold` entry box without shifting the main sidebar layout
- parallel bulk mode no longer pretends the left board is one live run
  - the board now switches to an aggregate `Parallel Bulk Training` status panel
  - the snapshot card now uses `Completed runs` instead of a misleading live-run counter
  - the right-side decision card becomes `Bulk Throughput` instead of Q-value bars
- moved the `Scores / Avg / Best` toggles out of the clipped sidebar area
  - they now render in the lower dock above the loss chart
  - the old sidebar `History` section was removed
- tightened lower-dock row spacing so the trainer-mode buttons fit inside the dock instead of clipping
- render-path smoke tests passed for:
  - one synthetic parallel-bulk frame
  - one synthetic single-mode frame

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
- hidden fast-mode training in single mode, with replay afterward
- local tabular baseline generation for comparison mode
- CPU/CUDA device selection through `auto`, `cpu`, and `cuda`
- optional fast training mode with hidden single-snake training and replay of the newest 3 runs afterward
- optional parallel training mode with batched headless workers and a short rendered evaluation tail

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
- added fast-mode controls and runtime behavior:
  - CLI support for `--fast-mode` and `--fast-tail-episodes`
  - a new `Fast Mode [X]` toggle in the training dashboard
  - fast mode changes apply at the next episode boundary
  - fast mode skips tabular-baseline generation while active
  - fast mode strips most per-step dashboard work for early episodes
  - the final tail episodes stay fully animated
  - after a rendered fast-mode session completes, both the overlay and the `Overview` sidebar can replay the last 3 recorded fast runs
  - replay history is kept in memory only and capped to the latest 3 episodes
  - the `Overview` sidebar now explains exactly why replay is unavailable when:
    - fast mode is off
    - `No Render` is on
    - `Keep open` is off
  - the final training screen now forces `Overview` mode so the replay card is visible immediately
  - stripped fast episodes now avoid the full per-step action-detail path and run without FPS pacing
  - throughput feedback is now surfaced in the dashboard and console:
    - episode steps
    - env steps/sec
    - updates/sec
  - fast-mode diagnostics now avoid forcing detailed GPU-to-CPU scalar sync on every single training step
  - snake collision checks now use a body-occupancy set instead of repeated list scans
  - the free-space state features now raycast against walls plus the occupancy set
- added a true headless `SnakeLogicEnv` so the speed-focused trainer can step non-rendered environments without going through the pygame wrapper
- refactored the replay buffer to use contiguous NumPy storage and vectorized batched inserts
- added parallel-mode controls and runtime behavior:
  - CLI support for `--trainer-mode single|parallel`
  - CLI support for `--parallel-envs`
  - CLI support for `--eval-tail-episodes`
  - pre-start UI buttons for `Single [J]` and `Parallel [P]`
  - checkpoint metadata for trainer mode, parallel env count, and eval-tail size
  - resume now restores saved trainer-mode metadata unless the CLI explicitly overrides it
  - parallel bulk mode batches multiple headless Snake environments into shared policy updates
  - parallel mode defaults to larger training schedules:
    - CUDA batch size `1024`
    - CPU batch size `512`
    - `update_every_transitions = 32`
    - `gradient_steps_per_update = 2`
    - default parallel env count `16` on CUDA, `8` on CPU when not explicitly provided
  - parallel bulk mode updates the dashboard on a summary cadence instead of every worker step
  - the `Overview` page now switches to throughput-oriented messaging in parallel bulk mode
  - the `Network` and `Algorithm` pages show placeholders during parallel bulk mode and return to teaching detail during the evaluation tail
  - parallel mode does not precompute or display the tabular baseline during bulk training
  - only the final rendered evaluation-tail runs are kept as replayable last-3 replays
  - parallel-mode metrics are buffered and flushed every 8 completed episodes
  - parallel mode checkpoints on interval and final exit, rather than every new best score
  - the parallel trainer now reuses preallocated state buffers instead of rebuilding NumPy state batches every step
  - DQN state encoding now supports filling preallocated output arrays to reduce allocation churn
- made network inspection lazy:
  - `agent.inspect_network(...)` is no longer built on every dashboard refresh
  - the heavy network payload is now built only when the `Network` tab is active
  - stripped fast episodes show an explanatory placeholder instead of live heatmaps
- confirmed the antigravity fixes are present in the current code:
  - dashboard mouse input now uses scaled pygame events in the trainer/viewer loops
  - the `Episode goal` field and neighboring controls have extra vertical spacing to avoid overlap
- cleaned up `app_core.py` so training and viewer entrypoints are consistent again:
  - CPU/GPU device changes are wired through the runtime
  - rendered training uses the `Start` gate again
  - viewer and trainer now both pass device state into the dashboard cleanly
- rebalanced the `Overview` layout to use the empty lower-left area under the board:
  - moved run-setup controls into a new bottom dock under the board
  - added a GUI `Parallel envs` input box there
  - added a dedicated `Loss Trend` graph there that tracks per-episode loss and moving average
  - kept the main right-side graph focused on score/comparison history
- added post-run results and replay behavior:
  - successful runs now enter a post-run results state instead of simply exiting
  - `No Render` sessions can still capture the final 3 replayable runs and open a post-run results window afterward
  - replay storage now uses compact logical frame snapshots instead of only relying on render-time dashboard captures
  - replay playback rebuilds the dashboard frame from stored logical state, action info, and context
  - the newest 3 captured runs remain in-memory only and are still not checkpointed
  - the finished-window loop now rebuilds the dashboard payload after tab changes, so post-run switching between `Overview`, `Network`, `Algorithm`, and `Results` no longer uses stale finished-state data
- added a post-training `Results [R]` tab:
  - it appears only after training finishes
  - it shows full-run score, moving-average, best-score, loss, and loss-average history
  - it now includes an interactive episode slider so you can scrub across the run and inspect values on both results graphs
  - the finished flow auto-focuses `Results`
  - the full results view is exported as `training_summary.png` in this project folder on each completed run
- trainer-mode switching is now queue-based during active training:
  - `Single [J]` / `Parallel [P]` no longer look silently broken after start
  - clicking the inactive mode during training now queues a switch
  - the queue applies at safe boundaries:
    - next episode boundary in single mode
    - after the current bulk iteration in parallel bulk mode
    - after the current evaluation episode in parallel eval mode
  - the dashboard labels and help text now show the queued trainer mode clearly
  - queued trainer-mode buttons now use a separate amber-style treatment so they are visually distinct from the currently live mode
- improved the pre-start lobby messaging:
  - the `Mode` metric now reflects whether the run is preparing single training, fast mode, or parallel mode
  - the start overlay subtitle now explains what fast mode or parallel mode will do before training begins
- generalized the shared graph renderer:
  - the score graph keeps its existing scale behavior
  - the loss graph now scales to the visible low loss values instead of pinning to `1.0`
  - the full-run results plots reuse the same graph renderer with full-history ranges
  - full-history results plots now use the real completed-episode range instead of forcing a fake 10-episode window on very short runs
  - the results summary card height now grows with the summary text so longer session summaries do not clip
- improved parallel bulk responsiveness:
  - UI events are now sampled inside long parallel-env stepping loops instead of only once per outer iteration
  - user interaction now forces a near-term redraw in parallel mode, so toggles like `Turbo` and trainer/device changes feel less frozen when `parallel_envs` is large
- updated the stronger default training settings for this project:
  - default hidden layers are now `256,256,128`
  - default `--episodes` is now `5000`
  - default parallel env count is now `64` on CUDA and `16` on CPU when not explicitly provided

## Validation completed

- ran `py_compile` successfully on:
  - `dqn_agent.py`
  - `dashboard.py`
  - `app_core.py`
  - `snake_game.py`
  - `train.py`
  - `visualizer.py`
  - `metrics_utils.py`
- verified in code that:
  - rendered training waits for `Start` before stepping the environment
  - `No Render` still auto-starts
  - dashboard device selection can switch the agent between CPU and CUDA at runtime
  - fast mode can skip heavy per-step rendering work while keeping episode history updates
  - fast mode keeps the final tail episodes fully animated when rendering is on
  - the fast-mode finish overlay can replay the latest 3 recorded fast runs
  - the sidebar replay card is populated from the same latest-3 in-memory replay list
  - stripped fast episodes no longer request pacing through `play_step(...)`
  - the DQN fast path can choose actions without building full Q-value payloads every step
  - the new parallel-mode CLI, dashboard state, and training code compile cleanly with the rest of the project
  - the new post-run/results flow compiles cleanly with:
    - no-render replay capture
    - queued single/parallel mode switching
    - full-history results plots
    - `training_summary.png` export
- ran limited runtime smoke checks on this machine:
  - `python train.py --episodes 2 --no-render`
    - training completed far enough to write:
      - `dqn_checkpoint.pt`
      - `training_metrics.jsonl`
      - `training_summary.png`
    - the process then stayed alive until timeout, which matches the intended new post-run results-window behavior
  - `python train.py --episodes 5 --trainer-mode parallel --parallel-envs 4 --eval-tail-episodes 2 --no-render`
    - training completed far enough to update:
      - `dqn_checkpoint.pt`
      - `training_metrics.jsonl`
      - `training_summary.png`
    - the process then stayed alive until timeout, again consistent with the intended post-run results hold state
- captured real window images during local verification:
  - the pre-start rendered dashboard appears and lays out correctly
  - a direct `PrintWindow` capture confirms the live window can be inspected programmatically
  - fully automated start-button interaction is still unreliable in this environment, so real manual in-window verification is still needed for interactive controls

## Blocked / not yet verified

- real runtime testing is still blocked in this sandbox by missing third-party packages
- the following have **not** been visually or behaviorally verified here:
  - `train.py --episodes 2 --comparison-mode --no-render`
  - `train.py --episodes 2 --hidden-layers 64 --no-render`
  - `train.py --episodes 2 --hidden-layers 256,128,64 --no-render`
  - `train.py --episodes 2 --device auto`
  - `train.py --episodes 2 --device cuda`
  - `train.py --episodes 8 --fast-mode --fast-tail-episodes 5`
  - `train.py --episodes 8 --fast-mode --no-render`
  - `train.py --episodes 20 --trainer-mode parallel --parallel-envs 8 --device auto`
  - `train.py --episodes 20 --trainer-mode parallel --parallel-envs 8 --eval-tail-episodes 3`
  - `visualizer.py --checkpoint ... --metrics-log ...`
  - the new on-screen `Start`, `CPU`, and `GPU` buttons in a real pygame session
  - the new `Single [J]` / `Parallel [P]` trainer-mode buttons in a real pygame session
  - the new `Fast Mode [X]` toggle and final-tail animation behavior in a real pygame session
  - the new fast-mode replay buttons after training completes
  - the new `Recent Replays` sidebar card after training completes
  - actual parallel-mode throughput gains and GPU-usage improvements on real hardware
  - the rendered evaluation-tail replay flow after a parallel-mode session
  - resume behavior when the saved checkpoint trainer mode is `parallel`
  - the new bottom-dock `Parallel envs` input in a real pygame session
  - the new bottom-dock `Loss Trend` graph in a real pygame session
  - the new post-run `Results [R]` tab in a real pygame session
  - automatic post-run results-window opening after a fully headless `--no-render` session without relying on timeout-based inference
  - replay buttons and `1/2/3` hotkeys from the new post-run results state
  - queued trainer-mode switching during a real active run
  - full-history results rendering and the saved `training_summary.png` image on a real completed run
  - the replay-unavailable reason text in the sidebar under:
    - default mode
    - `No Render`
    - `Keep open` off
  - actual pygame layout behavior with different window updates and longer runs

## Next recommended steps

1. Install dependencies locally:

```powershell
python -m pip install -r requirements.txt
```

Interpreter note:

- on this machine, `python` and `py -3.12` point to Python 3.12 with the project dependencies
- `py -3` points to Python 3.14, which currently does not have `numpy` installed
- use `python ...` or `py -3.12 ...` for this project unless the 3.14 environment is deliberately set up too

2. Run short headless smoke tests:

```powershell
python train.py --episodes 2 --no-render
python train.py --episodes 2 --comparison-mode --no-render
python train.py --episodes 2 --hidden-layers 64 --no-render
python train.py --episodes 2 --hidden-layers 256,128,64 --no-render
python train.py --episodes 2 --device auto
python train.py --episodes 8 --fast-mode --fast-tail-episodes 5
python train.py --episodes 8 --fast-mode --no-render
python train.py --episodes 20 --trainer-mode parallel --parallel-envs 8 --no-render
python train.py --episodes 20 --trainer-mode parallel --parallel-envs 8 --device auto
python train.py --episodes 20 --trainer-mode parallel --parallel-envs 8 --eval-tail-episodes 3
```

3. Run a resume mismatch check manually:

```powershell
python train.py --episodes 2 --hidden-layers 128,128
python train.py --episodes 2 --resume --hidden-layers 64
```

The second command should fail clearly if the checkpoint uses a different architecture.

4. Open the live UI and visually inspect:

```powershell
python train.py --episodes 20 --comparison-mode --hidden-layers 128,128
```

Check for:

- no overlap in the `Overview` page
- readable legends in the graph
- readable layer heatmaps in the `Network` page
- correct `Tab` / `N` / `E` view switching
- correct `Start [Enter]` behavior in rendered training
- correct `CPU [C]` / `GPU [U]` button states and runtime switching
- correct `Single [J]` / `Parallel [P]` pre-start selection behavior
- correct `Fast Mode [X]` toggle state and next-episode application behavior
- no tabular baseline lines visible while fast mode is active
- only the final tail episodes animate in fast mode
- the finish overlay shows replay buttons for the latest 3 fast-mode runs
- replay buttons actually play back the recorded last-3 runs on screen
- parallel-mode bulk training keeps the UI responsive while batching headless workers
- the final evaluation tail renders after parallel bulk training and the latest 3 runs become replayable
- clear teaching flow in the `Algorithm` page

## Good continuation targets for another Codex session

- do the real runtime smoke tests and fix any behavioral bugs found there
- profile the remaining training cost after fast mode to see whether replay-buffer sampling or tensor-transfer overhead is now the next bottleneck
- polish the `Network` page further if a very deep architecture makes columns too narrow
- add screenshots or a visual verification checklist to `README.md`
- optionally add richer checkpoint browsing or per-layer statistics if you want a more research-lab feel

## Environment note from this sandbox

- Python is available here through:
  - `E:\Vibe Coding\Codex\.uv-python\cpython-3.11.15-windows-x86_64-none\python.exe`
- `py_compile` worked through that interpreter
- dependency installation and real runtime verification were not completed in this sandbox
