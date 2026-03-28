# How Deep RL Works Here

## The short version

The tabular Snake project stored values in a lookup table:

`Q(state, action)`

This project replaces that table with a **neural network**:

`Q-network(state) -> [Q_straight, Q_right, Q_left]`

That lets the agent learn patterns across similar states instead of only memorizing exact ones.

## What the Deep RL agent sees

The DQN in this project uses a richer feature vector than the tabular version.
It now includes 30 inputs:

- danger straight / right / left
- current direction
- food direction
- normalized food offset
- free space straight / right / left
- normalized snake length
- normalized distance to the food
- for each relative move (`straight`, `right`, `left`):
  - reachable area ratio after the move
  - whether the projected head can still reach the tail
  - how many legal exits remain after the move
  - the next-step Manhattan distance to the food

## The loop

Each step of training does this:

1. observe the current state, including projected move-safety features
2. choose an action
3. play that action in the game
4. get reward and next state
5. aggregate a short 3-step return and store that transition in hybrid replay memory
6. sample a mixed mini-batch from memory
7. choose the next action with the policy network, then evaluate that action with the target network
8. bootstrap from that point using the Double DQN target
9. softly move the target network toward the policy network after updates instead of doing only hard syncs

## Why hybrid replay matters

Consecutive Snake states are highly correlated.
Replay memory mixes old and new experiences together so training is more stable.
This project now mixes uniform replay with a smaller prioritized slice.
That keeps batches diverse while still revisiting high-TD-error transitions more often than routine ones.

## Why there is a target network

If the same network creates both the prediction and the target, learning can become unstable.
The target network is a slower-moving copy used to compute better training targets.

## Why Double DQN helps

Plain DQN can overestimate future value because the same target path both picks and scores the best next action.
This project now uses Double DQN:

- the policy network picks the best next action
- the target network scores that chosen action

That usually makes targets less optimistic and helps training plateau less often.

## Why 3-step returns help

Snake often needs a few safe moves before the payoff becomes obvious.
This project now uses fixed 3-step returns so food progress and trap avoidance can push value backward across several moves instead of only one.

## How evaluation and LR decay work

The trainer also runs periodic greedy evaluation passes.
Those passes update `latest_eval_avg` and `best_eval_avg`, and the dashboard can show both values read-only.

When evaluation stalls for long enough, the learning rate decays toward a minimum value instead of staying fixed forever.
At the end of a run, the best-eval snapshot is restored before the rendered evaluation tail and post-run results are shown.

## How device choice works here

This project still aims to be a clear learning tool, but it now supports both CPU and CUDA.

- `--device auto` uses CUDA when available, otherwise CPU
- `--device cpu` forces CPU training
- `--device cuda` forces GPU training and fails clearly if CUDA is unavailable

The training logic is the same either way.
The main difference is where the neural-network forward and backward passes run.

## Parallel CUDA schedule

Parallel bulk mode keeps the teaching UI responsive while the worker schedule gets more aggressive on CUDA:

- CUDA parallel mode uses a `64T -> 8 updates` cadence
- CPU parallel mode keeps a lighter `32T -> 2 updates` cadence

That schedule matches the larger GPU batch size and the short rendered evaluation tail at the end.
## Current Stability Additions

- The trainer now combines `Double DQN`, `3-step returns`, and `Hybrid replay`.
- Target-network updates use soft Polyak averaging with `tau = 0.005`.
- Parallel training performs periodic greedy evaluation every `200` completed training episodes across `5` evaluation runs.
- The best greedy-evaluation checkpoint is kept in memory and restored before final checkpoint export so the saved model reflects the best measured policy, not just the latest weights.
- Learning-rate decay is tied to stalled greedy evaluation windows rather than noisy training-score spikes.
