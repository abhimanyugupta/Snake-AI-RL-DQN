# How Deep RL Works Here

## The short version

The tabular Snake project stored values in a lookup table:

`Q(state, action)`

This project replaces that table with a **neural network**:

`Q-network(state) -> [Q_straight, Q_right, Q_left]`

That lets the agent learn patterns across similar states instead of only memorizing exact ones.

## What the Deep RL agent sees

The DQN in this project uses a richer feature vector than the tabular version.
It includes:

- danger straight / right / left
- current direction
- food direction
- normalized food offset
- free space straight / right / left
- normalized snake length
- normalized distance to the food

## The loop

Each step of training does this:

1. observe the current state
2. choose an action
3. play that action in the game
4. get reward and next state
5. store that transition in replay memory
6. sample a random mini-batch from memory
7. move predicted Q-values toward Bellman targets
8. occasionally copy the policy network into the target network

## Why replay memory matters

Consecutive Snake states are highly correlated.
Replay memory mixes old and new experiences together so training is more stable.

## Why there is a target network

If the same network creates both the prediction and the target, learning can become unstable.
The target network is a slower-moving copy used to compute better training targets.

## How device choice works here

This project still aims to be a clear learning tool, but it now supports both CPU and CUDA.

- `--device auto` uses CUDA when available, otherwise CPU
- `--device cpu` forces CPU training
- `--device cuda` forces GPU training and fails clearly if CUDA is unavailable

The training logic is the same either way.
The main difference is where the neural-network forward and backward passes run.
