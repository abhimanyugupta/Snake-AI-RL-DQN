from __future__ import annotations

import random
from typing import Iterable, List, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency 'numpy'. Install requirements.txt before running the Deep RL project."
    ) from exc

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency 'torch'. Install requirements.txt before running the Deep RL project."
    ) from exc

from snake_game import Direction, Point


DEFAULT_HIDDEN_LAYERS = [256, 256, 128]
LEGACY_HIDDEN_LAYERS = list(DEFAULT_HIDDEN_LAYERS)
SUPPORTED_DEVICE_CHOICES = ("auto", "cpu", "cuda")


def normalize_hidden_layers(hidden_layers: Iterable[int] | None) -> List[int]:
    if hidden_layers is None:
        return list(DEFAULT_HIDDEN_LAYERS)

    normalized = [int(value) for value in hidden_layers]
    if not normalized:
        raise ValueError("At least one hidden layer must be provided.")
    if any(value <= 0 for value in normalized):
        raise ValueError("Hidden-layer sizes must be positive integers.")
    return normalized


def normalize_device_preference(device_preference: str | None) -> str:
    normalized = (device_preference or "auto").strip().lower()
    if normalized not in SUPPORTED_DEVICE_CHOICES:
        raise ValueError(
            "Device must be one of: auto, cpu, cuda."
        )
    return normalized


def resolve_torch_device(device_preference: str | None) -> torch.device:
    normalized = normalize_device_preference(device_preference)
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA was requested, but torch.cuda.is_available() is False on this machine."
        )
    return torch.device(normalized)


def cuda_is_available() -> bool:
    return bool(torch.cuda.is_available())


def extract_hidden_layers_from_checkpoint(checkpoint: dict) -> List[int]:
    network_config = checkpoint.get("network_config", {})
    hidden_layers = network_config.get("hidden_layers")
    if hidden_layers is None:
        return list(LEGACY_HIDDEN_LAYERS)
    return normalize_hidden_layers(hidden_layers)


def load_checkpoint_network_config(path: str) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    hidden_layers = extract_hidden_layers_from_checkpoint(checkpoint)
    network_config = dict(checkpoint.get("network_config", {}))
    network_config["hidden_layers"] = hidden_layers
    network_config.setdefault("input_size", len(DQNAgent.STATE_LABELS))
    network_config.setdefault("output_size", len(DQNAgent.ACTION_LABELS))
    return network_config


class ReplayBuffer:
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self.position = 0
        self.size = 0
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.size)

    def add(
        self,
        state: np.ndarray,
        action_index: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.states[index] = np.asarray(state, dtype=np.float32)
        self.actions[index] = int(action_index)
        self.rewards[index] = float(reward)
        self.next_states[index] = np.asarray(next_state, dtype=np.float32)
        self.dones[index] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        states = np.asarray(states, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)
        batch_size = int(states.shape[0])
        if batch_size <= 0:
            return
        if batch_size >= self.capacity:
            states = states[-self.capacity :]
            actions = actions[-self.capacity :]
            rewards = rewards[-self.capacity :]
            next_states = next_states[-self.capacity :]
            dones = dones[-self.capacity :]
            batch_size = self.capacity

        end = self.position + batch_size
        if end <= self.capacity:
            indices = slice(self.position, end)
            self.states[indices] = states
            self.actions[indices] = actions
            self.rewards[indices] = rewards
            self.next_states[indices] = next_states
            self.dones[indices] = dones
        else:
            first_count = self.capacity - self.position
            second_count = batch_size - first_count
            self.states[self.position :] = states[:first_count]
            self.actions[self.position :] = actions[:first_count]
            self.rewards[self.position :] = rewards[:first_count]
            self.next_states[self.position :] = next_states[:first_count]
            self.dones[self.position :] = dones[:first_count]

            self.states[:second_count] = states[first_count:]
            self.actions[:second_count] = actions[first_count:]
            self.rewards[:second_count] = rewards[first_count:]
            self.next_states[:second_count] = next_states[first_count:]
            self.dones[:second_count] = dones[first_count:]

        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=int(batch_size))
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def sample_tensors(self, batch_size: int, device: torch.device, pin_memory: bool = False):
        states, actions, rewards, next_states, dones = self.sample(batch_size)
        non_blocking = bool(pin_memory and device.type == "cuda")

        def to_device(array, dtype=None):
            tensor = torch.from_numpy(np.ascontiguousarray(array))
            if dtype is not None:
                tensor = tensor.to(dtype=dtype)
            if non_blocking:
                tensor = tensor.pin_memory()
            return tensor.to(device, non_blocking=non_blocking)

        return (
            to_device(states, dtype=torch.float32),
            to_device(actions, dtype=torch.long),
            to_device(rewards, dtype=torch.float32),
            to_device(next_states, dtype=torch.float32),
            to_device(dones, dtype=torch.float32),
        )

    def state_dict(self) -> dict:
        return {
            "capacity": int(self.capacity),
            "state_dim": int(self.state_dim),
            "size": int(self.size),
            "position": int(self.position),
            "states": self.states[: self.size].copy(),
            "actions": self.actions[: self.size].copy(),
            "rewards": self.rewards[: self.size].copy(),
            "next_states": self.next_states[: self.size].copy(),
            "dones": self.dones[: self.size].copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        capacity = int(state.get("capacity", self.capacity))
        state_dim = int(state.get("state_dim", self.state_dim))
        self.capacity = capacity
        self.state_dim = state_dim
        self.position = int(state.get("position", 0))
        self.size = int(state.get("size", 0))
        self.states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)

        if "states" in state:
            loaded_size = min(self.capacity, int(state.get("size", len(state["states"]))))
            self.size = loaded_size
            self.position = int(state.get("position", loaded_size % self.capacity))
            self.states[:loaded_size] = np.asarray(state["states"], dtype=np.float32)[:loaded_size]
            self.actions[:loaded_size] = np.asarray(state["actions"], dtype=np.int64)[:loaded_size]
            self.rewards[:loaded_size] = np.asarray(state["rewards"], dtype=np.float32)[:loaded_size]
            self.next_states[:loaded_size] = np.asarray(state["next_states"], dtype=np.float32)[:loaded_size]
            self.dones[:loaded_size] = np.asarray(state["dones"], dtype=np.float32)[:loaded_size]
            return

        raw_memory = state.get("memory", [])
        self.position = 0
        self.size = 0
        for item in raw_memory[: self.capacity]:
            self.add(*item)


class DQNNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_layers: Sequence[int], output_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_layers = list(normalize_hidden_layers(hidden_layers))
        self.output_size = output_size

        layer_sizes = [input_size] + self.hidden_layers + [output_size]
        self.linears = nn.ModuleList(
            nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            for index in range(len(layer_sizes) - 1)
        )

    def forward(self, x):  # pragma: no cover - thin wrapper
        current = x
        for linear in self.linears[:-1]:
            current = torch.relu(linear(current))
        return self.linears[-1](current)

    def forward_with_activations(self, x):
        current = x
        hidden_pre = []
        hidden_post = []

        for linear in self.linears[:-1]:
            pre_activation = linear(current)
            post_activation = torch.relu(pre_activation)
            hidden_pre.append(pre_activation)
            hidden_post.append(post_activation)
            current = post_activation

        output = self.linears[-1](current)
        return {
            "hidden_pre": hidden_pre,
            "hidden_post": hidden_post,
            "output": output,
        }


class DQNAgent:
    ACTION_KEYS = ["straight", "right", "left"]
    ACTION_LABELS = ["Straight", "Right turn", "Left turn"]
    ACTION_SHORT_LABELS = ["Straight", "Right", "Left"]
    STATE_LABELS = [
        ("danger_straight", "Danger straight", "Danger S"),
        ("danger_right", "Danger right", "Danger R"),
        ("danger_left", "Danger left", "Danger L"),
        ("moving_left", "Moving left", "Move L"),
        ("moving_right", "Moving right", "Move R"),
        ("moving_up", "Moving up", "Move U"),
        ("moving_down", "Moving down", "Move D"),
        ("food_left", "Food left", "Food L"),
        ("food_right", "Food right", "Food R"),
        ("food_up", "Food up", "Food U"),
        ("food_down", "Food down", "Food D"),
        ("food_dx", "Food dx", "Food dx"),
        ("food_dy", "Food dy", "Food dy"),
        ("space_straight", "Free space straight", "Space S"),
        ("space_right", "Free space right", "Space R"),
        ("space_left", "Free space left", "Space L"),
        ("snake_length", "Snake length", "Length"),
        ("food_distance", "Food distance", "Distance"),
    ]

    def __init__(
        self,
        learning_rate: float = 1e-3,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 25_000,
        batch_size: int = 256,
        warmup_size: int = 1_000,
        target_sync_interval: int = 250,
        update_every_transitions: int = 1,
        gradient_steps_per_update: int = 1,
        hidden_layers: Iterable[int] | None = None,
        device_preference: str = "auto",
    ):
        self.device_preference = normalize_device_preference(device_preference)
        self.device = resolve_torch_device(self.device_preference)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.base_epsilon = float(epsilon)
        self.reheat_epsilon = 0.12
        self.reheat_patience = 250
        self.reheat_cooldown = 150
        self.reheat_avg_margin = 0.5
        self.reheat_active_epsilon: float | None = None
        self.reheat_count = 0
        self.plateau_counter = 0
        self.cooldown_remaining = 0
        self.best_episode_score = float("-inf")
        self.best_moving_avg = float("-inf")
        self.exploration_mode = "decay"
        self.batch_size = batch_size
        self.warmup_size = warmup_size
        self.target_sync_interval = target_sync_interval
        self.update_every_transitions = max(1, int(update_every_transitions))
        self.gradient_steps_per_update = max(1, int(gradient_steps_per_update))
        self.hidden_layers = normalize_hidden_layers(hidden_layers)
        self.n_games = 0
        self.train_steps = 0
        self.pending_transitions = 0
        self.pin_memory_transfers = self.device.type == "cuda"
        self.last_train_info = {
            "status": "warmup",
            "loss": None,
            "td_error": None,
            "predicted_q": None,
            "target_q": None,
            "buffer_size": 0,
            "warmup_remaining": warmup_size,
            "synced_target": False,
            "target_sync_remaining": target_sync_interval,
            "grad_norm": None,
            "did_update": False,
            "update_count": 0,
            "batch_size": batch_size,
            "trainer_mode": "single",
        }

        self.policy_net = DQNNetwork(
            input_size=len(self.STATE_LABELS),
            hidden_layers=self.hidden_layers,
            output_size=len(self.ACTION_LABELS),
        ).to(self.device)
        self.target_net = DQNNetwork(
            input_size=len(self.STATE_LABELS),
            hidden_layers=self.hidden_layers,
            output_size=len(self.ACTION_LABELS),
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(replay_capacity, len(self.STATE_LABELS))
        self._refresh_exploration_state()

    @property
    def device_label(self) -> str:
        return "CUDA" if self.device.type == "cuda" else "CPU"

    @property
    def architecture_label(self) -> str:
        layer_sizes = [len(self.STATE_LABELS)] + list(self.hidden_layers) + [
            len(self.ACTION_LABELS)
        ]
        return " -> ".join(str(size) for size in layer_sizes)

    def get_action(self, state: np.ndarray) -> List[int]:
        return self.get_action_selection(state)["action"]

    def configure_training_schedule(
        self,
        *,
        batch_size: int | None = None,
        warmup_size: int | None = None,
        target_sync_interval: int | None = None,
        update_every_transitions: int | None = None,
        gradient_steps_per_update: int | None = None,
        trainer_mode: str | None = None,
    ) -> None:
        if batch_size is not None:
            self.batch_size = int(batch_size)
        if warmup_size is not None:
            self.warmup_size = int(warmup_size)
        if target_sync_interval is not None:
            self.target_sync_interval = int(target_sync_interval)
        if update_every_transitions is not None:
            self.update_every_transitions = max(1, int(update_every_transitions))
        if gradient_steps_per_update is not None:
            self.gradient_steps_per_update = max(1, int(gradient_steps_per_update))
        if trainer_mode is not None:
            self.last_train_info["trainer_mode"] = str(trainer_mode)
        self.last_train_info["batch_size"] = int(self.batch_size)

    def set_device(self, device_preference: str) -> bool:
        normalized = normalize_device_preference(device_preference)
        new_device = resolve_torch_device(normalized)
        changed = new_device.type != self.device.type

        self.device_preference = normalized
        if not changed:
            return False

        self.device = new_device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.loss_fn.to(self.device)
        self._move_optimizer_state_to_device()
        return True

    def get_action_details(self, state: np.ndarray, greedy: bool = False) -> dict:
        return self.get_action_selection(state, greedy=greedy, lightweight=False)

    def get_action_selection(
        self, state: np.ndarray, greedy: bool = False, lightweight: bool = False
    ) -> dict:
        if not greedy and random.random() < self.epsilon:
            action_index = random.randint(0, len(self.ACTION_LABELS) - 1)
            decision_type = "explore"
            q_values = [] if lightweight else self.get_q_values(state)
        else:
            q_tensor = self._get_q_tensor(state)
            action_index = self._select_best_action_index_from_q_tensor(q_tensor)
            decision_type = "policy preview" if greedy else "exploit"
            q_values = None if lightweight else self._q_tensor_to_list(q_tensor)

        if lightweight:
            return self._build_action_summary(action_index, decision_type)
        return self._build_action_details(action_index, q_values or [], decision_type)

    def get_action_indices_batch(self, states: np.ndarray, greedy: bool = False) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2:
            raise ValueError("states batch must be 2D")

        q_tensor = self._get_q_tensor_batch(states)
        greedy_actions = torch.argmax(q_tensor, dim=1)
        chosen = greedy_actions.detach().cpu().numpy().astype(np.int64)
        if greedy:
            return chosen

        explore_mask = np.random.random(size=states.shape[0]) < self.epsilon
        if np.any(explore_mask):
            chosen[explore_mask] = np.random.randint(
                0,
                len(self.ACTION_LABELS),
                size=int(np.sum(explore_mask)),
                dtype=np.int64,
            )
        return chosen

    def get_q_values(self, state: np.ndarray) -> List[float]:
        return self._q_tensor_to_list(self._get_q_tensor(state))

    def remember(
        self,
        state: np.ndarray,
        action_index: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.add(state, action_index, reward, next_state, done)

    def remember_batch(
        self,
        states: np.ndarray,
        action_indices: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        self.replay_buffer.add_batch(states, action_indices, rewards, next_states, dones)

    def train_step(
        self,
        collect_diagnostics: bool = True,
        num_new_transitions: int = 1,
    ) -> dict:
        buffer_size = len(self.replay_buffer)
        self.pending_transitions += max(0, int(num_new_transitions))
        if buffer_size < max(self.batch_size, self.warmup_size):
            info = {
                "status": "warmup",
                "loss": None,
                "td_error": None,
                "predicted_q": None,
                "target_q": None,
                "buffer_size": buffer_size,
                "warmup_remaining": max(0, self.warmup_size - buffer_size),
                "synced_target": False,
                "target_sync_remaining": self.target_sync_interval,
                "grad_norm": None,
                "did_update": False,
                "update_count": 0,
                "batch_size": int(self.batch_size),
                "trainer_mode": self.last_train_info.get("trainer_mode", "single"),
            }
            self.last_train_info = info
            return info

        if self.pending_transitions < self.update_every_transitions:
            info = dict(self.last_train_info)
            info.update(
                {
                    "status": "collecting",
                    "buffer_size": buffer_size,
                    "warmup_remaining": 0,
                    "did_update": False,
                    "update_count": 0,
                    "batch_size": int(self.batch_size),
                }
            )
            self.last_train_info = info
            return info

        self.pending_transitions = max(0, self.pending_transitions - self.update_every_transitions)

        synced_target = False
        loss_values = []
        td_values = []
        predicted_values = []
        target_values = []
        example_predicted = None
        example_target = None
        example_td = None
        grad_norm_value = None

        update_count = 0
        for _ in range(self.gradient_steps_per_update):
            if len(self.replay_buffer) < max(self.batch_size, self.warmup_size):
                break

            states_t, actions_t, rewards_t, next_states_t, dones_t = self.replay_buffer.sample_tensors(
                self.batch_size,
                device=self.device,
                pin_memory=self.pin_memory_transfers,
            )

            predicted_all = self.policy_net(states_t)
            predicted_selected = predicted_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_best = self.target_net(next_states_t).max(dim=1).values
                target = rewards_t + (1.0 - dones_t) * self.gamma * next_best

            td_error = target - predicted_selected
            loss = self.loss_fn(predicted_selected, target)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=5.0)
            self.optimizer.step()

            self.train_steps += 1
            update_count += 1
            if self.train_steps % self.target_sync_interval == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                synced_target = True

            if collect_diagnostics:
                loss_values.append(float(loss.detach().cpu().item()))
                td_values.append(float(td_error.detach().abs().mean().cpu().item()))
                predicted_values.append(float(predicted_selected.detach().mean().cpu().item()))
                target_values.append(float(target.detach().mean().cpu().item()))
                grad_norm_value = float(grad_norm.detach().cpu().item())
                example_predicted = float(predicted_selected[0].detach().cpu().item())
                example_target = float(target[0].detach().cpu().item())
                example_td = float(td_error[0].detach().cpu().item())

        info = {
            "status": "training" if update_count else "collecting",
            "buffer_size": buffer_size,
            "warmup_remaining": 0,
            "synced_target": synced_target,
            "target_sync_remaining": self.target_sync_interval
            - (self.train_steps % self.target_sync_interval),
            "did_update": bool(update_count),
            "update_count": int(update_count),
            "batch_size": int(self.batch_size),
            "trainer_mode": self.last_train_info.get("trainer_mode", "single"),
        }
        if collect_diagnostics and update_count:
            info.update(
                {
                    "loss": float(sum(loss_values) / max(1, len(loss_values))),
                    "td_error": float(sum(td_values) / max(1, len(td_values))),
                    "predicted_q": float(sum(predicted_values) / max(1, len(predicted_values))),
                    "target_q": float(sum(target_values) / max(1, len(target_values))),
                    "example_predicted_q": example_predicted,
                    "example_target_q": example_target,
                    "example_td_error": example_td,
                    "grad_norm": grad_norm_value,
                }
            )
        else:
            info.update(
                {
                    "loss": self.last_train_info.get("loss"),
                    "td_error": self.last_train_info.get("td_error"),
                    "predicted_q": self.last_train_info.get("predicted_q"),
                    "target_q": self.last_train_info.get("target_q"),
                    "example_predicted_q": self.last_train_info.get("example_predicted_q"),
                    "example_target_q": self.last_train_info.get("example_target_q"),
                    "example_td_error": self.last_train_info.get("example_td_error"),
                    "grad_norm": self.last_train_info.get("grad_norm"),
                }
            )
        self.last_train_info = info
        return info

    def decay_epsilon(self) -> None:
        self.base_epsilon = max(
            self.epsilon_min,
            float(self.base_epsilon) * float(self.epsilon_decay),
        )
        if self.reheat_active_epsilon is not None:
            self.reheat_active_epsilon = max(
                self.base_epsilon,
                float(self.reheat_active_epsilon) * float(self.epsilon_decay),
            )
            if self.reheat_active_epsilon <= self.base_epsilon + 1e-9:
                self.reheat_active_epsilon = None
        self._refresh_exploration_state()

    def record_episode_outcome(self, score: float, moving_avg: float) -> None:
        score = float(score)
        moving_avg = float(moving_avg)
        improved = False

        if score > self.best_episode_score:
            self.best_episode_score = score
            improved = True
        if moving_avg > self.best_moving_avg + self.reheat_avg_margin:
            self.best_moving_avg = moving_avg
            improved = True
        elif self.best_moving_avg == float("-inf"):
            self.best_moving_avg = moving_avg

        if improved:
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        if (
            self.reheat_active_epsilon is None
            and self.cooldown_remaining <= 0
            and self.plateau_counter >= self.reheat_patience
            and self.base_epsilon < self.reheat_epsilon
        ):
            self.reheat_active_epsilon = max(
                float(self.reheat_epsilon),
                float(self.base_epsilon),
                float(self.epsilon),
            )
            self.reheat_count += 1
            self.cooldown_remaining = int(self.reheat_cooldown)
            self.plateau_counter = 0

        self._refresh_exploration_state()

    def exploration_status(self) -> dict:
        return {
            "mode": self.exploration_mode,
            "reheat_count": int(self.reheat_count),
            "plateau_counter": int(self.plateau_counter),
            "plateau_patience": int(self.reheat_patience),
            "cooldown_remaining": int(self.cooldown_remaining),
            "base_epsilon": float(self.base_epsilon),
            "reheat_epsilon": (
                None
                if self.reheat_active_epsilon is None
                else float(self.reheat_active_epsilon)
            ),
            "epsilon": float(self.epsilon),
        }

    def _refresh_exploration_state(self) -> None:
        if (
            self.reheat_active_epsilon is not None
            and self.reheat_active_epsilon > self.base_epsilon + 1e-9
        ):
            self.exploration_mode = "reheat"
            self.epsilon = float(self.reheat_active_epsilon)
        else:
            self.reheat_active_epsilon = None
            self.exploration_mode = "decay"
            self.epsilon = float(self.base_epsilon)

    def index_to_action(self, index: int) -> List[int]:
        action = [0, 0, 0]
        action[index] = 1
        return action

    def action_to_index(self, action: Iterable[int]) -> int:
        return list(action).index(1)

    def describe_state(self, state: np.ndarray) -> List[dict]:
        values = np.asarray(state, dtype=np.float32).tolist()
        described = []
        for index, ((key, label, short_label), value) in enumerate(
            zip(self.STATE_LABELS, values)
        ):
            described.append(
                {
                    "key": key,
                    "label": label,
                    "short_label": short_label,
                    "value": float(value),
                    "index": index,
                }
            )
        return described

    def explain_food_view(self, state: np.ndarray) -> str:
        parts = []
        if state[7] > 0.5:
            parts.append("left")
        if state[8] > 0.5:
            parts.append("right")
        if state[9] > 0.5:
            parts.append("up")
        if state[10] > 0.5:
            parts.append("down")
        if not parts:
            return "Food is aligned with the snake head on at least one axis."
        return "Food is " + " and ".join(parts) + " of the head."

    def inspect_network(self, state: np.ndarray, chosen_action_index: int | None = None) -> dict:
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            activations = self.policy_net.forward_with_activations(state_tensor)

        described_state = self.describe_state(state)
        input_nodes = self._build_input_nodes(described_state, limit=8)

        layers = [
            self._build_layer_view(
                title="Input",
                short_title="Input",
                kind="input",
                size=len(self.STATE_LABELS),
                nodes=input_nodes,
                stats={
                    "mean_activation": float(np.mean(np.abs(state))),
                    "max_activation": float(np.max(np.abs(state))) if len(state) else 0.0,
                    "active_count": sum(abs(item["value"]) > 1e-6 for item in described_state),
                },
            )
        ]

        hidden_highlights = []
        connection_blocks = []
        previous_nodes = input_nodes
        previous_title = "Input"
        previous_short_title = "Input"

        for layer_index, hidden_tensor in enumerate(activations["hidden_post"], start=1):
            hidden_values = hidden_tensor.squeeze(0).detach().cpu().tolist()
            hidden_nodes = self._build_hidden_nodes(
                hidden_values,
                prefix=f"H{layer_index}",
                limit=8,
            )
            hidden_layer = self._build_layer_view(
                title=f"Hidden {layer_index}",
                short_title=f"H{layer_index}",
                kind="hidden",
                size=len(hidden_values),
                nodes=hidden_nodes,
                stats={
                    "mean_activation": float(np.mean(hidden_values)) if hidden_values else 0.0,
                    "max_activation": float(max(hidden_values)) if hidden_values else 0.0,
                    "active_count": sum(value > 0.0 for value in hidden_values),
                },
            )
            layers.append(hidden_layer)
            hidden_highlights.append(
                {
                    "title": hidden_layer["title"],
                    "items": [
                        {
                            "label": node["label"],
                            "short_label": node["short_label"],
                            "value": float(node["value"]),
                        }
                        for node in hidden_nodes[:4]
                    ],
                }
            )

            weight_matrix = self.policy_net.linears[layer_index - 1].weight.detach().cpu()
            connection_blocks.append(
                self._build_connection_block(
                    title=f"{previous_title} -> Hidden {layer_index}",
                    short_title=f"{previous_short_title} -> H{layer_index}",
                    weights=weight_matrix,
                    source_nodes=previous_nodes,
                    target_nodes=hidden_nodes,
                )
            )
            previous_nodes = hidden_nodes
            previous_title = f"Hidden {layer_index}"
            previous_short_title = f"H{layer_index}"

        output_values = activations["output"].squeeze(0).detach().cpu().tolist()
        output_nodes = self._build_output_nodes(output_values, chosen_action_index)
        output_layer = self._build_layer_view(
            title="Output",
            short_title="Output",
            kind="output",
            size=len(self.ACTION_LABELS),
            nodes=output_nodes,
            stats={
                "mean_activation": float(np.mean(output_values)) if output_values else 0.0,
                "max_activation": float(max(output_values)) if output_values else 0.0,
                "active_count": len(output_values),
            },
        )
        layers.append(output_layer)

        output_weights = self.policy_net.linears[-1].weight.detach().cpu()
        connection_blocks.append(
            self._build_connection_block(
                title=f"{previous_title} -> Output",
                short_title=f"{previous_short_title} -> Out",
                weights=output_weights,
                source_nodes=previous_nodes,
                target_nodes=output_nodes,
            )
        )

        layer_stats = [
            {"label": "Architecture", "value": self.architecture_label},
            {"label": "Hidden layers", "value": str(len(self.hidden_layers))},
            {
                "label": "Sampled neurons",
                "value": f"{sum(len(layer['nodes']) for layer in layers)}/{sum(layer['size'] for layer in layers)}",
            },
            {"label": "Policy device", "value": self.device_label},
        ]
        for layer in layers:
            stats = layer["stats"]
            layer_stats.append(
                {
                    "label": f"{layer['short_title']} active",
                    "value": f"{stats['active_count']}/{layer['size']}",
                }
            )

        dominant_path_nodes = []
        for layer in layers:
            nodes = layer.get("nodes", [])
            if nodes:
                dominant_path_nodes.append(nodes[0]["label"])

        return {
            "architecture_label": self.architecture_label,
            "layers": layers,
            "connection_blocks": connection_blocks,
            "dominant_path": " -> ".join(dominant_path_nodes),
            "top_inputs": [
                {
                    "label": node["label"],
                    "short_label": node["short_label"],
                    "value": float(node["value"]),
                }
                for node in input_nodes[:6]
            ],
            "hidden_highlights": hidden_highlights,
            "output_nodes": [
                {
                    "label": node["label"],
                    "short_label": node["short_label"],
                    "value": float(node["value"]),
                    "chosen": bool(node.get("chosen", False)),
                }
                for node in output_nodes
            ],
            "layer_stats": layer_stats,
        }

    def save(self, path: str, extra_state: dict | None = None) -> None:
        checkpoint = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "n_games": self.n_games,
            "train_steps": self.train_steps,
            "replay_buffer": self.replay_buffer.state_dict(),
            "last_train_info": self.last_train_info,
            "network_config": {
                "hidden_layers": list(self.hidden_layers),
                "input_size": len(self.STATE_LABELS),
                "output_size": len(self.ACTION_LABELS),
            },
            "runtime_config": {
                "device_preference": self.device_preference,
                "device_label": self.device_label,
                "batch_size": int(self.batch_size),
                "warmup_size": int(self.warmup_size),
                "target_sync_interval": int(self.target_sync_interval),
                "update_every_transitions": int(self.update_every_transitions),
                "gradient_steps_per_update": int(self.gradient_steps_per_update),
                "pending_transitions": int(self.pending_transitions),
            },
            "exploration_state": {
                "base_epsilon": float(self.base_epsilon),
                "reheat_active_epsilon": self.reheat_active_epsilon,
                "reheat_epsilon": float(self.reheat_epsilon),
                "reheat_patience": int(self.reheat_patience),
                "reheat_cooldown": int(self.reheat_cooldown),
                "reheat_avg_margin": float(self.reheat_avg_margin),
                "reheat_count": int(self.reheat_count),
                "plateau_counter": int(self.plateau_counter),
                "cooldown_remaining": int(self.cooldown_remaining),
                "best_episode_score": float(self.best_episode_score),
                "best_moving_avg": float(self.best_moving_avg),
                "exploration_mode": str(self.exploration_mode),
            },
            "extra_state": extra_state or {},
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> dict:
        checkpoint = torch.load(path, map_location=self.device)
        checkpoint_hidden_layers = extract_hidden_layers_from_checkpoint(checkpoint)
        if checkpoint_hidden_layers != self.hidden_layers:
            raise ValueError(
                "Checkpoint architecture "
                f"{checkpoint_hidden_layers} does not match agent architecture {self.hidden_layers}."
            )

        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._move_optimizer_state_to_device()
        self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
        self.n_games = int(checkpoint.get("n_games", 0))
        self.train_steps = int(checkpoint.get("train_steps", 0))
        self.last_train_info = checkpoint.get("last_train_info", self.last_train_info)
        runtime_config = checkpoint.get("runtime_config", {})
        self.batch_size = int(runtime_config.get("batch_size", self.batch_size))
        self.warmup_size = int(runtime_config.get("warmup_size", self.warmup_size))
        self.target_sync_interval = int(
            runtime_config.get("target_sync_interval", self.target_sync_interval)
        )
        self.update_every_transitions = int(
            runtime_config.get("update_every_transitions", self.update_every_transitions)
        )
        self.gradient_steps_per_update = int(
            runtime_config.get("gradient_steps_per_update", self.gradient_steps_per_update)
        )
        self.pending_transitions = int(
            runtime_config.get("pending_transitions", self.pending_transitions)
        )
        exploration_state = checkpoint.get("exploration_state", {})
        self.base_epsilon = float(
            exploration_state.get("base_epsilon", checkpoint.get("epsilon", self.epsilon))
        )
        active_reheat = exploration_state.get("reheat_active_epsilon")
        self.reheat_active_epsilon = (
            None if active_reheat is None else float(active_reheat)
        )
        self.reheat_epsilon = float(
            exploration_state.get("reheat_epsilon", self.reheat_epsilon)
        )
        self.reheat_patience = int(
            exploration_state.get("reheat_patience", self.reheat_patience)
        )
        self.reheat_cooldown = int(
            exploration_state.get("reheat_cooldown", self.reheat_cooldown)
        )
        self.reheat_avg_margin = float(
            exploration_state.get("reheat_avg_margin", self.reheat_avg_margin)
        )
        self.reheat_count = int(
            exploration_state.get("reheat_count", self.reheat_count)
        )
        self.plateau_counter = int(
            exploration_state.get("plateau_counter", self.plateau_counter)
        )
        self.cooldown_remaining = int(
            exploration_state.get("cooldown_remaining", self.cooldown_remaining)
        )
        self.best_episode_score = float(
            exploration_state.get("best_episode_score", self.best_episode_score)
        )
        self.best_moving_avg = float(
            exploration_state.get("best_moving_avg", self.best_moving_avg)
        )
        self.exploration_mode = str(
            exploration_state.get("exploration_mode", self.exploration_mode)
        )
        self._refresh_exploration_state()
        replay_state = checkpoint.get("replay_buffer")
        if replay_state:
            self.replay_buffer.load_state_dict(replay_state)
        return checkpoint.get("extra_state", {})

    def encode_state(self, game, out: np.ndarray | None = None) -> np.ndarray:
        head = game.head
        block = game.block_size
        direction = game.direction

        point_straight = self._next_point(head, direction, block)
        point_right = self._next_point(head, self._turn_right(direction), block)
        point_left = self._next_point(head, self._turn_left(direction), block)

        max_x_steps = max(1, game.board_w // block)
        max_y_steps = max(1, game.board_h // block)
        max_travel = float(max(max_x_steps, max_y_steps))
        total_cells = float(max_x_steps * max_y_steps)

        food_dx_cells = (game.food.x - game.head.x) / float(block)
        food_dy_cells = (game.food.y - game.head.y) / float(block)

        target = out
        if target is None:
            target = np.empty(len(self.STATE_LABELS), dtype=np.float32)

        target[0] = float(game.is_collision(point_straight))
        target[1] = float(game.is_collision(point_right))
        target[2] = float(game.is_collision(point_left))
        target[3] = float(direction == Direction.LEFT)
        target[4] = float(direction == Direction.RIGHT)
        target[5] = float(direction == Direction.UP)
        target[6] = float(direction == Direction.DOWN)
        target[7] = float(game.food.x < game.head.x)
        target[8] = float(game.food.x > game.head.x)
        target[9] = float(game.food.y < game.head.y)
        target[10] = float(game.food.y > game.head.y)
        target[11] = float(food_dx_cells / max_x_steps)
        target[12] = float(food_dy_cells / max_y_steps)
        target[13] = self._free_space_ratio(game, head, direction, max_travel)
        target[14] = self._free_space_ratio(
            game, head, self._turn_right(direction), max_travel
        )
        target[15] = self._free_space_ratio(
            game, head, self._turn_left(direction), max_travel
        )
        target[16] = float(len(game.snake) / total_cells)
        target[17] = float(
            (abs(food_dx_cells) + abs(food_dy_cells)) / float(max_x_steps + max_y_steps)
        )
        return target

    def encode_states(self, games, out: np.ndarray | None = None) -> np.ndarray:
        game_count = len(games)
        target = out
        if target is None:
            target = np.empty((game_count, len(self.STATE_LABELS)), dtype=np.float32)
        for index, game in enumerate(games):
            self.encode_state(game, out=target[index])
        return target

    def _build_layer_view(
        self,
        title: str,
        short_title: str,
        kind: str,
        size: int,
        nodes: List[dict],
        stats: dict,
    ) -> dict:
        return {
            "title": title,
            "short_title": short_title,
            "kind": kind,
            "size": int(size),
            "nodes": list(nodes),
            "sampled_count": len(nodes),
            "heatmap": self._build_heatmap(nodes),
            "stats": {
                "mean_activation": float(stats.get("mean_activation", 0.0)),
                "max_activation": float(stats.get("max_activation", 0.0)),
                "active_count": int(stats.get("active_count", 0)),
            },
        }

    def _build_input_nodes(self, state_items: List[dict], limit: int) -> List[dict]:
        top_items = sorted(state_items, key=lambda item: abs(item["value"]), reverse=True)[
            :limit
        ]
        max_abs = max(1e-6, max(abs(item["value"]) for item in top_items))
        return [
            {
                "index": int(item["index"]),
                "label": item["label"],
                "short_label": item["short_label"],
                "value": float(item["value"]),
                "strength": float(abs(item["value"]) / max_abs),
                "positive": bool(item["value"] >= 0.0),
            }
            for item in top_items
        ]

    def _build_hidden_nodes(self, values: List[float], prefix: str, limit: int) -> List[dict]:
        indexed = sorted(
            enumerate(values),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:limit]
        max_abs = max(1e-6, max(abs(value) for _, value in indexed))
        return [
            {
                "index": int(index),
                "label": f"{prefix}-{index}",
                "short_label": f"{prefix}-{index}",
                "value": float(value),
                "strength": float(abs(value) / max_abs),
                "positive": bool(value >= 0.0),
            }
            for index, value in indexed
        ]

    def _build_output_nodes(
        self, values: List[float], chosen_action_index: int | None
    ) -> List[dict]:
        max_abs = max(1e-6, max(abs(value) for value in values))
        return [
            {
                "index": int(index),
                "label": self.ACTION_LABELS[index],
                "short_label": self.ACTION_SHORT_LABELS[index],
                "value": float(value),
                "strength": float(abs(value) / max_abs),
                "positive": bool(value >= 0.0),
                "chosen": bool(chosen_action_index == index),
            }
            for index, value in enumerate(values)
        ]

    def _build_heatmap(self, nodes: List[dict]) -> dict:
        if not nodes:
            return {"columns": 1, "cells": []}

        columns = 2 if len(nodes) <= 4 else 3
        return {
            "columns": columns,
            "cells": [
                {
                    "rank": index,
                    "label": node["short_label"],
                    "full_label": node["label"],
                    "value": float(node["value"]),
                    "strength": float(node["strength"]),
                    "positive": bool(node.get("positive", True)),
                    "chosen": bool(node.get("chosen", False)),
                }
                for index, node in enumerate(nodes)
            ],
        }

    def _build_connection_block(
        self,
        title: str,
        short_title: str,
        weights,
        source_nodes: List[dict],
        target_nodes: List[dict],
    ) -> dict:
        max_abs = float(weights.abs().max().item())
        max_abs = max(1e-6, max_abs)
        rows = []
        total_weights = []
        positive_count = 0

        for target in target_nodes:
            row = []
            for source in source_nodes:
                weight = float(weights[target["index"], source["index"]].item())
                total_weights.append(abs(weight))
                if weight >= 0.0:
                    positive_count += 1
                row.append(
                    {
                        "weight": weight,
                        "strength": abs(weight) / max_abs,
                        "positive": bool(weight >= 0.0),
                    }
                )
            rows.append(row)

        return {
            "title": title,
            "short_title": short_title,
            "rows": rows,
            "source_labels": [node["short_label"] for node in source_nodes],
            "target_labels": [node["short_label"] for node in target_nodes],
            "summary": {
                "max_abs": max(total_weights) if total_weights else 0.0,
                "mean_abs": float(np.mean(total_weights)) if total_weights else 0.0,
                "positive_ratio": (
                    float(positive_count / max(1, len(total_weights))) if total_weights else 0.0
                ),
            },
        }

    def _build_action_details(
        self, action_index: int, q_values: List[float], decision_type: str
    ) -> dict:
        return {
            "action": self.index_to_action(action_index),
            "action_index": action_index,
            "action_key": self.ACTION_KEYS[action_index],
            "action_label": self.ACTION_LABELS[action_index],
            "decision_type": decision_type,
            "q_values": list(q_values),
        }

    def _build_action_summary(self, action_index: int, decision_type: str) -> dict:
        return {
            "action": self.index_to_action(action_index),
            "action_index": action_index,
            "action_key": self.ACTION_KEYS[action_index],
            "action_label": self.ACTION_LABELS[action_index],
            "decision_type": decision_type,
        }

    def _get_q_tensor(self, state: np.ndarray):
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            return self.policy_net(state_tensor.unsqueeze(0)).squeeze(0)

    def _get_q_tensor_batch(self, states: np.ndarray):
        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            return self.policy_net(states_tensor)

    def _q_tensor_to_list(self, q_tensor) -> List[float]:
        return [float(value) for value in q_tensor.detach().cpu().tolist()]

    def _select_best_action_index_from_q_tensor(self, q_tensor) -> int:
        best_value = q_tensor.max()
        best_indices = torch.nonzero(q_tensor == best_value, as_tuple=False).flatten()
        if best_indices.numel() <= 1:
            return int(best_indices[0].item())
        choice = best_indices[torch.randint(best_indices.numel(), (1,), device=best_indices.device)]
        return int(choice.item())

    def _move_optimizer_state_to_device(self) -> None:
        for state in self.optimizer.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.device)

    def _free_space_ratio(
        self, game, start: Point, direction: Direction, max_travel: float
    ) -> float:
        steps = game.raycast_free_steps(start, direction)
        return float(steps / max_travel)

    def _next_point(self, head: Point, direction: Direction, block_size: int) -> Point:
        if direction == Direction.RIGHT:
            return Point(head.x + block_size, head.y)
        if direction == Direction.LEFT:
            return Point(head.x - block_size, head.y)
        if direction == Direction.UP:
            return Point(head.x, head.y - block_size)
        return Point(head.x, head.y + block_size)

    def _turn_right(self, direction: Direction) -> Direction:
        turns = {
            Direction.RIGHT: Direction.DOWN,
            Direction.DOWN: Direction.LEFT,
            Direction.LEFT: Direction.UP,
            Direction.UP: Direction.RIGHT,
        }
        return turns[direction]

    def _turn_left(self, direction: Direction) -> Direction:
        turns = {
            Direction.RIGHT: Direction.UP,
            Direction.UP: Direction.LEFT,
            Direction.LEFT: Direction.DOWN,
            Direction.DOWN: Direction.RIGHT,
        }
        return turns[direction]
