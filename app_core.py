from __future__ import annotations

import argparse
import copy
import os
import time
from collections import deque

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency 'numpy'. Install requirements.txt before running the Deep RL project."
    ) from exc

try:
    import pygame
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency 'pygame'. Install requirements.txt before running the Deep RL project."
    ) from exc

from dashboard import TrainingDashboard
from dqn_agent import (
    DEFAULT_HIDDEN_LAYERS,
    DQNAgent,
    cuda_is_available,
    load_checkpoint_network_config,
    normalize_hidden_layers,
)
from metrics_utils import (
    append_metric_entry,
    append_metric_entries,
    build_history,
    group_entries_by_algo,
    load_histories_from_log,
    load_metric_entries,
    rewrite_metric_entries,
)
from snake_game import SnakeGameAI, SnakeLogicEnv
from tabular_agent import QLearningAgent


DEFAULT_REWARD_CONFIG = {"food": 10.0, "death": -10.0, "step": 0.0}
DEFAULT_TRAINER_MODE = "single"
SUPPORTED_TRAINER_MODES = ("single", "parallel")
SUMMARY_IMAGE_NAME = "training_summary.png"


def parse_hidden_layers_arg(raw_value):
    if raw_value is None:
        return None

    pieces = [piece.strip() for piece in str(raw_value).split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise argparse.ArgumentTypeError(
            "Hidden layers must be a comma-separated list like 64 or 128,128."
        )

    try:
        return normalize_hidden_layers(int(piece) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_positive_int_arg(raw_value):
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Value must be a positive integer.") from exc

    if value <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return value


def parse_trainer_mode_arg(raw_value):
    normalized = str(raw_value).strip().lower()
    if normalized not in SUPPORTED_TRAINER_MODES:
        raise argparse.ArgumentTypeError(
            f"Trainer mode must be one of: {', '.join(SUPPORTED_TRAINER_MODES)}."
        )
    return normalized


def resolve_trainer_mode_for_session(checkpoint_extra_state, resume, explicit_trainer_mode):
    if explicit_trainer_mode:
        return explicit_trainer_mode
    if resume:
        trainer_config = dict((checkpoint_extra_state or {}).get("trainer_config", {}))
        restored_mode = str(trainer_config.get("mode", "")).strip().lower()
        if restored_mode in SUPPORTED_TRAINER_MODES:
            return restored_mode
    return DEFAULT_TRAINER_MODE


def resolve_hidden_layers_for_session(checkpoint_path, resume, explicit_hidden_layers):
    if resume and os.path.exists(checkpoint_path):
        checkpoint_config = load_checkpoint_network_config(checkpoint_path)
        checkpoint_hidden_layers = checkpoint_config["hidden_layers"]
        if (
            explicit_hidden_layers is not None
            and list(explicit_hidden_layers) != list(checkpoint_hidden_layers)
        ):
            raise ValueError(
                "Checkpoint architecture mismatch: checkpoint uses "
                f"{checkpoint_hidden_layers}, but --hidden-layers requested "
                f"{list(explicit_hidden_layers)}."
            )
        return list(checkpoint_hidden_layers)

    if explicit_hidden_layers is not None:
        return list(explicit_hidden_layers)

    return list(DEFAULT_HIDDEN_LAYERS)


def hold_training_window_open(
    game,
    dashboard,
    recent_episode_replays=None,
    *,
    base_view=None,
    summary_image_path=None,
    agent=None,
    trainer_mode="single",
    fast_mode_requested=False,
    fast_mode_effective=False,
    fast_tail_episodes=5,
    eval_tail_episodes=3,
    parallel_phase="bulk",
):
    pygame.event.clear()
    recent_episode_replays = list(recent_episode_replays or [])
    base_results_view = copy.deepcopy(base_view or game.dashboard_data or {})
    recent_replays_panel = build_recent_replays_panel(
        dashboard,
        recent_episode_replays,
        trainer_mode=trainer_mode,
        fast_mode_requested=fast_mode_requested,
        fast_mode_effective=fast_mode_effective,
        fast_tail_episodes=max(1, int(fast_tail_episodes)),
        eval_tail_episodes=max(1, int(eval_tail_episodes)),
        parallel_phase=parallel_phase,
        training_completed=True,
    )

    summary_view = copy.deepcopy(base_results_view)
    summary_view.pop("overlay_title", None)
    summary_view.pop("overlay_subtitle", None)
    summary_view.pop("overlay_buttons", None)
    if summary_image_path:
        game.set_dashboard_data(summary_view)
        game.draw()
        pygame.image.save(game.display, summary_image_path)

    game.set_dashboard_data(
        build_training_finished_view(
            base_results_view,
            game,
            recent_episode_replays,
            recent_replays_panel=recent_replays_panel,
        )
    )

    while not game.quit_requested:
        events = pygame.event.get()
        scaled_events = game.scale_events(events)
        game.handle_system_events(events)

        replay_index = None
        for event in scaled_events:
            if event.type == pygame.KEYDOWN and event.key in (
                pygame.K_RETURN,
                pygame.K_KP_ENTER,
                pygame.K_q,
                pygame.K_ESCAPE,
            ):
                return
            if event.type == pygame.KEYDOWN and event.key in (
                pygame.K_1,
                pygame.K_2,
                pygame.K_3,
            ):
                requested_index = event.key - pygame.K_1
                if requested_index < len(recent_episode_replays):
                    replay_index = requested_index
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                replay_buttons = list(game.dashboard_data.get("overlay_buttons", []))
                replay_buttons.extend(
                    game.dashboard_data.get("recent_replays", {}).get("buttons", [])
                )
                for button in replay_buttons:
                    rect = pygame.Rect(
                        button["x"],
                        button["y"],
                        button["w"],
                        button["h"],
                    )
                    if rect.collidepoint(event.pos):
                        replay_index = int(button.get("replay_index", -1))
                        break

        if replay_index is not None and 0 <= replay_index < len(recent_episode_replays):
            play_episode_replay(game, dashboard, agent, recent_episode_replays[replay_index])
            if game.quit_requested:
                return
            game.set_dashboard_data(
                build_training_finished_view(
                    base_results_view,
                    game,
                    recent_episode_replays,
                    recent_replays_panel=recent_replays_panel,
                )
            )
            continue

        dashboard.sync_graph_rect(game)
        dashboard.handle_events(scaled_events)
        game.dashboard_data["graph_view_end"] = dashboard.graph_view_end
        game.dashboard_data["graph_view_size"] = dashboard.graph_view_size
        game.dashboard_data["graph_hover_index"] = dashboard.graph_hover_index
        game.draw()
        pygame.time.delay(30)


def prepare_metrics_log(metrics_log_path, resume):
    if not metrics_log_path:
        return {}

    if not resume:
        entries = load_metric_entries(metrics_log_path)
        grouped = group_entries_by_algo(entries)
        baseline_entries = grouped.get("tabular", [])
        rewrite_metric_entries(metrics_log_path, baseline_entries)

    return load_histories_from_log(metrics_log_path)


def replace_algo_entries(metrics_log_path, algo, replacement_entries):
    existing = load_metric_entries(metrics_log_path)
    filtered = [entry for entry in existing if entry.get("algo") != algo]
    rewrite_metric_entries(metrics_log_path, filtered + list(replacement_entries))


def build_metric_entry(
    algo,
    episode,
    score,
    episode_reward,
    moving_avg,
    epsilon,
    loss,
    steps,
    buffer_size,
    best_score,
):
    return {
        "algo": algo,
        "episode": int(episode),
        "score": int(score),
        "episode_reward": float(episode_reward),
        "moving_avg_20": float(moving_avg),
        "epsilon": float(epsilon),
        "loss": None if loss is None else float(loss),
        "steps": int(steps),
        "buffer_size": int(buffer_size),
        "best_score": int(best_score),
    }


def run_tabular_baseline(episodes, reward_config, metrics_log_path):
    if metrics_log_path:
        grouped = group_entries_by_algo(load_metric_entries(metrics_log_path))
        existing = grouped.get("tabular", [])
        if len(existing) >= episodes:
            return build_history(existing)

    baseline_entries = []
    agent = QLearningAgent()
    game = SnakeLogicEnv(speed=0)
    game.set_reward_config(reward_config)

    best_score = 0
    score_history = []

    try:
        for episode_index in range(1, episodes + 1):
            game.reset()
            state = agent.get_state(game)
            episode_reward = 0.0
            step_count = 0

            while True:
                action_info = agent.get_action_details(state)
                reward, game_over, score = game.play_step(
                    action_info["action"],
                    events=[],
                    draw_frame=False,
                )
                episode_reward += reward
                step_count += 1

                next_state = agent.get_state(game)
                agent.train_step(
                    state=state,
                    action_index=action_info["action_index"],
                    reward=reward,
                    next_state=next_state,
                    done=game_over,
                )
                state = next_state

                if game_over:
                    break

            agent.n_games += 1
            agent.decay_epsilon()
            best_score = max(best_score, score)
            score_history.append(score)
            moving_avg = sum(score_history[-20:]) / len(score_history[-20:])
            baseline_entries.append(
                build_metric_entry(
                    algo="tabular",
                    episode=episode_index,
                    score=score,
                    episode_reward=episode_reward,
                    moving_avg=moving_avg,
                    epsilon=agent.epsilon,
                    loss=0.0,
                    steps=step_count,
                    buffer_size=len(agent.q_table),
                    best_score=best_score,
                )
            )
    finally:
        game.close()

    if metrics_log_path:
        replace_algo_entries(metrics_log_path, "tabular", baseline_entries)

    return build_history(baseline_entries)


def save_checkpoint(
    agent,
    checkpoint_path,
    dashboard,
    metrics_log_path,
    last_transition,
    *,
    trainer_mode="single",
    parallel_envs=8,
    eval_tail_episodes=3,
):
    extra_state = {
        "reward_config": dashboard.reward_config,
        "deep_history": {
            "scores": list(dashboard.deep_scores),
            "moving_avg": list(dashboard.deep_average_history),
            "best_scores": list(dashboard.deep_best_history),
            "episode_rewards": list(dashboard.deep_episode_rewards),
            "losses": list(dashboard.deep_loss_history),
            "loss_moving_avg": list(dashboard.deep_loss_average_history),
        },
        "metrics_log_path": metrics_log_path,
        "last_transition": dict(last_transition or {}),
        "trainer_config": {
            "mode": str(trainer_mode or DEFAULT_TRAINER_MODE).strip().lower(),
            "parallel_envs": int(max(1, parallel_envs)),
            "eval_tail_episodes": int(max(1, eval_tail_episodes)),
        },
    }
    agent.save(checkpoint_path, extra_state=extra_state)


def default_summary_image_path():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), SUMMARY_IMAGE_NAME)


def capture_replay_frame(
    game,
    *,
    state,
    action_info,
    current_game_number,
    episode_goal,
    best_score,
    context,
    show_baseline=True,
    overlay_title=None,
    overlay_subtitle=None,
):
    return {
        "snake": list(game.snake),
        "head": game.head,
        "food": game.food,
        "direction": game.direction,
        "score": int(game.score),
        "frame_iteration": int(game.frame_iteration),
        "state": np.asarray(state, dtype=np.float32).tolist(),
        "action_info": copy.deepcopy(action_info),
        "current_game_number": int(current_game_number),
        "episode_goal": int(max(current_game_number, episode_goal)),
        "best_score": int(best_score),
        "context": copy.deepcopy(context),
        "show_baseline": bool(show_baseline),
        "overlay_title": overlay_title,
        "overlay_subtitle": overlay_subtitle,
    }


def build_replay_overlay_buttons(game, recent_episode_replays):
    if not recent_episode_replays:
        return []

    buttons = []
    button_w = 170
    button_h = 38
    gap = 14
    count = len(recent_episode_replays)
    total_w = (button_w * count) + (gap * (count - 1))
    start_x = max(18, (game.board_w - total_w) // 2)
    y = (game.board_h // 2) + 44

    for index, replay in enumerate(recent_episode_replays):
        buttons.append(
            {
                "replay_index": index,
                "label": f"Replay Run {replay['run_number']}",
                "x": start_x + index * (button_w + gap),
                "y": y,
                "w": button_w,
                "h": button_h,
            }
        )
    return buttons


def build_recent_replay_button_specs(recent_episode_replays):
    buttons = []
    for index, replay in enumerate(recent_episode_replays):
        buttons.append(
            {
                "replay_index": index,
                "label": f"Run {replay['run_number']} | Score {replay['score']}",
            }
        )
    return buttons


def build_recent_replays_panel(
    dashboard,
    recent_episode_replays,
    *,
    trainer_mode="single",
    fast_mode_requested,
    fast_mode_effective,
    fast_tail_episodes,
    eval_tail_episodes=3,
    parallel_phase="bulk",
    training_completed=False,
):
    replays = list(recent_episode_replays or [])
    panel = {
        "title": "Recent Replays",
        "lines": [],
        "buttons": [],
        "footer": "",
    }

    trainer_mode = str(trainer_mode or DEFAULT_TRAINER_MODE).strip().lower()
    replay_label = "evaluation tail" if trainer_mode == "parallel" else "session tail"

    if training_completed and replays:
        panel["lines"] = [
            f"Latest {len(replays)} captured {replay_label} run(s).",
            "Click a run below or press 1/2/3 to replay it on the board.",
        ]
        panel["buttons"] = build_recent_replay_button_specs(replays)
        panel["footer"] = "Only the newest 3 runs are kept in memory."
        return panel

    if trainer_mode == "parallel":
        if training_completed:
            panel["lines"] = [
                "No evaluation-tail replays were captured in this session.",
                "The final evaluation runs must finish to appear here.",
            ]
            return panel

        if parallel_phase == "eval":
            panel["lines"] = [
                "Parallel training is complete.",
                f"The final {max(1, int(eval_tail_episodes))} evaluation run(s) are being captured now and will appear here.",
            ]
        else:
            panel["lines"] = [
                "Parallel mode trains headlessly for speed, then captures a short evaluation tail.",
                f"The final {max(1, int(eval_tail_episodes))} evaluation run(s) become replayable here after training finishes.",
            ]
        if replays:
            panel["footer"] = f"Captured so far this session: {len(replays)}/3"
        return panel

    if training_completed:
        panel["lines"] = [
            "No session-tail replays were captured in this session.",
            "The final runs must finish to appear here.",
        ]
        return panel

    if fast_mode_requested or fast_mode_effective:
        panel["lines"] = [
            "Fast mode will keep the newest 3 completed tail runs in memory.",
            f"The final {max(1, int(fast_tail_episodes))} episode(s) are still the detailed tail for this session.",
        ]
    else:
        panel["lines"] = [
            "The newest 3 completed runs from this session will appear here.",
            "This still works when No Render is on because replay data is captured logically.",
        ]
    if fast_mode_requested and fast_mode_requested != fast_mode_effective:
        panel["lines"].append("Fast mode is queued and will apply from the next episode.")
    if replays:
        panel["footer"] = f"Captured so far this session: {len(replays)}/3"
    return panel


def build_training_finished_view(
    base_view,
    game,
    recent_episode_replays,
    recent_replays_panel=None,
):
    final_view = dict(base_view or {})
    final_view["view_mode"] = "results" if final_view.get("results_ready") else "overview"
    final_view["overlay_title"] = "Training finished"
    if recent_episode_replays:
        final_view["overlay_subtitle"] = (
            "Replay one of the last 3 captured runs, or press Enter, Q, or Esc to close."
        )
        final_view["overlay_buttons"] = build_replay_overlay_buttons(game, recent_episode_replays)
    else:
        final_view["overlay_subtitle"] = "Press Enter, Q, Esc, or close the window."
        final_view["overlay_buttons"] = []
    if recent_replays_panel is not None:
        final_view["recent_replays"] = recent_replays_panel
    return final_view


def apply_replay_frame(game, dashboard, agent, frame):
    game.snake = list(frame.get("snake", []))
    game.head = frame.get("head", game.head)
    game.food = frame.get("food", game.food)
    game.direction = frame.get("direction", game.direction)
    game.score = int(frame.get("score", game.score))
    game.frame_iteration = int(frame.get("frame_iteration", game.frame_iteration))
    game.snake_body_set = set(game.snake[1:])
    stored_dashboard = frame.get("dashboard_data")
    if stored_dashboard:
        game.set_dashboard_data(copy.deepcopy(stored_dashboard))
        return

    stored_state = frame.get("state")
    state = (
        np.asarray(stored_state, dtype=np.float32)
        if stored_state is not None
        else agent.encode_state(game)
    )
    action_info = copy.deepcopy(frame.get("action_info") or agent.get_action_details(state, greedy=True))
    context = copy.deepcopy(frame.get("context") or build_training_context("Replay", 0.0, agent.last_train_info, {"reward_text": "n/a"}))
    frame_data = build_dashboard_frame(
        game=game,
        dashboard=dashboard,
        agent=agent,
        state=state,
        action_info=action_info,
        current_game_number=int(frame.get("current_game_number", 1)),
        episode_goal=int(frame.get("episode_goal", 1)),
        best_score=int(frame.get("best_score", game.score)),
        context=context,
        show_baseline=bool(frame.get("show_baseline", dashboard.baseline_visible)),
        overlay_title=frame.get("overlay_title"),
        overlay_subtitle=frame.get("overlay_subtitle"),
    )
    game.set_dashboard_data(frame_data)


def copy_env_state_to_game(game, env):
    game.snake = list(env.snake)
    game.head = env.head
    game.food = env.food
    game.direction = env.direction
    game.score = int(env.score)
    game.frame_iteration = int(env.frame_iteration)
    game.snake_body_set = set(env.snake_body_set)


def play_episode_replay(game, dashboard, agent, episode_replay):
    frames = list(episode_replay.get("frames", []))
    if not frames or game.quit_requested:
        return

    for frame in frames:
        events = pygame.event.get()
        scaled_events = game.scale_events(events)
        game.handle_system_events(events)

        for event in scaled_events:
            if event.type != pygame.KEYDOWN:
                continue
            if event.key in (
                pygame.K_RETURN,
                pygame.K_KP_ENTER,
                pygame.K_SPACE,
                pygame.K_q,
                pygame.K_ESCAPE,
            ):
                return

        if game.quit_requested:
            return

        apply_replay_frame(game, dashboard, agent, frame)
        game.draw()
        pygame.time.delay(55)


def build_live_train_info(train_info, session_perf, episode_steps):
    info = dict(train_info or {})
    elapsed = max(1e-6, time.perf_counter() - session_perf["start_time"])
    info["episode_steps"] = int(max(0, episode_steps))
    info["env_steps_per_sec"] = float(session_perf["env_steps"] / elapsed)
    info["updates_per_sec"] = float(session_perf["updates"] / elapsed)
    return info


def build_training_context(
    mode_label,
    episode_reward,
    train_info,
    transition,
    *,
    trainer_mode="single",
    parallel_envs=1,
    parallel_phase="single",
    eval_tail_episodes=3,
    fast_mode_requested=False,
    fast_mode_effective=False,
    fast_tail_episodes=5,
    episodes_remaining=0,
):
    return {
        "mode_label": mode_label,
        "episode_reward": episode_reward,
        "train_info": train_info,
        "transition": transition,
        "trainer_mode": str(trainer_mode or DEFAULT_TRAINER_MODE),
        "parallel_envs": int(max(1, parallel_envs)),
        "parallel_phase": str(parallel_phase or "single"),
        "eval_tail_episodes": int(max(1, eval_tail_episodes)),
        "fast_mode_requested": bool(fast_mode_requested),
        "fast_mode_effective": bool(fast_mode_effective),
        "fast_tail_episodes": int(fast_tail_episodes),
        "episodes_remaining": int(max(0, episodes_remaining)),
    }


def build_dashboard_frame(
    game,
    dashboard,
    agent,
    state,
    action_info,
    current_game_number,
    episode_goal,
    best_score,
    context,
    *,
    lightweight=False,
    show_baseline=True,
    recent_replays_panel=None,
    overlay_title=None,
    overlay_subtitle=None,
):
    frame = dashboard.build_dashboard_data(
        agent=agent,
        game=game,
        state=state,
        action_info=action_info,
        current_game_number=current_game_number,
        episode_goal=episode_goal,
        best_score=best_score,
        context=context,
        lightweight=lightweight,
        show_baseline=show_baseline,
    )
    if recent_replays_panel is not None:
        frame["recent_replays"] = recent_replays_panel
    if overlay_title:
        frame["overlay_title"] = overlay_title
    if overlay_subtitle:
        frame["overlay_subtitle"] = overlay_subtitle
    return frame


def draw_dashboard_frame(
    game,
    dashboard,
    agent,
    state,
    action_info,
    current_game_number,
    episode_goal,
    best_score,
    context,
    *,
    lightweight=False,
    show_baseline=True,
    recent_replays_panel=None,
    overlay_title=None,
    overlay_subtitle=None,
):
    frame = build_dashboard_frame(
        game=game,
        dashboard=dashboard,
        agent=agent,
        state=state,
        action_info=action_info,
        current_game_number=current_game_number,
        episode_goal=episode_goal,
        best_score=best_score,
        context=context,
        lightweight=lightweight,
        show_baseline=show_baseline,
        recent_replays_panel=recent_replays_panel,
        overlay_title=overlay_title,
        overlay_subtitle=overlay_subtitle,
    )
    game.set_dashboard_data(frame)
    game.draw()


def copy_game_state_to_game(target_game, source_game):
    target_game.snake = list(source_game.snake)
    target_game.head = source_game.head
    target_game.food = source_game.food
    target_game.direction = source_game.direction
    target_game.score = int(source_game.score)
    target_game.frame_iteration = int(source_game.frame_iteration)
    target_game.snake_body_set = set(source_game.snake_body_set)


def clone_dashboard_state(target_dashboard, source_dashboard):
    target_dashboard.load_deep_history(source_dashboard.export_deep_history())
    target_dashboard.set_baseline_visibility(source_dashboard.baseline_visible)
    if source_dashboard.baseline_scores:
        target_dashboard.set_baseline_history(source_dashboard.export_baseline_history())
    target_dashboard.show_arrows_toggle.value = source_dashboard.show_arrows_toggle.value
    target_dashboard.show_dangers_toggle.value = source_dashboard.show_dangers_toggle.value
    target_dashboard.show_graph_toggle.value = source_dashboard.show_graph_toggle.value
    target_dashboard.turbo_toggle.value = source_dashboard.turbo_toggle.value
    target_dashboard.show_scores_toggle.value = source_dashboard.show_scores_toggle.value
    target_dashboard.show_avg_toggle.value = source_dashboard.show_avg_toggle.value
    target_dashboard.show_best_toggle.value = source_dashboard.show_best_toggle.value
    target_dashboard.selected_device_preference = source_dashboard.selected_device_preference
    target_dashboard.selected_trainer_mode = source_dashboard.selected_trainer_mode
    target_dashboard.fast_mode_toggle.value = source_dashboard.fast_mode_toggle.value
    target_dashboard.episode_input.text = source_dashboard.episode_input.text
    target_dashboard.parallel_env_input.text = source_dashboard.parallel_env_input.text
    target_dashboard.graph_view_end = source_dashboard.graph_view_end
    target_dashboard.graph_view_size = source_dashboard.graph_view_size
    target_dashboard.pending_trainer_mode = None


def build_post_run_base_view(
    *,
    game,
    dashboard,
    agent,
    best_score,
    last_transition,
    session_perf,
    trainer_mode,
    parallel_envs,
    eval_tail_episodes,
    fast_tail_episodes,
):
    dashboard.set_results_ready(True, auto_focus=True)
    state = agent.encode_state(game)
    action_info = agent.get_action_details(state, greedy=True)
    context = build_training_context(
        "Training finished",
        0.0,
        build_live_train_info(
            agent.last_train_info,
            session_perf,
            agent.last_train_info.get("episode_steps", 0),
        ),
        last_transition,
        trainer_mode=trainer_mode,
        parallel_envs=parallel_envs,
        parallel_phase="eval" if trainer_mode == "parallel" else "single",
        eval_tail_episodes=eval_tail_episodes,
        fast_mode_requested=bool(dashboard.fast_mode_toggle.value),
        fast_mode_effective=bool(dashboard.fast_mode_toggle.value and trainer_mode == "single"),
        fast_tail_episodes=fast_tail_episodes,
        episodes_remaining=0,
    )
    return build_dashboard_frame(
        game=game,
        dashboard=dashboard,
        agent=agent,
        state=state,
        action_info=action_info,
        current_game_number=max(1, len(dashboard.deep_scores)),
        episode_goal=max(1, dashboard.get_episode_goal()),
        best_score=best_score,
        context=context,
        show_baseline=dashboard.baseline_visible,
    )


def create_post_run_results_window(
    *,
    source_game,
    source_dashboard,
    agent,
    trainer_mode,
    parallel_envs,
):
    results_game = SnakeGameAI(
        w=source_game.board_w,
        h=source_game.board_h,
        window_h=max(source_game.logical_h, 880),
        render=True,
        speed=max(10, int(source_game.speed or 10)),
    )
    results_dashboard = TrainingDashboard(
        results_game,
        initial_speed=max(10, int(source_game.speed or 10)),
        initial_delay_ms=0,
        initial_episode_goal=max(1, source_dashboard.get_episode_goal()),
        initial_reward_config=source_dashboard.reward_config,
        initial_headless=False,
        initial_device_preference=agent.device.type,
        cuda_available=cuda_is_available(),
        require_manual_start=False,
        initial_fast_mode=source_dashboard.fast_mode_toggle.value,
        initial_trainer_mode=trainer_mode,
        initial_parallel_envs=parallel_envs,
    )
    clone_dashboard_state(results_dashboard, source_dashboard)
    return results_game, results_dashboard


def show_post_run_results(
    *,
    game,
    dashboard,
    agent,
    recent_episode_replays,
    best_score,
    last_transition,
    session_perf,
    trainer_mode,
    parallel_envs,
    eval_tail_episodes,
    fast_tail_episodes,
):
    owns_window = not game.render
    results_game = game
    results_dashboard = dashboard
    if owns_window:
        results_game, results_dashboard = create_post_run_results_window(
            source_game=game,
            source_dashboard=dashboard,
            agent=agent,
            trainer_mode=trainer_mode,
            parallel_envs=parallel_envs,
        )

    if recent_episode_replays and recent_episode_replays[-1].get("frames"):
        last_frame = recent_episode_replays[-1]["frames"][-1]
        results_game.snake = list(last_frame.get("snake", results_game.snake))
        results_game.head = last_frame.get("head", results_game.head)
        results_game.food = last_frame.get("food", results_game.food)
        results_game.direction = last_frame.get("direction", results_game.direction)
        results_game.score = int(last_frame.get("score", results_game.score))
        results_game.frame_iteration = int(
            last_frame.get("frame_iteration", results_game.frame_iteration)
        )
        results_game.snake_body_set = set(results_game.snake[1:])
    else:
        copy_game_state_to_game(results_game, game)

    base_view = build_post_run_base_view(
        game=results_game,
        dashboard=results_dashboard,
        agent=agent,
        best_score=best_score,
        last_transition=last_transition,
        session_perf=session_perf,
        trainer_mode=trainer_mode,
        parallel_envs=parallel_envs,
        eval_tail_episodes=eval_tail_episodes,
        fast_tail_episodes=fast_tail_episodes,
    )
    hold_training_window_open(
        results_game,
        results_dashboard,
        recent_episode_replays,
        base_view=base_view,
        summary_image_path=default_summary_image_path(),
        agent=agent,
        trainer_mode=trainer_mode,
        fast_mode_requested=bool(results_dashboard.fast_mode_toggle.value),
        fast_mode_effective=bool(results_dashboard.fast_mode_toggle.value and trainer_mode == "single"),
        fast_tail_episodes=fast_tail_episodes,
        eval_tail_episodes=eval_tail_episodes,
        parallel_phase="eval" if trainer_mode == "parallel" else "single",
    )
    if owns_window:
        results_game.close()


def sync_agent_device_from_dashboard(agent, dashboard, session_label):
    if dashboard.selected_device_preference != agent.device_preference:
        if agent.set_device(dashboard.selected_device_preference):
            print(f"Switched {session_label} device to {agent.device_label} ({agent.device})")
            return True
    return False


def apply_pending_trainer_mode_switch(agent, dashboard):
    next_mode = dashboard.consume_pending_trainer_mode()
    if not next_mode:
        return None
    configure_agent_for_mode(agent, next_mode)
    print(f"Queued trainer-mode switch applied: {next_mode}")
    return next_mode


def ensure_baseline_history(
    dashboard,
    *,
    comparison_mode,
    fast_mode_effective,
    baseline_history,
    baseline_episodes,
    metrics_log_path,
):
    show_baseline = bool(comparison_mode and not fast_mode_effective)
    dashboard.set_baseline_visibility(show_baseline)

    if not show_baseline:
        return baseline_history

    if dashboard.baseline_scores:
        return baseline_history

    if baseline_history:
        dashboard.set_baseline_history(baseline_history)
        return baseline_history

    baseline_history = run_tabular_baseline(
        baseline_episodes,
        reward_config=dashboard.reward_config,
        metrics_log_path=metrics_log_path,
    )
    dashboard.set_baseline_history(baseline_history)
    return baseline_history


def resolve_parallel_batch_size(device_type):
    return 1024 if str(device_type) == "cuda" else 512


def resolve_parallel_env_count(device_type, requested_envs):
    if requested_envs is not None:
        return max(1, int(requested_envs))
    return 16 if str(device_type) == "cuda" else 8


def configure_agent_for_mode(agent, trainer_mode):
    trainer_mode = str(trainer_mode).strip().lower()
    if trainer_mode == "parallel":
        agent.configure_training_schedule(
            batch_size=resolve_parallel_batch_size(agent.device.type),
            warmup_size=1_000,
            target_sync_interval=250,
            update_every_transitions=32,
            gradient_steps_per_update=2,
            trainer_mode="parallel",
        )
        return

    agent.configure_training_schedule(
        batch_size=256,
        warmup_size=1_000,
        target_sync_interval=250,
        update_every_transitions=1,
        gradient_steps_per_update=1,
        trainer_mode="single",
    )


def flush_metric_buffer(metrics_log_path, pending_entries):
    if pending_entries:
        append_metric_entries(metrics_log_path, pending_entries)
        pending_entries.clear()


def maybe_build_parallel_frame(
    *,
    render,
    dashboard,
    last_frame_time,
    force=False,
):
    if not render or dashboard.headless_toggle.value:
        return False, last_frame_time

    now = time.perf_counter()
    if force or (now - last_frame_time) >= 0.18:
        return True, now
    return False, last_frame_time


def resolve_positive_session_int(checkpoint_extra_state, resume, explicit_value, key, default_value):
    if explicit_value is not None:
        return max(1, int(explicit_value))
    if resume:
        trainer_config = dict((checkpoint_extra_state or {}).get("trainer_config", {}))
        restored_value = trainer_config.get(key)
        if restored_value is not None:
            try:
                return max(1, int(restored_value))
            except (TypeError, ValueError):
                pass
    return max(1, int(default_value))


def await_training_start(
    *,
    game,
    dashboard,
    agent,
    comparison_mode,
    best_score,
    last_transition,
    session_perf,
    parallel_envs,
    eval_tail_episodes,
    fast_tail_episodes,
):
    if not game.render or dashboard.started:
        return dashboard.selected_trainer_mode

    preview_state = agent.encode_state(game)
    while not game.quit_requested and not dashboard.started:
        events = pygame.event.get()
        scaled_events = game.scale_events(events)
        dashboard.sync_graph_rect(game)
        dashboard.handle_events(scaled_events)
        game.handle_system_events(events)

        if game.quit_requested:
            break

        selected_mode = dashboard.selected_trainer_mode
        if sync_agent_device_from_dashboard(agent, dashboard, "training lobby"):
            configure_agent_for_mode(agent, selected_mode)
        else:
            configure_agent_for_mode(agent, selected_mode)

        if dashboard.headless_toggle.value:
            dashboard.started = True
            break

        preview_fast_mode = bool(dashboard.fast_mode_toggle.value)
        dashboard.set_baseline_visibility(
            bool(comparison_mode and selected_mode == "single" and not preview_fast_mode)
        )
        preview_action = agent.get_action_details(preview_state, greedy=True)
        if selected_mode == "parallel":
            preview_mode_label = "Ready: parallel trainer"
        elif preview_fast_mode:
            preview_mode_label = "Ready: fast mode"
        else:
            preview_mode_label = "Ready: single trainer"
        preview_context = build_training_context(
            preview_mode_label,
            0.0,
            build_live_train_info(agent.last_train_info, session_perf, 0),
            last_transition,
            trainer_mode=selected_mode,
            parallel_envs=dashboard.get_parallel_env_count(),
            parallel_phase="prestart" if selected_mode == "parallel" else "single",
            eval_tail_episodes=eval_tail_episodes,
            fast_mode_requested=preview_fast_mode,
            fast_mode_effective=preview_fast_mode,
            fast_tail_episodes=fast_tail_episodes,
            episodes_remaining=dashboard.get_episode_goal(),
        )
        overlay_subtitle = (
            f"Parallel mode will batch {dashboard.get_parallel_env_count()} headless envs after you press Start [Enter]."
            if selected_mode == "parallel"
            else (
                f"Fast mode is armed. Press Start [Enter] to strip early episodes and animate the final {fast_tail_episodes}."
                if preview_fast_mode
                else "Click Start [Enter] to begin training."
            )
        )
        draw_dashboard_frame(
            game=game,
            dashboard=dashboard,
            agent=agent,
            state=preview_state,
            action_info=preview_action,
            current_game_number=1,
            episode_goal=dashboard.get_episode_goal(),
            best_score=best_score,
            context=preview_context,
            lightweight=False,
            show_baseline=dashboard.baseline_visible,
            recent_replays_panel=build_recent_replays_panel(
                dashboard,
                [],
                trainer_mode=selected_mode,
                fast_mode_requested=preview_fast_mode,
                fast_mode_effective=preview_fast_mode,
                fast_tail_episodes=fast_tail_episodes,
                eval_tail_episodes=eval_tail_episodes,
                parallel_phase="prestart",
                training_completed=False,
            ),
            overlay_title="Press Start",
            overlay_subtitle=overlay_subtitle,
        )
        pygame.time.delay(30)

    return dashboard.selected_trainer_mode


def train_parallel_mode(
    *,
    agent,
    game,
    dashboard,
    checkpoint_path,
    metrics_log_path,
    checkpoint_every,
    session_start_games,
    best_score,
    last_transition,
    recent_episode_replays,
    session_perf,
    parallel_envs,
    eval_tail_episodes,
):
    dashboard.set_baseline_visibility(False)
    pending_metric_entries = []
    parallel_envs = max(1, int(parallel_envs))
    eval_tail_episodes = max(1, int(eval_tail_episodes))
    envs = [SnakeLogicEnv(speed=0) for _ in range(parallel_envs)]
    reward_config = dict(dashboard.reward_config)
    for env in envs:
        env.set_reward_config(reward_config)
    env_states = agent.encode_states(envs)
    next_states_batch = np.empty_like(env_states)
    rewards_batch = np.empty(parallel_envs, dtype=np.float32)
    dones_batch = np.empty(parallel_envs, dtype=np.float32)
    episode_rewards = [0.0 for _ in envs]
    episode_steps = [0 for _ in envs]
    last_frame_time = 0.0
    bulk_iteration = 0
    completed_since_print = 0

    def completed_in_session():
        return max(0, agent.n_games - session_start_games)

    def should_run_eval_tail():
        return True

    def current_recent_replays_panel(*, training_completed=False, parallel_phase="bulk"):
        return build_recent_replays_panel(
            dashboard,
            recent_episode_replays,
            trainer_mode="parallel",
            fast_mode_requested=False,
            fast_mode_effective=False,
            fast_tail_episodes=1,
            eval_tail_episodes=eval_tail_episodes,
            parallel_phase=parallel_phase,
            training_completed=training_completed,
        )

    def current_context(
        mode_label,
        *,
        phase,
        episode_reward,
        episode_steps_value,
        transition_override=None,
    ):
        train_info = build_live_train_info(
            agent.last_train_info,
            session_perf,
            episode_steps_value,
        )
        total_goal = dashboard.get_episode_goal()
        return build_training_context(
            mode_label,
            episode_reward,
            train_info,
            transition_override if transition_override is not None else last_transition,
            trainer_mode="parallel",
            parallel_envs=parallel_envs,
            parallel_phase=phase,
            eval_tail_episodes=eval_tail_episodes,
            fast_mode_requested=False,
            fast_mode_effective=False,
            fast_tail_episodes=1,
            episodes_remaining=max(0, total_goal - completed_in_session()),
        )

    def save_parallel_checkpoint():
        save_checkpoint(
            agent,
            checkpoint_path,
            dashboard,
            metrics_log_path,
            last_transition,
            trainer_mode="parallel",
            parallel_envs=parallel_envs,
            eval_tail_episodes=eval_tail_episodes,
        )

    while not game.quit_requested:
        current_goal = dashboard.get_episode_goal()
        eval_count = min(eval_tail_episodes, current_goal) if should_run_eval_tail() else 0
        bulk_target = max(0, current_goal - eval_count)
        if completed_in_session() >= bulk_target:
            break

        bulk_iteration += 1
        events = pygame.event.get() if game.render else []
        scaled_events = game.scale_events(events) if game.render else []
        if game.render:
            dashboard.sync_graph_rect(game)
            dashboard.handle_events(scaled_events)
            game.handle_system_events(events)

        if game.quit_requested:
            break

        if sync_agent_device_from_dashboard(agent, dashboard, "parallel training"):
            configure_agent_for_mode(agent, "parallel")

        current_reward_config = dict(dashboard.reward_config)
        if current_reward_config != reward_config:
            reward_config = current_reward_config
            for env in envs:
                env.set_reward_config(reward_config)

        frame_due, last_frame_time = maybe_build_parallel_frame(
            render=game.render,
            dashboard=dashboard,
            last_frame_time=last_frame_time,
            force=False,
        )

        if dashboard.pause_toggle.value:
            if game.render and not dashboard.headless_toggle.value:
                representative_index = 0
                representative_env = envs[representative_index]
                copy_env_state_to_game(game, representative_env)
                preview_state = env_states[representative_index]
                preview_action = agent.get_action_details(preview_state, greedy=True)
                game.set_dashboard_data(
                    build_dashboard_frame(
                        game=game,
                        dashboard=dashboard,
                        agent=agent,
                        state=preview_state,
                        action_info=preview_action,
                        current_game_number=completed_in_session() + 1,
                        episode_goal=dashboard.get_episode_goal(),
                        best_score=best_score,
                        context=current_context(
                            "Parallel paused",
                            phase="bulk",
                            episode_reward=episode_rewards[representative_index],
                            episode_steps_value=episode_steps[representative_index],
                        ),
                        lightweight=False,
                        show_baseline=False,
                        recent_replays_panel=current_recent_replays_panel(parallel_phase="bulk"),
                    )
                )
                game.draw()
                pygame.time.delay(30)
            else:
                pygame.time.delay(5)
            continue

        if env_states.size == 0:
            break

        action_indices = agent.get_action_indices_batch(env_states, greedy=False)
        completed_records = []

        for env_index, env in enumerate(envs):
            reward, game_over, score = env.play_step(
                agent.index_to_action(int(action_indices[env_index])),
                events=[],
                draw_frame=False,
                apply_pacing=False,
            )
            episode_rewards[env_index] += reward
            episode_steps[env_index] += 1
            rewards_batch[env_index] = float(reward)
            dones_batch[env_index] = float(game_over)
            agent.encode_state(env, out=next_states_batch[env_index])
            if game_over:
                completed_records.append(
                    {
                        "env_index": env_index,
                        "score": int(score),
                        "episode_reward": float(episode_rewards[env_index]),
                        "steps": int(episode_steps[env_index]),
                        "final_reward": float(reward),
                    }
                )

        agent.remember_batch(
            env_states,
            action_indices,
            rewards_batch,
            next_states_batch,
            dones_batch,
        )
        env_states, next_states_batch = next_states_batch, env_states
        session_perf["env_steps"] += parallel_envs
        collect_diagnostics = bool(completed_records or frame_due or (bulk_iteration % 16 == 0))
        train_info = agent.train_step(
            collect_diagnostics=collect_diagnostics,
            num_new_transitions=parallel_envs,
        )
        session_perf["updates"] += int(train_info.get("update_count", 0))

        remaining_bulk_slots = max(0, bulk_target - completed_in_session())
        for record in completed_records[:remaining_bulk_slots]:
            current_run_number = completed_in_session() + 1
            agent.n_games += 1
            agent.decay_epsilon()
            dashboard.record_deep_episode(
                record["score"],
                record["episode_reward"],
                loss=agent.last_train_info.get("loss"),
            )
            best_score = max(best_score, record["score"])
            moving_avg = dashboard.deep_average_history[-1]
            last_transition = {
                "reward_text": f"{record['final_reward']:+.2f}",
                "done": True,
                "score": int(record["score"]),
            }
            pending_metric_entries.append(
                build_metric_entry(
                    algo="deep",
                    episode=agent.n_games,
                    score=record["score"],
                    episode_reward=record["episode_reward"],
                    moving_avg=moving_avg,
                    epsilon=agent.epsilon,
                    loss=agent.last_train_info.get("loss"),
                    steps=record["steps"],
                    buffer_size=len(agent.replay_buffer),
                    best_score=best_score,
                )
            )
            if len(pending_metric_entries) >= 8:
                flush_metric_buffer(metrics_log_path, pending_metric_entries)
            if agent.n_games % checkpoint_every == 0:
                save_parallel_checkpoint()

            completed_since_print += 1

            current_goal = dashboard.get_episode_goal()
            eval_count = min(eval_tail_episodes, current_goal) if should_run_eval_tail() else 0
            bulk_target = max(0, current_goal - eval_count)
            if completed_in_session() < bulk_target:
                env = envs[record["env_index"]]
                env.reset()
                env.set_reward_config(reward_config)
                agent.encode_state(env, out=env_states[record["env_index"]])
                episode_rewards[record["env_index"]] = 0.0
                episode_steps[record["env_index"]] = 0

        if completed_since_print >= 4 or (
            completed_records and completed_in_session() >= bulk_target
        ):
            perf_info = build_live_train_info(
                agent.last_train_info,
                session_perf,
                max(episode_steps) if episode_steps else 0,
            )
            print(
                f"Parallel progress {completed_in_session():>4}/{dashboard.get_episode_goal():<4} | "
                f"Best: {best_score:>2} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Env/s: {perf_info.get('env_steps_per_sec', 0.0):6.1f} | "
                f"Updates/s: {perf_info.get('updates_per_sec', 0.0):6.1f} | "
                f"Buffer: {len(agent.replay_buffer)} | "
                f"Loss: {agent.last_train_info.get('loss')} | "
                f"Parallel envs: {parallel_envs}"
            )
            completed_since_print = 0

        if game.render and not dashboard.headless_toggle.value and (frame_due or completed_records):
            representative_env = envs[0]
            representative_state = env_states[0]
            copy_env_state_to_game(game, representative_env)
            preview_action = agent.get_action_details(representative_state, greedy=True)
            game.set_dashboard_data(
                build_dashboard_frame(
                    game=game,
                    dashboard=dashboard,
                    agent=agent,
                    state=representative_state,
                    action_info=preview_action,
                    current_game_number=completed_in_session() + 1,
                    episode_goal=dashboard.get_episode_goal(),
                    best_score=best_score,
                    context=current_context(
                        "Parallel bulk",
                        phase="bulk",
                        episode_reward=episode_rewards[0],
                        episode_steps_value=episode_steps[0],
                    ),
                    lightweight=False,
                    show_baseline=False,
                    recent_replays_panel=current_recent_replays_panel(parallel_phase="bulk"),
                )
            )
            game.draw()
            if dashboard.current_delay_ms > 0 and not dashboard.turbo_toggle.value:
                pygame.time.delay(min(30, dashboard.current_delay_ms))

        switched_mode = apply_pending_trainer_mode_switch(agent, dashboard)
        if switched_mode:
            flush_metric_buffer(metrics_log_path, pending_metric_entries)
            return best_score, last_transition, False, switched_mode

    flush_metric_buffer(metrics_log_path, pending_metric_entries)

    if game.quit_requested:
        return best_score, last_transition, False, "parallel"

    remaining_goal = max(0, dashboard.get_episode_goal() - completed_in_session())
    eval_runs = min(eval_tail_episodes, remaining_goal) if should_run_eval_tail() else 0
    for _ in range(eval_runs):
        if game.quit_requested:
            break

        current_game_number = completed_in_session() + 1
        game.reset()
        game.set_reward_config(dashboard.reward_config)
        state = agent.encode_state(game)
        episode_reward = 0.0
        step_count = 0
        episode_replay_frames = []
        capture_replay = True

        while not game.quit_requested:
            events = pygame.event.get() if game.render else []
            scaled_events = game.scale_events(events) if game.render else []
            if game.render:
                dashboard.sync_graph_rect(game)
                dashboard.handle_events(scaled_events)
                game.handle_system_events(events)

            if game.quit_requested:
                break

            if sync_agent_device_from_dashboard(agent, dashboard, "parallel evaluation"):
                configure_agent_for_mode(agent, "parallel")
            game.speed = 0 if dashboard.headless_toggle.value else dashboard.current_fps

            if dashboard.pause_toggle.value:
                preview_action = agent.get_action_details(state, greedy=True)
                game.set_dashboard_data(
                    build_dashboard_frame(
                        game=game,
                        dashboard=dashboard,
                        agent=agent,
                        state=state,
                        action_info=preview_action,
                        current_game_number=current_game_number,
                        episode_goal=dashboard.get_episode_goal(),
                        best_score=best_score,
                        context=current_context(
                            "Parallel eval",
                            phase="eval",
                            episode_reward=episode_reward,
                            episode_steps_value=step_count,
                        ),
                        lightweight=False,
                        show_baseline=False,
                        recent_replays_panel=current_recent_replays_panel(parallel_phase="eval"),
                    )
                )
                game.draw()
                pygame.time.delay(30)
                continue

            action_info = agent.get_action_details(state, greedy=True)
            step_count += 1
            render_frame = bool(game.render and not dashboard.headless_toggle.value)
            replay_frame = None
            replay_context = current_context(
                "Parallel eval",
                phase="eval",
                episode_reward=episode_reward,
                episode_steps_value=step_count,
            )
            if render_frame or capture_replay:
                frame_data = build_dashboard_frame(
                    game=game,
                    dashboard=dashboard,
                    agent=agent,
                    state=state,
                    action_info=action_info,
                    current_game_number=current_game_number,
                    episode_goal=dashboard.get_episode_goal(),
                    best_score=best_score,
                    context=replay_context,
                    lightweight=False,
                    show_baseline=False,
                    recent_replays_panel=current_recent_replays_panel(parallel_phase="eval"),
                )
                if capture_replay:
                    replay_frame = capture_replay_frame(
                        game,
                        state=state,
                        action_info=action_info,
                        current_game_number=current_game_number,
                        episode_goal=dashboard.get_episode_goal(),
                        best_score=best_score,
                        context=replay_context,
                        show_baseline=False,
                    )
                    episode_replay_frames.append(replay_frame)
                if render_frame:
                    game.set_dashboard_data(frame_data)
                    if dashboard.should_draw_frame(step_count, force=True):
                        game.draw()
                        if dashboard.current_delay_ms > 0:
                            pygame.time.delay(dashboard.current_delay_ms)

            reward, game_over, score = game.play_step(
                action_info["action"],
                events=[],
                draw_frame=False,
                apply_pacing=not dashboard.headless_toggle.value,
            )
            episode_reward += reward
            state = agent.encode_state(game)
            last_transition = {
                "reward_text": f"{reward:+.2f}",
                "done": game_over,
                "score": int(score),
            }

            if render_frame and game_over:
                final_context = current_context(
                    "Parallel eval",
                    phase="eval",
                    episode_reward=episode_reward,
                    episode_steps_value=step_count,
                    transition_override=last_transition,
                )
                final_view = build_dashboard_frame(
                    game=game,
                    dashboard=dashboard,
                    agent=agent,
                    state=state,
                    action_info=action_info,
                    current_game_number=current_game_number,
                    episode_goal=dashboard.get_episode_goal(),
                    best_score=best_score,
                    context=final_context,
                    lightweight=False,
                    show_baseline=False,
                    recent_replays_panel=current_recent_replays_panel(parallel_phase="eval"),
                )
                final_view["overlay_title"] = "Evaluation run finished"
                final_view["overlay_subtitle"] = f"Score: {score}"
                game.set_dashboard_data(final_view)
                if capture_replay:
                    episode_replay_frames.append(
                        capture_replay_frame(
                            game,
                            state=state,
                            action_info=action_info,
                            current_game_number=current_game_number,
                            episode_goal=dashboard.get_episode_goal(),
                            best_score=best_score,
                            context=final_context,
                            show_baseline=False,
                            overlay_title="Evaluation run finished",
                            overlay_subtitle=f"Score: {score}",
                        )
                    )
                game.draw()
                pygame.time.delay(120 if dashboard.turbo_toggle.value else max(140, dashboard.current_delay_ms))
            elif game_over and capture_replay:
                final_context = current_context(
                    "Parallel eval",
                    phase="eval",
                    episode_reward=episode_reward,
                    episode_steps_value=step_count,
                    transition_override=last_transition,
                )
                episode_replay_frames.append(
                    capture_replay_frame(
                        game,
                        state=state,
                        action_info=action_info,
                        current_game_number=current_game_number,
                        episode_goal=dashboard.get_episode_goal(),
                        best_score=best_score,
                        context=final_context,
                        show_baseline=False,
                        overlay_title="Evaluation run finished",
                        overlay_subtitle=f"Score: {score}",
                    )
                )

            if game_over:
                break

        if game.quit_requested:
            break

        agent.n_games += 1
        dashboard.record_deep_episode(
            score,
            episode_reward,
            loss=agent.last_train_info.get("loss"),
        )
        best_score = max(best_score, score)
        if capture_replay and episode_replay_frames:
            recent_episode_replays.append(
                {
                    "run_number": current_game_number,
                    "score": int(score),
                    "frames": episode_replay_frames,
                }
            )

        moving_avg = dashboard.deep_average_history[-1]
        pending_metric_entries.append(
            build_metric_entry(
                algo="deep",
                episode=agent.n_games,
                score=score,
                episode_reward=episode_reward,
                moving_avg=moving_avg,
                epsilon=agent.epsilon,
                loss=agent.last_train_info.get("loss"),
                steps=step_count,
                buffer_size=len(agent.replay_buffer),
                best_score=best_score,
            )
        )
        if len(pending_metric_entries) >= 8:
            flush_metric_buffer(metrics_log_path, pending_metric_entries)
        if agent.n_games % checkpoint_every == 0:
            save_parallel_checkpoint()

        perf_info = build_live_train_info(agent.last_train_info, session_perf, step_count)
        print(
            f"Parallel eval {current_game_number:>4}/{dashboard.get_episode_goal():<4} | "
            f"Total games: {agent.n_games:>4} | "
            f"Score: {score:>2} | "
            f"Best: {best_score:>2} | "
            f"Epsilon: {agent.epsilon:.3f} | "
            f"Episode steps: {step_count:>4} | "
            f"Env/s: {perf_info.get('env_steps_per_sec', 0.0):6.1f} | "
            f"Updates/s: {perf_info.get('updates_per_sec', 0.0):6.1f} | "
            f"Buffer: {len(agent.replay_buffer)} | "
            f"Loss: {agent.last_train_info.get('loss')} | "
            f"Mode: parallel-eval"
        )

        switched_mode = apply_pending_trainer_mode_switch(agent, dashboard)
        if switched_mode:
            flush_metric_buffer(metrics_log_path, pending_metric_entries)
            return best_score, last_transition, False, switched_mode

    flush_metric_buffer(metrics_log_path, pending_metric_entries)
    return best_score, last_transition, not game.quit_requested, "parallel"


def train_session(
    episodes,
    render,
    speed,
    delay_ms,
    checkpoint_path,
    metrics_log_path,
    resume,
    checkpoint_every,
    comparison_mode,
    baseline_episodes,
    hidden_layers,
    device_preference="auto",
    fast_mode=False,
    fast_tail_episodes=5,
    trainer_mode=None,
    parallel_envs=None,
    eval_tail_episodes=None,
):
    histories = prepare_metrics_log(metrics_log_path, resume)
    deep_history = histories.get("deep", {})
    baseline_history = histories.get("tabular", {})

    agent = DQNAgent(hidden_layers=hidden_layers, device_preference=device_preference)
    checkpoint_state = {}
    if resume and os.path.exists(checkpoint_path):
        checkpoint_state = agent.load(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")

    resolved_trainer_mode = resolve_trainer_mode_for_session(
        checkpoint_state,
        resume,
        trainer_mode,
    )
    resolved_parallel_envs = resolve_positive_session_int(
        checkpoint_state,
        resume,
        parallel_envs,
        "parallel_envs",
        resolve_parallel_env_count(agent.device.type, None),
    )
    resolved_eval_tail_episodes = resolve_positive_session_int(
        checkpoint_state,
        resume,
        eval_tail_episodes,
        "eval_tail_episodes",
        3,
    )
    initial_reward_config = checkpoint_state.get("reward_config", DEFAULT_REWARD_CONFIG)
    deep_history = checkpoint_state.get("deep_history", deep_history)

    game = SnakeGameAI(w=640, h=700, window_h=880, render=render, speed=speed)
    dashboard = TrainingDashboard(
        game,
        initial_speed=speed,
        initial_delay_ms=delay_ms,
        initial_episode_goal=episodes,
        initial_reward_config=initial_reward_config,
        initial_headless=not render,
        initial_device_preference=agent.device.type,
        cuda_available=cuda_is_available(),
        require_manual_start=render,
        initial_fast_mode=fast_mode,
        initial_trainer_mode=resolved_trainer_mode,
        initial_parallel_envs=resolved_parallel_envs,
    )
    dashboard.load_deep_history(deep_history)
    dashboard.set_baseline_visibility(
        bool(comparison_mode and not fast_mode and resolved_trainer_mode == "single")
    )
    if comparison_mode and not fast_mode and resolved_trainer_mode == "single" and baseline_history:
        dashboard.set_baseline_history(baseline_history)

    best_score = max(dashboard.deep_best_history) if dashboard.deep_best_history else 0
    session_start_games = agent.n_games
    training_completed = False
    last_transition = checkpoint_state.get("last_transition", {"reward_text": "n/a"})
    fast_tail_episodes = max(1, int(fast_tail_episodes))
    recent_episode_replays = deque(maxlen=3)
    active_trainer_mode = resolved_trainer_mode
    session_perf = {
        "start_time": time.perf_counter(),
        "env_steps": 0,
        "updates": 0,
    }

    try:
        active_trainer_mode = await_training_start(
            game=game,
            dashboard=dashboard,
            agent=agent,
            comparison_mode=comparison_mode,
            best_score=best_score,
            last_transition=last_transition,
            session_perf=session_perf,
            parallel_envs=resolved_parallel_envs,
            eval_tail_episodes=resolved_eval_tail_episodes,
            fast_tail_episodes=fast_tail_episodes,
        )
        if game.quit_requested:
            training_completed = False
        else:
            dashboard.selected_trainer_mode = active_trainer_mode
            resolved_parallel_envs = dashboard.get_parallel_env_count()
            configure_agent_for_mode(agent, active_trainer_mode)

        while not game.quit_requested:
            current_goal = dashboard.get_episode_goal()
            completed_in_session = agent.n_games - session_start_games
            if completed_in_session >= current_goal:
                training_completed = True
                break

            if active_trainer_mode == "parallel":
                best_score, last_transition, phase_completed, next_mode = train_parallel_mode(
                    agent=agent,
                    game=game,
                    dashboard=dashboard,
                    checkpoint_path=checkpoint_path,
                    metrics_log_path=metrics_log_path,
                    checkpoint_every=checkpoint_every,
                    session_start_games=session_start_games,
                    best_score=best_score,
                    last_transition=last_transition,
                    recent_episode_replays=recent_episode_replays,
                    session_perf=session_perf,
                    parallel_envs=resolved_parallel_envs,
                    eval_tail_episodes=resolved_eval_tail_episodes,
                )
                if game.quit_requested:
                    training_completed = False
                    break
                if phase_completed:
                    training_completed = True
                    break
                active_trainer_mode = next_mode
                dashboard.selected_trainer_mode = next_mode
                if next_mode == "parallel":
                    resolved_parallel_envs = dashboard.get_parallel_env_count()
                continue

            episode_fast_mode = bool(dashboard.fast_mode_toggle.value)
            baseline_history = ensure_baseline_history(
                dashboard,
                comparison_mode=comparison_mode,
                fast_mode_effective=episode_fast_mode,
                baseline_history=baseline_history,
                baseline_episodes=baseline_episodes,
                metrics_log_path=metrics_log_path,
            )
            episode_show_baseline = bool(dashboard.baseline_visible)
            episodes_remaining = max(0, current_goal - completed_in_session)
            current_game_number = completed_in_session + 1
            stripped_episode = bool(
                episode_fast_mode
                and (
                    not render
                    or dashboard.headless_toggle.value
                    or episodes_remaining > fast_tail_episodes
                )
            )
            animated_tail_episode = bool(episode_fast_mode and not stripped_episode)
            replay_capture_episode = episodes_remaining <= 3

            game.reset()
            state = agent.encode_state(game)
            step_count = 0
            episode_reward = 0.0
            episode_replay_frames = []
            capture_episode_replay = bool(replay_capture_episode)

            def current_recent_replays_panel(training_completed=False):
                return build_recent_replays_panel(
                    dashboard,
                    recent_episode_replays,
                    trainer_mode=active_trainer_mode,
                    fast_mode_requested=dashboard.fast_mode_toggle.value,
                    fast_mode_effective=episode_fast_mode,
                    fast_tail_episodes=fast_tail_episodes,
                    eval_tail_episodes=resolved_eval_tail_episodes,
                    parallel_phase="single",
                    training_completed=training_completed,
                )

            def current_context(
                mode_label,
                *,
                train_info_override=None,
                episode_steps_override=None,
                remaining_override=None,
            ):
                train_info = build_live_train_info(
                    train_info_override if train_info_override is not None else agent.last_train_info,
                    session_perf,
                    step_count if episode_steps_override is None else episode_steps_override,
                )
                return build_training_context(
                    mode_label,
                    episode_reward,
                    train_info,
                    last_transition,
                    trainer_mode=active_trainer_mode,
                    parallel_envs=1,
                    parallel_phase="single",
                    eval_tail_episodes=resolved_eval_tail_episodes,
                    fast_mode_requested=dashboard.fast_mode_toggle.value,
                    fast_mode_effective=episode_fast_mode,
                    fast_tail_episodes=fast_tail_episodes,
                    episodes_remaining=(
                        episodes_remaining
                        if remaining_override is None
                        else remaining_override
                    ),
                )

            while True:
                episode_goal = dashboard.get_episode_goal()
                events = pygame.event.get() if render else []
                scaled_events = game.scale_events(events) if render else []
                dashboard.sync_graph_rect(game)
                headless_before_events = dashboard.headless_toggle.value
                dashboard.handle_events(scaled_events)
                game.handle_system_events(events)
                headless_changed = dashboard.headless_toggle.value != headless_before_events

                if game.quit_requested:
                    break

                sync_agent_device_from_dashboard(agent, dashboard, "training")
                game.speed = 0 if dashboard.headless_toggle.value else dashboard.current_fps
                game.set_reward_config(dashboard.reward_config)

                if render and headless_changed:
                    preview_info = agent.get_action_details(state, greedy=True)
                    draw_dashboard_frame(
                        game=game,
                        dashboard=dashboard,
                        agent=agent,
                        state=state,
                        action_info=preview_info,
                        current_game_number=current_game_number,
                        episode_goal=episode_goal,
                        best_score=best_score,
                        context=current_context("Training"),
                        lightweight=stripped_episode,
                        show_baseline=episode_show_baseline,
                        recent_replays_panel=current_recent_replays_panel(),
                    )

                if not dashboard.started:
                    if dashboard.headless_toggle.value:
                        dashboard.started = True
                        continue

                    preview_info = agent.get_action_details(state, greedy=True)
                    draw_dashboard_frame(
                        game=game,
                        dashboard=dashboard,
                        agent=agent,
                        state=state,
                        action_info=preview_info,
                        current_game_number=current_game_number,
                        episode_goal=episode_goal,
                        best_score=best_score,
                        context=current_context("Ready to start"),
                        lightweight=False,
                        show_baseline=episode_show_baseline,
                        recent_replays_panel=current_recent_replays_panel(),
                        overlay_title="Press Start",
                        overlay_subtitle="Click Start [Enter] to begin training.",
                    )
                    pygame.time.delay(30)
                    continue

                if dashboard.pause_toggle.value:
                    preview_info = agent.get_action_details(state, greedy=True)
                    game.set_dashboard_data(
                        build_dashboard_frame(
                            game=game,
                            dashboard=dashboard,
                            agent=agent,
                            state=state,
                            action_info=preview_info,
                            current_game_number=current_game_number,
                            episode_goal=episode_goal,
                            best_score=best_score,
                            context=current_context("Paused"),
                            lightweight=stripped_episode,
                            show_baseline=episode_show_baseline,
                            recent_replays_panel=current_recent_replays_panel(),
                        )
                    )
                    if render:
                        game.draw()
                        pygame.time.delay(30)
                    else:
                        pygame.time.delay(5)
                    continue

                action_info = agent.get_action_selection(
                    state,
                    lightweight=stripped_episode,
                )
                step_count += 1
                live_render_enabled = (
                    render and not dashboard.headless_toggle.value and not stripped_episode
                )
                live_context = current_context(
                    "Fast tail" if animated_tail_episode else "Training"
                )

                if live_render_enabled or capture_episode_replay:
                    frame_data = build_dashboard_frame(
                        game=game,
                        dashboard=dashboard,
                        agent=agent,
                        state=state,
                        action_info=action_info,
                        current_game_number=current_game_number,
                        episode_goal=episode_goal,
                        best_score=best_score,
                        context=live_context,
                        lightweight=False,
                        show_baseline=episode_show_baseline,
                        recent_replays_panel=current_recent_replays_panel(),
                    )
                    if capture_episode_replay:
                        episode_replay_frames.append(
                            capture_replay_frame(
                                game,
                                state=state,
                                action_info=action_info,
                                current_game_number=current_game_number,
                                episode_goal=episode_goal,
                                best_score=best_score,
                                context=live_context,
                                show_baseline=episode_show_baseline,
                            )
                        )
                    if live_render_enabled:
                        game.set_dashboard_data(frame_data)
                        if dashboard.should_draw_frame(step_count):
                            game.draw()
                            if dashboard.current_delay_ms > 0:
                                pygame.time.delay(dashboard.current_delay_ms)

                reward, game_over, score = game.play_step(
                    action_info["action"],
                    events=[],
                    draw_frame=False,
                    apply_pacing=not dashboard.headless_toggle.value and not stripped_episode,
                )
                episode_reward += reward
                next_state = agent.encode_state(game)
                agent.remember(
                    state=state,
                    action_index=action_info["action_index"],
                    reward=reward,
                    next_state=next_state,
                    done=game_over,
                )
                session_perf["env_steps"] += 1
                collect_diagnostics = bool(
                    not stripped_episode or game_over or (step_count % 64 == 0)
                )
                train_info = agent.train_step(collect_diagnostics=collect_diagnostics)
                if train_info.get("did_update"):
                    session_perf["updates"] += 1
                state = next_state
                last_transition = {
                    "reward_text": f"{reward:+.2f}",
                    "done": game_over,
                    "score": score,
                }

                if render and game_over and not dashboard.headless_toggle.value and not stripped_episode:
                    final_context = current_context(
                        "Fast tail" if animated_tail_episode else "Training",
                        train_info_override=train_info,
                    )
                    final_view = build_dashboard_frame(
                        game=game,
                        dashboard=dashboard,
                        agent=agent,
                        state=state,
                        action_info=action_info,
                        current_game_number=current_game_number,
                        episode_goal=episode_goal,
                        best_score=best_score,
                        context=final_context,
                        lightweight=False,
                        show_baseline=episode_show_baseline,
                        recent_replays_panel=current_recent_replays_panel(),
                    )
                    final_view["overlay_title"] = "Episode finished"
                    final_view["overlay_subtitle"] = f"Reward: {reward:+.2f}"
                    game.set_dashboard_data(final_view)
                    if capture_episode_replay:
                        episode_replay_frames.append(
                            capture_replay_frame(
                                game,
                                state=state,
                                action_info=action_info,
                                current_game_number=current_game_number,
                                episode_goal=episode_goal,
                                best_score=best_score,
                                context=final_context,
                                show_baseline=episode_show_baseline,
                                overlay_title="Episode finished",
                                overlay_subtitle=f"Reward: {reward:+.2f}",
                            )
                        )
                    game.draw()
                    pygame.time.delay(
                        120 if dashboard.turbo_toggle.value else max(140, dashboard.current_delay_ms)
                    )
                elif game_over and capture_episode_replay:
                    final_context = current_context(
                        "Fast tail" if animated_tail_episode else "Training",
                        train_info_override=train_info,
                    )
                    episode_replay_frames.append(
                        capture_replay_frame(
                            game,
                            state=state,
                            action_info=action_info,
                            current_game_number=current_game_number,
                            episode_goal=episode_goal,
                            best_score=best_score,
                            context=final_context,
                            show_baseline=episode_show_baseline,
                            overlay_title="Episode finished",
                            overlay_subtitle=f"Reward: {reward:+.2f}",
                        )
                    )

                if game.quit_requested or game_over:
                    break

            if game.quit_requested:
                print("Training stopped because the game window was closed.")
                break

            agent.n_games += 1
            agent.decay_epsilon()
            dashboard.record_deep_episode(
                score,
                episode_reward,
                loss=agent.last_train_info.get("loss"),
            )
            best_score = max(best_score, score)
            if capture_episode_replay and episode_replay_frames:
                recent_episode_replays.append(
                    {
                        "run_number": current_game_number,
                        "score": int(score),
                        "frames": episode_replay_frames,
                    }
                )

            moving_avg = dashboard.deep_average_history[-1]
            metric_entry = build_metric_entry(
                algo="deep",
                episode=agent.n_games,
                score=score,
                episode_reward=episode_reward,
                moving_avg=moving_avg,
                epsilon=agent.epsilon,
                loss=agent.last_train_info.get("loss"),
                steps=step_count,
                buffer_size=len(agent.replay_buffer),
                best_score=best_score,
            )
            append_metric_entry(metrics_log_path, metric_entry)

            if render and (dashboard.headless_toggle.value or stripped_episode):
                summary_action = agent.get_action_details(state, greedy=True)
                draw_dashboard_frame(
                    game=game,
                    dashboard=dashboard,
                    agent=agent,
                    state=state,
                    action_info=summary_action,
                    current_game_number=current_game_number,
                    episode_goal=dashboard.get_episode_goal(),
                    best_score=best_score,
                    context=current_context(
                        "Fast pass" if stripped_episode else "Headless training",
                        remaining_override=max(
                            0,
                            dashboard.get_episode_goal() - (agent.n_games - session_start_games),
                        ),
                    ),
                    lightweight=stripped_episode,
                    show_baseline=episode_show_baseline,
                    recent_replays_panel=current_recent_replays_panel(),
                )

            if agent.n_games % checkpoint_every == 0 or score == best_score:
                save_checkpoint(
                    agent,
                    checkpoint_path,
                    dashboard,
                    metrics_log_path,
                    last_transition,
                    trainer_mode=active_trainer_mode,
                    parallel_envs=resolved_parallel_envs,
                    eval_tail_episodes=resolved_eval_tail_episodes,
                )

            perf_info = build_live_train_info(agent.last_train_info, session_perf, step_count)
            print(
                f"Run {current_game_number:>4}/{dashboard.get_episode_goal():<4} | "
                f"Total games: {agent.n_games:>4} | "
                f"Score: {score:>2} | "
                f"Best: {best_score:>2} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Episode steps: {step_count:>4} | "
                f"Env/s: {perf_info.get('env_steps_per_sec', 0.0):6.1f} | "
                f"Updates/s: {perf_info.get('updates_per_sec', 0.0):6.1f} | "
                f"Buffer: {len(agent.replay_buffer)} | "
                f"Loss: {agent.last_train_info.get('loss')} | "
                f"Mode: {'fast' if episode_fast_mode else 'default'}"
            )
            switched_mode = apply_pending_trainer_mode_switch(agent, dashboard)
            if switched_mode:
                active_trainer_mode = switched_mode
                if switched_mode == "parallel":
                    resolved_parallel_envs = dashboard.get_parallel_env_count()
                continue

        if active_trainer_mode == "single" and not game.quit_requested:
            training_completed = True
    finally:
        save_checkpoint(
            agent,
            checkpoint_path,
            dashboard,
            metrics_log_path,
            last_transition,
            trainer_mode=active_trainer_mode,
            parallel_envs=resolved_parallel_envs,
            eval_tail_episodes=resolved_eval_tail_episodes,
        )
        if training_completed and not game.quit_requested:
            show_post_run_results(
                game=game,
                dashboard=dashboard,
                agent=agent,
                recent_episode_replays=recent_episode_replays,
                best_score=best_score,
                last_transition=last_transition,
                session_perf=session_perf,
                trainer_mode=active_trainer_mode,
                parallel_envs=resolved_parallel_envs,
                eval_tail_episodes=resolved_eval_tail_episodes,
                fast_tail_episodes=fast_tail_episodes,
            )
        game.close()

    print(f"Training finished. Checkpoint saved to {checkpoint_path}")


def run_visualizer_session(checkpoint_path, metrics_log_path, speed, device_preference="auto"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_config = load_checkpoint_network_config(checkpoint_path)
    agent = DQNAgent(
        hidden_layers=checkpoint_config["hidden_layers"],
        device_preference=device_preference,
    )
    checkpoint_state = agent.load(checkpoint_path)
    resolved_metrics_log = metrics_log_path or checkpoint_state.get("metrics_log_path")
    histories = load_histories_from_log(resolved_metrics_log)
    deep_history = checkpoint_state.get("deep_history", histories.get("deep", {}))
    baseline_history = histories.get("tabular", {})

    game = SnakeGameAI(w=640, h=700, window_h=880, render=True, speed=speed)
    dashboard = TrainingDashboard(
        game,
        initial_speed=speed,
        initial_delay_ms=40,
        initial_episode_goal=max(1, len(deep_history.get("scores", []))),
        initial_reward_config=checkpoint_state.get("reward_config", DEFAULT_REWARD_CONFIG),
        initial_headless=False,
        initial_device_preference=agent.device.type,
        cuda_available=cuda_is_available(),
        require_manual_start=False,
        initial_trainer_mode=checkpoint_state.get("trainer_config", {}).get(
            "mode",
            DEFAULT_TRAINER_MODE,
        ),
        initial_parallel_envs=checkpoint_state.get("trainer_config", {}).get(
            "parallel_envs",
            8,
        ),
    )
    dashboard.load_deep_history(deep_history)
    dashboard.set_baseline_visibility(bool(baseline_history))
    if baseline_history:
        dashboard.set_baseline_history(baseline_history)

    best_score = max(dashboard.deep_best_history) if dashboard.deep_best_history else 0
    last_transition = checkpoint_state.get("last_transition", {"reward_text": "n/a"})

    try:
        viewer_episode = 0
        while not game.quit_requested:
            viewer_episode += 1
            game.reset()
            state = agent.encode_state(game)
            episode_reward = 0.0
            step_count = 0

            while not game.quit_requested:
                events = pygame.event.get()
                scaled_events = game.scale_events(events)
                dashboard.sync_graph_rect(game)
                headless_before_events = dashboard.headless_toggle.value
                dashboard.handle_events(scaled_events)
                game.handle_system_events(events)
                headless_changed = dashboard.headless_toggle.value != headless_before_events

                if game.quit_requested:
                    break

                sync_agent_device_from_dashboard(agent, dashboard, "viewer")
                game.speed = 0 if dashboard.headless_toggle.value else dashboard.current_fps
                action_info = agent.get_action_details(state, greedy=True)
                context = build_training_context(
                    "Viewer",
                    episode_reward,
                    agent.last_train_info,
                    last_transition,
                )

                game.set_dashboard_data(
                    dashboard.build_dashboard_data(
                        agent=agent,
                        game=game,
                        state=state,
                        action_info=action_info,
                        current_game_number=viewer_episode,
                        episode_goal=viewer_episode,
                        best_score=best_score,
                        context=context,
                        show_baseline=dashboard.baseline_visible,
                    )
                )

                if headless_changed:
                    game.draw()

                if dashboard.pause_toggle.value:
                    game.draw()
                    pygame.time.delay(30)
                    continue

                step_count += 1
                if dashboard.should_draw_frame(step_count, force=True):
                    game.draw()
                    if dashboard.current_delay_ms > 0:
                        pygame.time.delay(dashboard.current_delay_ms)

                reward, game_over, score = game.play_step(
                    action_info["action"],
                    events=[],
                    draw_frame=False,
                )
                episode_reward += reward
                state = agent.encode_state(game)
                last_transition = {
                    "reward_text": f"{reward:+.2f}",
                    "done": game_over,
                    "score": score,
                }

                if game_over:
                    best_score = max(best_score, score)
                    if not dashboard.headless_toggle.value:
                        final_view = dashboard.build_dashboard_data(
                            agent=agent,
                            game=game,
                            state=state,
                            action_info=action_info,
                            current_game_number=viewer_episode,
                            episode_goal=viewer_episode,
                            best_score=best_score,
                            context=context,
                            show_baseline=dashboard.baseline_visible,
                        )
                        final_view["overlay_title"] = "Viewer episode finished"
                        final_view["overlay_subtitle"] = f"Score: {score}"
                        game.set_dashboard_data(final_view)
                        game.draw()
                        pygame.time.delay(250)
                    break
    finally:
        game.close()


def main_train():
    parser = argparse.ArgumentParser(
        description="Train a Snake DQN with the teaching dashboard or a stripped fast mode."
    )
    parser.add_argument("--episodes", type=int, default=300, help="How many games to play this session.")
    parser.add_argument("--speed", type=int, default=16, help="Initial render speed.")
    parser.add_argument("--delay-ms", type=int, default=60, help="Initial frame delay.")
    parser.add_argument("--checkpoint-path", default="dqn_checkpoint.pt", help="Checkpoint file path.")
    parser.add_argument("--metrics-log", default="training_metrics.jsonl", help="JSONL metric log path.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing checkpoint.")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Save every N finished games.")
    parser.add_argument("--no-render", action="store_true", help="Run without the pygame window.")
    parser.add_argument("--comparison-mode", action="store_true", help="Overlay a local tabular baseline.")
    parser.add_argument("--baseline-episodes", type=int, default=120, help="How many baseline episodes to precompute.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device. auto uses CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=parse_hidden_layers_arg,
        default=None,
        help="Comma-separated hidden-layer sizes, for example 64 or 128,128,64.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Strip most visualization overhead and animate only the final tail episodes.",
    )
    parser.add_argument(
        "--fast-tail-episodes",
        type=parse_positive_int_arg,
        default=5,
        help="How many ending episodes to animate in fast mode.",
    )
    parser.add_argument(
        "--trainer-mode",
        type=parse_trainer_mode_arg,
        default=None,
        help="Trainer mode. single keeps the teaching workflow, parallel batches headless workers for speed.",
    )
    parser.add_argument(
        "--parallel-envs",
        type=parse_positive_int_arg,
        default=None,
        help="How many headless Snake environments to step in parallel when --trainer-mode parallel is used.",
    )
    parser.add_argument(
        "--eval-tail-episodes",
        type=parse_positive_int_arg,
        default=None,
        help="How many rendered evaluation episodes to show at the end of a parallel-mode session.",
    )

    args = parser.parse_args()
    render = not args.no_render
    speed = args.speed if render else 0
    delay_ms = args.delay_ms if render else 0
    hidden_layers = resolve_hidden_layers_for_session(
        checkpoint_path=args.checkpoint_path,
        resume=args.resume,
        explicit_hidden_layers=args.hidden_layers,
    )

    try:
        train_session(
            episodes=args.episodes,
            render=render,
            speed=speed,
            delay_ms=delay_ms,
            checkpoint_path=args.checkpoint_path,
            metrics_log_path=args.metrics_log,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            comparison_mode=args.comparison_mode,
            baseline_episodes=args.baseline_episodes,
            hidden_layers=hidden_layers,
            device_preference=args.device,
            fast_mode=args.fast_mode,
            fast_tail_episodes=args.fast_tail_episodes,
            trainer_mode=args.trainer_mode,
            parallel_envs=args.parallel_envs,
            eval_tail_episodes=args.eval_tail_episodes,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def main_visualizer():
    parser = argparse.ArgumentParser(
        description="View a saved Snake DQN checkpoint with the pygame dashboard."
    )
    parser.add_argument("--checkpoint", default="dqn_checkpoint.pt", help="Checkpoint file path.")
    parser.add_argument("--metrics-log", default=None, help="Optional JSONL metric log path.")
    parser.add_argument("--speed", type=int, default=16, help="Playback speed.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Viewer device. auto uses CUDA when available, otherwise CPU.",
    )
    args = parser.parse_args()

    try:
        run_visualizer_session(
            checkpoint_path=args.checkpoint,
            metrics_log_path=args.metrics_log,
            speed=args.speed,
            device_preference=args.device,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
