from __future__ import annotations

import argparse
import copy
import os
from collections import deque

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
    build_history,
    group_entries_by_algo,
    load_histories_from_log,
    load_metric_entries,
    rewrite_metric_entries,
)
from snake_game import SnakeGameAI
from tabular_agent import QLearningAgent


DEFAULT_REWARD_CONFIG = {"food": 10.0, "death": -10.0, "step": 0.0}


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


def hold_training_window_open(game, dashboard, recent_episode_replays=None):
    pygame.event.clear()
    recent_episode_replays = list(recent_episode_replays or [])
    game.set_dashboard_data(
        build_training_finished_view(game.dashboard_data, game, recent_episode_replays)
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
                for button in game.dashboard_data.get("overlay_buttons", []):
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
            play_episode_replay(game, recent_episode_replays[replay_index])
            if game.quit_requested:
                return
            game.set_dashboard_data(
                build_training_finished_view(game.dashboard_data, game, recent_episode_replays)
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
    game = SnakeGameAI(render=False, speed=0)
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


def save_checkpoint(agent, checkpoint_path, dashboard, metrics_log_path, last_transition):
    extra_state = {
        "reward_config": dashboard.reward_config,
        "deep_history": {
            "scores": list(dashboard.deep_scores),
            "moving_avg": list(dashboard.deep_average_history),
            "best_scores": list(dashboard.deep_best_history),
            "episode_rewards": list(dashboard.deep_episode_rewards),
        },
        "metrics_log_path": metrics_log_path,
        "last_transition": dict(last_transition or {}),
    }
    agent.save(checkpoint_path, extra_state=extra_state)


def capture_replay_frame(game, frame_data):
    return {
        "snake": list(game.snake),
        "head": game.head,
        "food": game.food,
        "direction": game.direction,
        "score": int(game.score),
        "frame_iteration": int(game.frame_iteration),
        "dashboard_data": copy.deepcopy(frame_data),
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


def build_training_finished_view(base_view, game, recent_episode_replays):
    final_view = dict(base_view or {})
    final_view["overlay_title"] = "Training finished"
    if recent_episode_replays:
        final_view["overlay_subtitle"] = (
            "Replay one of the last 3 fast-mode runs, or press Enter, Q, or Esc to close."
        )
        final_view["overlay_buttons"] = build_replay_overlay_buttons(game, recent_episode_replays)
    else:
        final_view["overlay_subtitle"] = "Press Enter, Q, Esc, or close the window."
        final_view["overlay_buttons"] = []
    return final_view


def apply_replay_frame(game, frame):
    game.snake = list(frame.get("snake", []))
    game.head = frame.get("head", game.head)
    game.food = frame.get("food", game.food)
    game.direction = frame.get("direction", game.direction)
    game.score = int(frame.get("score", game.score))
    game.frame_iteration = int(frame.get("frame_iteration", game.frame_iteration))
    game.set_dashboard_data(copy.deepcopy(frame.get("dashboard_data", {})))


def play_episode_replay(game, episode_replay):
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

        apply_replay_frame(game, frame)
        game.draw()
        pygame.time.delay(55)


def build_training_context(
    mode_label,
    episode_reward,
    train_info,
    transition,
    *,
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
        "fast_mode_requested": bool(fast_mode_requested),
        "fast_mode_effective": bool(fast_mode_effective),
        "fast_tail_episodes": int(fast_tail_episodes),
        "episodes_remaining": int(max(0, episodes_remaining)),
    }


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
    if overlay_title:
        frame["overlay_title"] = overlay_title
    if overlay_subtitle:
        frame["overlay_subtitle"] = overlay_subtitle
    game.set_dashboard_data(frame)
    game.draw()


def sync_agent_device_from_dashboard(agent, dashboard, session_label):
    if dashboard.selected_device_preference != agent.device_preference:
        if agent.set_device(dashboard.selected_device_preference):
            print(f"Switched {session_label} device to {agent.device_label} ({agent.device})")


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
):
    histories = prepare_metrics_log(metrics_log_path, resume)
    deep_history = histories.get("deep", {})
    baseline_history = histories.get("tabular", {})

    agent = DQNAgent(hidden_layers=hidden_layers, device_preference=device_preference)
    checkpoint_state = {}
    if resume and os.path.exists(checkpoint_path):
        checkpoint_state = agent.load(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")

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
    )
    dashboard.load_deep_history(deep_history)
    dashboard.set_baseline_visibility(comparison_mode and not fast_mode)
    if comparison_mode and not fast_mode and baseline_history:
        dashboard.set_baseline_history(baseline_history)

    best_score = max(dashboard.deep_best_history) if dashboard.deep_best_history else 0
    session_start_games = agent.n_games
    training_completed = False
    last_transition = checkpoint_state.get("last_transition", {"reward_text": "n/a"})
    fast_tail_episodes = max(1, int(fast_tail_episodes))
    recent_episode_replays = deque(maxlen=3)

    try:
        while True:
            current_goal = dashboard.get_episode_goal()
            completed_in_session = agent.n_games - session_start_games
            if completed_in_session >= current_goal:
                break

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

            game.reset()
            state = agent.encode_state(game)
            step_count = 0
            episode_reward = 0.0
            episode_replay_frames = []
            capture_episode_replay = bool(
                episode_fast_mode
                and render
                and not dashboard.headless_toggle.value
                and animated_tail_episode
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
                        context=build_training_context(
                            "Training",
                            episode_reward,
                            agent.last_train_info,
                            last_transition,
                            fast_mode_requested=dashboard.fast_mode_toggle.value,
                            fast_mode_effective=episode_fast_mode,
                            fast_tail_episodes=fast_tail_episodes,
                            episodes_remaining=episodes_remaining,
                        ),
                        lightweight=stripped_episode,
                        show_baseline=episode_show_baseline,
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
                        context=build_training_context(
                            "Ready to start",
                            episode_reward,
                            agent.last_train_info,
                            last_transition,
                            fast_mode_requested=dashboard.fast_mode_toggle.value,
                            fast_mode_effective=episode_fast_mode,
                            fast_tail_episodes=fast_tail_episodes,
                            episodes_remaining=episodes_remaining,
                        ),
                        lightweight=False,
                        show_baseline=episode_show_baseline,
                        overlay_title="Press Start",
                        overlay_subtitle="Click Start [Enter] to begin training.",
                    )
                    pygame.time.delay(30)
                    continue

                if dashboard.pause_toggle.value:
                    preview_info = agent.get_action_details(state, greedy=True)
                    context = build_training_context(
                        "Paused",
                        episode_reward,
                        agent.last_train_info,
                        last_transition,
                        fast_mode_requested=dashboard.fast_mode_toggle.value,
                        fast_mode_effective=episode_fast_mode,
                        fast_tail_episodes=fast_tail_episodes,
                        episodes_remaining=episodes_remaining,
                    )
                    game.set_dashboard_data(
                        dashboard.build_dashboard_data(
                            agent=agent,
                            game=game,
                            state=state,
                            action_info=preview_info,
                            current_game_number=current_game_number,
                            episode_goal=episode_goal,
                            best_score=best_score,
                            context=context,
                            lightweight=stripped_episode,
                            show_baseline=episode_show_baseline,
                        )
                    )
                    if render:
                        game.draw()
                        pygame.time.delay(30)
                    else:
                        pygame.time.delay(5)
                    continue

                action_info = agent.get_action_details(state)
                step_count += 1
                live_render_enabled = (
                    render and not dashboard.headless_toggle.value and not stripped_episode
                )

                if live_render_enabled:
                    frame_data = dashboard.build_dashboard_data(
                        agent=agent,
                        game=game,
                        state=state,
                        action_info=action_info,
                        current_game_number=current_game_number,
                        episode_goal=episode_goal,
                        best_score=best_score,
                        context=build_training_context(
                            "Fast tail" if animated_tail_episode else "Training",
                            episode_reward,
                            agent.last_train_info,
                            last_transition,
                            fast_mode_requested=dashboard.fast_mode_toggle.value,
                            fast_mode_effective=episode_fast_mode,
                            fast_tail_episodes=fast_tail_episodes,
                            episodes_remaining=episodes_remaining,
                        ),
                        lightweight=False,
                        show_baseline=episode_show_baseline,
                    )
                    game.set_dashboard_data(frame_data)
                    if capture_episode_replay:
                        episode_replay_frames.append(capture_replay_frame(game, frame_data))
                    if dashboard.should_draw_frame(step_count):
                        game.draw()
                        if dashboard.current_delay_ms > 0:
                            pygame.time.delay(dashboard.current_delay_ms)

                reward, game_over, score = game.play_step(
                    action_info["action"],
                    events=[],
                    draw_frame=False,
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
                train_info = agent.train_step()
                state = next_state
                last_transition = {
                    "reward_text": f"{reward:+.2f}",
                    "done": game_over,
                    "score": score,
                }

                if render and game_over and not dashboard.headless_toggle.value and not stripped_episode:
                    final_view = dashboard.build_dashboard_data(
                        agent=agent,
                        game=game,
                        state=state,
                        action_info=action_info,
                        current_game_number=current_game_number,
                        episode_goal=episode_goal,
                        best_score=best_score,
                        context=build_training_context(
                            "Fast tail" if animated_tail_episode else "Training",
                            episode_reward,
                            train_info,
                            last_transition,
                            fast_mode_requested=dashboard.fast_mode_toggle.value,
                            fast_mode_effective=episode_fast_mode,
                            fast_tail_episodes=fast_tail_episodes,
                            episodes_remaining=episodes_remaining,
                        ),
                        lightweight=False,
                        show_baseline=episode_show_baseline,
                    )
                    final_view["overlay_title"] = "Episode finished"
                    final_view["overlay_subtitle"] = f"Reward: {reward:+.2f}"
                    game.set_dashboard_data(final_view)
                    if capture_episode_replay:
                        episode_replay_frames.append(capture_replay_frame(game, final_view))
                    game.draw()
                    pygame.time.delay(
                        120 if dashboard.turbo_toggle.value else max(140, dashboard.current_delay_ms)
                    )

                if game.quit_requested or game_over:
                    break

            if game.quit_requested:
                print("Training stopped because the game window was closed.")
                break

            agent.n_games += 1
            agent.decay_epsilon()
            dashboard.record_deep_episode(score, episode_reward)
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
                    context=build_training_context(
                        "Fast pass" if stripped_episode else "Headless training",
                        episode_reward,
                        agent.last_train_info,
                        last_transition,
                        fast_mode_requested=dashboard.fast_mode_toggle.value,
                        fast_mode_effective=episode_fast_mode,
                        fast_tail_episodes=fast_tail_episodes,
                        episodes_remaining=max(
                            0,
                            dashboard.get_episode_goal() - (agent.n_games - session_start_games),
                        ),
                    ),
                    lightweight=stripped_episode,
                    show_baseline=episode_show_baseline,
                )

            if agent.n_games % checkpoint_every == 0 or score == best_score:
                save_checkpoint(agent, checkpoint_path, dashboard, metrics_log_path, last_transition)

            print(
                f"Run {current_game_number:>4}/{dashboard.get_episode_goal():<4} | "
                f"Total games: {agent.n_games:>4} | "
                f"Score: {score:>2} | "
                f"Best: {best_score:>2} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.replay_buffer)} | "
                f"Loss: {agent.last_train_info.get('loss')} | "
                f"Mode: {'fast' if episode_fast_mode else 'default'}"
            )

        training_completed = not game.quit_requested
    finally:
        save_checkpoint(agent, checkpoint_path, dashboard, metrics_log_path, last_transition)
        if render and training_completed and dashboard.keep_open_toggle.value:
            hold_training_window_open(game, dashboard, recent_episode_replays)
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
