from __future__ import annotations

import argparse
import os

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


def hold_training_window_open(game, dashboard):
    pygame.event.clear()
    final_view = dict(game.dashboard_data)
    final_view["overlay_title"] = "Training finished"
    final_view["overlay_subtitle"] = "Press Enter, Q, Esc, or close the window."
    game.set_dashboard_data(final_view)

    while not game.quit_requested:
        events = pygame.event.get()
        game.handle_system_events(events)

        for event in events:
            if event.type == pygame.KEYDOWN and event.key in (
                pygame.K_RETURN,
                pygame.K_KP_ENTER,
                pygame.K_q,
                pygame.K_ESCAPE,
            ):
                return

        dashboard.sync_graph_rect(game)
        dashboard.handle_events(events)
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
):
    histories = prepare_metrics_log(metrics_log_path, resume)
    deep_history = histories.get("deep", {})
    baseline_history = histories.get("tabular", {})

    agent = DQNAgent(hidden_layers=hidden_layers)
    checkpoint_state = {}
    if resume and os.path.exists(checkpoint_path):
        checkpoint_state = agent.load(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")

    initial_reward_config = checkpoint_state.get("reward_config", DEFAULT_REWARD_CONFIG)
    deep_history = checkpoint_state.get("deep_history", deep_history)

    game = SnakeGameAI(w=640, h=700, window_h=980, render=render, speed=speed)
    dashboard = TrainingDashboard(
        game,
        initial_speed=speed,
        initial_delay_ms=delay_ms,
        initial_episode_goal=episodes,
        initial_reward_config=initial_reward_config,
    )
    dashboard.load_deep_history(deep_history)
    if comparison_mode and baseline_history:
        dashboard.set_baseline_history(baseline_history)

    if comparison_mode and not dashboard.baseline_scores:
        baseline_history = run_tabular_baseline(
            baseline_episodes,
            reward_config=dashboard.reward_config,
            metrics_log_path=metrics_log_path,
        )
        dashboard.set_baseline_history(baseline_history)

    best_score = max(dashboard.deep_best_history) if dashboard.deep_best_history else 0
    session_start_games = agent.n_games
    training_completed = False
    last_transition = checkpoint_state.get("last_transition", {"reward_text": "n/a"})

    try:
        while True:
            current_goal = dashboard.get_episode_goal()
            if (agent.n_games - session_start_games) >= current_goal:
                break

            game.reset()
            current_game_number = (agent.n_games - session_start_games) + 1
            state = agent.encode_state(game)
            step_count = 0
            episode_reward = 0.0

            while True:
                episode_goal = dashboard.get_episode_goal()
                events = pygame.event.get() if render else []
                dashboard.sync_graph_rect(game)
                dashboard.handle_events(events)
                game.handle_system_events(events)

                if game.quit_requested:
                    break

                game.speed = 0 if dashboard.headless_toggle.value else dashboard.current_fps
                game.set_reward_config(dashboard.reward_config)

                if dashboard.pause_toggle.value:
                    preview_info = agent.get_action_details(state, greedy=True)
                    context = {
                        "mode_label": "Paused",
                        "episode_reward": episode_reward,
                        "train_info": agent.last_train_info,
                        "transition": last_transition,
                    }
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
                context = {
                    "mode_label": "Training",
                    "episode_reward": episode_reward,
                    "train_info": agent.last_train_info,
                    "transition": last_transition,
                }
                game.set_dashboard_data(
                    dashboard.build_dashboard_data(
                        agent=agent,
                        game=game,
                        state=state,
                        action_info=action_info,
                        current_game_number=current_game_number,
                        episode_goal=episode_goal,
                        best_score=best_score,
                        context=context,
                    )
                )

                if render and dashboard.should_draw_frame(step_count):
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

                if render and game_over and not dashboard.headless_toggle.value:
                    final_view = dashboard.build_dashboard_data(
                        agent=agent,
                        game=game,
                        state=state,
                        action_info=action_info,
                        current_game_number=current_game_number,
                        episode_goal=episode_goal,
                        best_score=best_score,
                        context={
                            "mode_label": "Training",
                            "episode_reward": episode_reward,
                            "train_info": train_info,
                            "transition": last_transition,
                        },
                    )
                    final_view["overlay_title"] = "Episode finished"
                    final_view["overlay_subtitle"] = f"Reward: {reward:+.2f}"
                    game.set_dashboard_data(final_view)
                    game.draw()
                    pygame.time.delay(120 if dashboard.turbo_toggle.value else max(140, dashboard.current_delay_ms))

                if game.quit_requested or game_over:
                    break

            if game.quit_requested:
                print("Training stopped because the game window was closed.")
                break

            agent.n_games += 1
            agent.decay_epsilon()
            dashboard.record_deep_episode(score, episode_reward)
            best_score = max(best_score, score)

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

            if agent.n_games % checkpoint_every == 0 or score == best_score:
                save_checkpoint(agent, checkpoint_path, dashboard, metrics_log_path, last_transition)

            print(
                f"Run {current_game_number:>4}/{dashboard.get_episode_goal():<4} | "
                f"Total games: {agent.n_games:>4} | "
                f"Score: {score:>2} | "
                f"Best: {best_score:>2} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Buffer: {len(agent.replay_buffer)} | "
                f"Loss: {agent.last_train_info.get('loss')}"
            )

        training_completed = not game.quit_requested
    finally:
        save_checkpoint(agent, checkpoint_path, dashboard, metrics_log_path, last_transition)
        if render and training_completed and dashboard.keep_open_toggle.value:
            hold_training_window_open(game, dashboard)
        game.close()

    print(f"Training finished. Checkpoint saved to {checkpoint_path}")


def run_visualizer_session(checkpoint_path, metrics_log_path, speed):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint_config = load_checkpoint_network_config(checkpoint_path)
    agent = DQNAgent(hidden_layers=checkpoint_config["hidden_layers"])
    checkpoint_state = agent.load(checkpoint_path)
    resolved_metrics_log = metrics_log_path or checkpoint_state.get("metrics_log_path")
    histories = load_histories_from_log(resolved_metrics_log)
    deep_history = checkpoint_state.get("deep_history", histories.get("deep", {}))
    baseline_history = histories.get("tabular", {})

    game = SnakeGameAI(w=640, h=700, window_h=980, render=True, speed=speed)
    dashboard = TrainingDashboard(
        game,
        initial_speed=speed,
        initial_delay_ms=40,
        initial_episode_goal=max(1, len(deep_history.get("scores", []))),
        initial_reward_config=checkpoint_state.get("reward_config", DEFAULT_REWARD_CONFIG),
    )
    dashboard.load_deep_history(deep_history)
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
                dashboard.sync_graph_rect(game)
                dashboard.handle_events(events)
                game.handle_system_events(events)

                if game.quit_requested:
                    break

                game.speed = 0 if dashboard.headless_toggle.value else dashboard.current_fps
                action_info = agent.get_action_details(state, greedy=True)
                context = {
                    "mode_label": "Viewer",
                    "episode_reward": episode_reward,
                    "train_info": agent.last_train_info,
                    "transition": last_transition,
                }

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
                    )
                )

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
        description="Train a CPU-first Snake DQN with a live pygame dashboard."
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
        "--hidden-layers",
        type=parse_hidden_layers_arg,
        default=None,
        help="Comma-separated hidden-layer sizes, for example 64 or 128,128,64.",
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
    )


def main_visualizer():
    parser = argparse.ArgumentParser(
        description="View a saved Snake DQN checkpoint with the pygame dashboard."
    )
    parser.add_argument("--checkpoint", default="dqn_checkpoint.pt", help="Checkpoint file path.")
    parser.add_argument("--metrics-log", default=None, help="Optional JSONL metric log path.")
    parser.add_argument("--speed", type=int, default=16, help="Playback speed.")
    args = parser.parse_args()

    run_visualizer_session(
        checkpoint_path=args.checkpoint,
        metrics_log_path=args.metrics_log,
        speed=args.speed,
    )
