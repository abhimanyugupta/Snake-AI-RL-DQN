from __future__ import annotations

try:
    import pygame
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency 'pygame'. Install requirements.txt before running the Deep RL project."
    ) from exc


class SliderControl:
    def __init__(self, label, min_value, max_value, value, x, y, width, formatter=None):
        self.label = label
        self.min_value = min_value
        self.max_value = max_value
        self.value = value
        self.formatter = formatter or (lambda current: f"{current:.2f}")
        self.track_rect = pygame.Rect(x, y + 20, width, 6)
        self.hit_rect = pygame.Rect(x, y + 10, width, 28)
        self.dragging = False

    @property
    def normalized(self):
        span = self.max_value - self.min_value
        if span == 0:
            return 0.0
        return (self.value - self.min_value) / span

    def set_normalized(self, ratio):
        ratio = max(0.0, min(1.0, ratio))
        self.value = self.min_value + ratio * (self.max_value - self.min_value)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hit_rect.collidepoint(event.pos):
                self.dragging = True
                self._update_from_mouse(event.pos[0])
                return True

        if event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_from_mouse(event.pos[0])
            return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            self.dragging = False
            self._update_from_mouse(event.pos[0])
            return True

        return False

    def _update_from_mouse(self, mouse_x):
        ratio = (mouse_x - self.track_rect.x) / self.track_rect.width
        self.set_normalized(ratio)

    def draw_data(self, value_text=None):
        knob_x = self.track_rect.x + int(self.normalized * self.track_rect.width)
        return {
            "label": self.label,
            "value_text": value_text or self.formatter(self.value),
            "x": self.track_rect.x,
            "y": self.track_rect.y - 20,
            "track_x": self.track_rect.x,
            "track_y": self.track_rect.y,
            "track_w": self.track_rect.width,
            "track_h": self.track_rect.height,
            "ratio": self.normalized,
            "knob_x": knob_x,
            "knob_y": self.track_rect.y + self.track_rect.height // 2,
            "knob_radius": 8,
        }


class ToggleControl:
    def __init__(self, label, value, x, y, width, height):
        self.label = label
        self.value = value
        self.rect = pygame.Rect(x, y, width, height)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.value = not self.value
                return True
        return False

    def toggle(self):
        self.value = not self.value

    def draw_data(self):
        return {
            "label": self.label,
            "value": self.value,
            "x": self.rect.x,
            "y": self.rect.y,
            "w": self.rect.width,
            "h": self.rect.height,
        }


class TextInputControl:
    def __init__(self, label, value, x, y, width, height, max_length=6):
        self.label = label
        self.text = str(value)
        self.rect = pygame.Rect(x, y, width, height)
        self.max_length = max_length
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            was_active = self.active
            self.active = self.rect.collidepoint(event.pos)
            return was_active or self.active

        if not self.active or event.type != pygame.KEYDOWN:
            return False

        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE):
            self.active = False
            return True

        if event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
            return True

        if event.unicode.isdigit() and len(self.text) < self.max_length:
            self.text += event.unicode
            return True

        return False

    def get_int(self, default=1, minimum=1, maximum=999999):
        raw_value = self.text.strip()
        if not raw_value:
            return max(minimum, min(maximum, default))

        value = int(raw_value)
        return max(minimum, min(maximum, value))

    def draw_data(self):
        return {
            "label": self.label,
            "text": self.text,
            "active": self.active,
            "x": self.rect.x,
            "y": self.rect.y,
            "w": self.rect.width,
            "h": self.rect.height,
            "hint": "Type a number",
        }


class ButtonControl:
    def __init__(self, label, x, y, width, height, mode_value):
        self.label = label
        self.rect = pygame.Rect(x, y, width, height)
        self.mode_value = mode_value

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            return self.rect.collidepoint(event.pos)
        return False

    def draw_data(self, active=False, disabled=False, style="default", label=None):
        return {
            "label": label or self.label,
            "active": active,
            "disabled": disabled,
            "style": style,
            "x": self.rect.x,
            "y": self.rect.y,
            "w": self.rect.width,
            "h": self.rect.height,
        }


class TrainingDashboard:
    def __init__(
        self,
        game,
        initial_speed,
        initial_delay_ms,
        initial_episode_goal,
        initial_reward_config=None,
        initial_headless=False,
        initial_device_preference="cpu",
        cuda_available=False,
        require_manual_start=False,
        initial_fast_mode=False,
    ):
        initial_reward_config = initial_reward_config or {
            "food": 10.0,
            "death": -10.0,
            "step": 0.0,
        }

        padding = 20
        self.metrics_row_h = 28
        start_y = padding + 46 + 42 + 52 + (7 * self.metrics_row_h) + 20
        col_w = (game.sidebar_width - (padding * 2) - 20) // 2
        col_x = game.board_w + padding
        slider_width = col_w
        toggle_w = (col_w - 10) // 2

        self.view_mode = "overview"
        self.view_order = ["overview", "network", "algorithm"]
        self.cuda_available = bool(cuda_available)
        self.selected_device_preference = (
            "cuda"
            if str(initial_device_preference).lower() == "cuda" and self.cuda_available
            else "cpu"
        )
        self.require_manual_start = bool(require_manual_start)
        self.started = not self.require_manual_start
        self.overview_button = ButtonControl("Overview [Tab]", col_x, padding + 44, 156, 30, "overview")
        self.network_button = ButtonControl("Network [N]", col_x + 168, padding + 44, 134, 30, "network")
        self.algorithm_button = ButtonControl("Algorithm [E]", col_x + 314, padding + 44, 144, 30, "algorithm")
        self.view_buttons = [
            self.overview_button,
            self.network_button,
            self.algorithm_button,
        ]

        y = start_y
        
        y += 18
        speed_ratio = self._speed_ratio_from_settings(initial_speed, initial_delay_ms)
        self.speed_slider = SliderControl("Training speed", 0.0, 1.0, speed_ratio, col_x, y, slider_width)
        y += 40
        
        y += 18
        self.food_reward_slider = SliderControl(
            "Food reward",
            1.0,
            20.0,
            float(initial_reward_config.get("food", 10.0)),
            col_x,
            y,
            slider_width,
        )
        y += 40
        self.death_reward_slider = SliderControl(
            "Death penalty",
            -20.0,
            -1.0,
            float(initial_reward_config.get("death", -10.0)),
            col_x,
            y,
            slider_width,
        )
        y += 40
        self.step_reward_slider = SliderControl(
            "Step reward",
            -1.0,
            1.0,
            float(initial_reward_config.get("step", 0.0)),
            col_x,
            y,
            slider_width,
        )

        y += 42
        y += 18
        self.show_arrows_toggle = ToggleControl("Arrows [A]", True, col_x, y, toggle_w, 26)
        self.show_dangers_toggle = ToggleControl("Danger [D]", True, col_x + toggle_w + 10, y, toggle_w, 26)
        y += 32
        self.show_graph_toggle = ToggleControl("Graph [G]", True, col_x, y, toggle_w, 26)
        self.pause_toggle = ToggleControl("Pause [Space]", False, col_x + toggle_w + 10, y, toggle_w, 26)
        self.start_button = ButtonControl("Start [Enter]", col_x + toggle_w + 10, y, toggle_w, 26, "start")
        y += 48
        self.turbo_toggle = ToggleControl("Turbo [T]", False, col_x, y, toggle_w, 26)
        self.episode_input = TextInputControl(
            "Episode goal",
            initial_episode_goal,
            col_x + toggle_w + 10,
            y,
            toggle_w,
            28,
        )
        y += 36
        self.keep_open_toggle = ToggleControl("Keep open [K]", True, col_x, y, toggle_w, 26)
        self.headless_toggle = ToggleControl("No Render [H]", bool(initial_headless), col_x + toggle_w + 10, y, toggle_w, 26)
        y += 32

        self.fast_mode_toggle = ToggleControl("Fast Mode [X]", bool(initial_fast_mode), col_x, y, slider_width, 26)
        y += 32

        self.cpu_device_button = ButtonControl("CPU [C]", col_x, y, toggle_w, 26, "device_cpu")
        self.gpu_device_button = ButtonControl("GPU [U]", col_x + toggle_w + 10, y, toggle_w, 26, "device_cuda")
        y += 32

        self.show_scores_toggle = ToggleControl("Scores [S]", True, col_x, y, toggle_w, 26)
        self.show_avg_toggle = ToggleControl("Avg [M]", True, col_x + toggle_w + 10, y, toggle_w, 26)
        y += 32
        self.show_best_toggle = ToggleControl("Best [B]", True, col_x, y, toggle_w, 26)

        self.sliders = [
            self.speed_slider,
            self.food_reward_slider,
            self.death_reward_slider,
            self.step_reward_slider,
        ]
        self.toggles = [
            self.show_arrows_toggle,
            self.show_dangers_toggle,
            self.show_graph_toggle,
            self.pause_toggle,
            self.turbo_toggle,
            self.keep_open_toggle,
            self.headless_toggle,
            self.fast_mode_toggle,
            self.show_scores_toggle,
            self.show_avg_toggle,
            self.show_best_toggle,
        ]
        self.inputs = [self.episode_input]
        self.control_buttons = [
            self.start_button,
            self.cpu_device_button,
            self.gpu_device_button,
        ]

        self.initial_episode_goal = initial_episode_goal
        self.last_reward = 0.0
        self.deep_scores = []
        self.deep_average_history = []
        self.deep_best_history = []
        self.deep_episode_rewards = []
        self.baseline_scores = []
        self.baseline_average_history = []
        self.baseline_best_history = []
        self.baseline_episode_rewards = []

        self.graph_view_end = None
        self.graph_view_size = 60
        self.graph_drag_active = False
        self.graph_drag_start_x = 0
        self.graph_drag_start_end = 0
        self.graph_hover_index = None
        self.graph_rect = None
        self.baseline_visible = True

    @property
    def current_fps(self):
        if self.turbo_toggle.value:
            return int(60 + (self.speed_slider.value * 240))
        return int(6 + (self.speed_slider.value * 100))

    @property
    def current_delay_ms(self):
        if self.turbo_toggle.value:
            return 0
        return int((1.0 - self.speed_slider.value) * 130)

    @property
    def render_every_n_steps(self):
        if not self.turbo_toggle.value:
            return 1
        return 2 + int(self.speed_slider.value * 10)

    @property
    def speed_mode_label(self):
        if self.turbo_toggle.value:
            return "Turbo"
        ratio = self.speed_slider.value
        if ratio < 0.34:
            return "Slow"
        if ratio < 0.67:
            return "Medium"
        return "Fast"

    @property
    def reward_config(self):
        return {
            "food": round(self.food_reward_slider.value, 2),
            "death": round(self.death_reward_slider.value, 2),
            "step": round(self.step_reward_slider.value, 2),
        }

    def _speed_ratio_from_settings(self, speed, delay_ms):
        speed_ratio = max(0.0, min(1.0, (speed - 5) / 115))
        delay_ratio = 1.0 - max(0.0, min(1.0, delay_ms / 140 if delay_ms else 0.0))
        return round((speed_ratio + delay_ratio) / 2, 2)

    def get_episode_goal(self):
        return self.episode_input.get_int(default=self.initial_episode_goal)

    def should_draw_frame(self, step_number, force=False):
        if self.headless_toggle.value:
            return False
        if force or not self.turbo_toggle.value:
            return True
        return step_number <= 1 or (step_number % self.render_every_n_steps == 0)

    def load_deep_history(self, history):
        self.deep_scores = list(history.get("scores", []))
        self.deep_average_history = list(history.get("moving_avg", []))
        self.deep_best_history = list(history.get("best_scores", []))
        self.deep_episode_rewards = list(history.get("episode_rewards", []))

    def set_baseline_history(self, history):
        self.baseline_scores = list(history.get("scores", []))
        self.baseline_average_history = list(history.get("moving_avg", []))
        self.baseline_best_history = list(history.get("best_scores", []))
        self.baseline_episode_rewards = list(history.get("episode_rewards", []))

    def set_baseline_visibility(self, visible):
        self.baseline_visible = bool(visible)

    def record_deep_episode(self, score, episode_reward):
        self.deep_scores.append(score)
        self.deep_episode_rewards.append(float(episode_reward))
        recent_scores = self.deep_scores[-20:]
        moving_average = sum(recent_scores) / len(recent_scores)
        self.deep_average_history.append(moving_average)
        best_score = max(self.deep_best_history[-1], score) if self.deep_best_history else score
        self.deep_best_history.append(best_score)

    def sync_graph_rect(self, game):
        data = game.dashboard_data
        if data and "_graph_rect" in data:
            self.graph_rect = data["_graph_rect"]

    def graph_total_points(self):
        total = 0
        if self.show_scores_toggle.value:
            total = max(total, len(self.deep_scores))
            if self.baseline_visible:
                total = max(total, len(self.baseline_scores))
        if self.show_avg_toggle.value:
            total = max(total, len(self.deep_average_history))
            if self.baseline_visible:
                total = max(total, len(self.baseline_average_history))
        if self.show_best_toggle.value:
            total = max(total, len(self.deep_best_history))
            if self.baseline_visible:
                total = max(total, len(self.baseline_best_history))
        return total

    def handle_events(self, events):
        for event in events:
            consumed_by_input = False
            for input_control in self.inputs:
                if input_control.handle_event(event):
                    consumed_by_input = True
                    break

            if consumed_by_input:
                continue

            if self._handle_graph_event(event):
                continue

            for button in self.view_buttons:
                if button.handle_event(event):
                    self.view_mode = button.mode_value
                    break
            else:
                if event.type == pygame.KEYDOWN:
                    self._handle_shortcuts(event.key)

                for control_button in self.control_buttons:
                    if control_button.handle_event(event):
                        self._handle_control_button(control_button.mode_value)
                        break
                else:
                    for slider in self.sliders:
                        if slider.handle_event(event):
                            break
                    else:
                        for toggle in self.toggles:
                            if toggle.handle_event(event):
                                break

    def _handle_graph_event(self, event):
        gr = self.graph_rect
        total = self.graph_total_points()
        if gr is None or total < 2:
            return False

        if event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if gr.collidepoint(mx, my):
                if event.y > 0:
                    new_size = max(10, int(self.graph_view_size / 1.4))
                else:
                    new_size = min(total, int(self.graph_view_size * 1.4))
                self.graph_view_size = new_size
                if self.graph_view_end is not None:
                    self.graph_view_end = min(total, max(new_size, self.graph_view_end))
                return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and gr.collidepoint(event.pos):
            self.graph_drag_active = True
            self.graph_drag_start_x = event.pos[0]
            view_end = self.graph_view_end if self.graph_view_end is not None else total
            self.graph_drag_start_end = view_end
            return True

        if event.type == pygame.MOUSEMOTION:
            if gr.collidepoint(event.pos) and not self.graph_drag_active:
                plot_x = gr.x + 15
                plot_w = gr.width - 30
                if plot_w > 0:
                    view_end = self.graph_view_end if self.graph_view_end is not None else total
                    view_start = max(0, view_end - self.graph_view_size)
                    n_visible = view_end - view_start
                    rel_x = event.pos[0] - plot_x
                    idx = view_start + int(rel_x / plot_w * max(1, n_visible))
                    idx = max(view_start, min(view_end - 1, idx))
                    self.graph_hover_index = idx
            elif not self.graph_drag_active:
                self.graph_hover_index = None

            if self.graph_drag_active:
                dx_pixels = event.pos[0] - self.graph_drag_start_x
                plot_w = gr.width - 30
                if plot_w > 0:
                    runs_per_pixel = self.graph_view_size / plot_w
                    delta_runs = int(-dx_pixels * runs_per_pixel)
                    new_end = self.graph_drag_start_end + delta_runs
                    new_end = max(self.graph_view_size, min(total, new_end))
                    self.graph_view_end = new_end
                return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.graph_drag_active:
            self.graph_drag_active = False
            if self.graph_view_end is not None and self.graph_view_end >= total:
                self.graph_view_end = None
            return True

        return False

    def _handle_shortcuts(self, key):
        if key == pygame.K_SPACE:
            if self.started:
                self.pause_toggle.toggle()
        elif key == pygame.K_TAB:
            current_index = self.view_order.index(self.view_mode)
            self.view_mode = self.view_order[(current_index + 1) % len(self.view_order)]
        elif key == pygame.K_n:
            self.view_mode = "network"
        elif key == pygame.K_e:
            self.view_mode = "algorithm"
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._handle_control_button("start")
        elif key == pygame.K_a:
            self.show_arrows_toggle.toggle()
        elif key == pygame.K_d:
            self.show_dangers_toggle.toggle()
        elif key == pygame.K_g:
            self.show_graph_toggle.toggle()
        elif key == pygame.K_t:
            self.turbo_toggle.toggle()
        elif key == pygame.K_k:
            self.keep_open_toggle.toggle()
        elif key == pygame.K_h:
            self.headless_toggle.toggle()
        elif key == pygame.K_x:
            self.fast_mode_toggle.toggle()
        elif key == pygame.K_c:
            self._handle_control_button("device_cpu")
        elif key == pygame.K_u:
            self._handle_control_button("device_cuda")
        elif key == pygame.K_s:
            self.show_scores_toggle.toggle()
        elif key == pygame.K_m:
            self.show_avg_toggle.toggle()
        elif key == pygame.K_b:
            self.show_best_toggle.toggle()
        elif key == pygame.K_1:
            self.turbo_toggle.value = False
            self.speed_slider.set_normalized(0.15)
        elif key == pygame.K_2:
            self.turbo_toggle.value = False
            self.speed_slider.set_normalized(0.5)
        elif key == pygame.K_3:
            self.turbo_toggle.value = False
            self.speed_slider.set_normalized(0.9)
        elif key == pygame.K_4:
            self.turbo_toggle.value = True
            self.speed_slider.set_normalized(1.0)
        elif key == pygame.K_f:
            total = self.graph_total_points()
            if total > 0:
                self.graph_view_size = total
                self.graph_view_end = None

    def _handle_control_button(self, mode_value):
        if mode_value == "start":
            self.started = True
            self.pause_toggle.value = False
        elif mode_value == "device_cpu":
            self.selected_device_preference = "cpu"
        elif mode_value == "device_cuda" and self.cuda_available:
            self.selected_device_preference = "cuda"

    def build_dashboard_data(
        self,
        agent,
        game,
        state,
        action_info,
        current_game_number,
        episode_goal,
        best_score,
        context,
        lightweight=False,
        show_baseline=None,
    ):
        show_baseline = self.baseline_visible if show_baseline is None else bool(show_baseline)
        include_network = self.view_mode == "network" and not lightweight
        candidate_points = game.get_relative_points() if not lightweight else {}
        deadly_moves = (
            {key: game.is_collision(point) for key, point in candidate_points.items()}
            if candidate_points
            else {}
        )
        q_values = action_info["q_values"]
        train_info = context["train_info"]
        fast_mode_requested = bool(context.get("fast_mode_requested", False))
        fast_mode_effective = bool(context.get("fast_mode_effective", False))
        fast_tail_episodes = int(context.get("fast_tail_episodes", 5))
        episodes_remaining = int(context.get("episodes_remaining", 0))

        if lightweight:
            network_view = {
                "message": (
                    "Network inspection is disabled during stripped fast episodes. "
                    f"It returns for the final {fast_tail_episodes} animated episode(s)."
                )
            }
            feature_lines = [
                "Fast mode is stripping step-by-step visualization for this episode.",
                f"Final animated tail: last {fast_tail_episodes} episode(s).",
                f"Episodes remaining in this session: {episodes_remaining}",
                f"Warmup remaining: {train_info.get('warmup_remaining', 0)}",
                f"Replay buffer fill: {train_info.get('buffer_size', 0)}",
                f"Current device: {agent.device_label}",
            ]
            help_lines = [
                "[accent]Board redraws, network heatmaps, and per-step dashboard rebuilds are skipped.",
                "[accent]Episode metrics, score history, checkpoints, and logs still update.",
            ]
        else:
            state_items = agent.describe_state(state)
            top_state_items = sorted(
                state_items,
                key=lambda item: abs(item["value"]),
                reverse=True,
            )[:6]
            network_view = (
                agent.inspect_network(state, action_info["action_index"])
                if include_network
                else {}
            )
            feature_lines = [
                f"{item['label']}: {item['value']:+.2f}"
                for item in top_state_items
            ]
            feature_lines.extend(
                [
                    f"Food view: {agent.explain_food_view(state)}",
                    f"Warmup remaining: {train_info.get('warmup_remaining', 0)}",
                    f"Replay buffer fill: {train_info.get('buffer_size', 0)}",
                ]
            )
            help_lines = [
                "[accent]Overview: board on the left, current decision on the right, history below.",
                "[accent]Network: live forward pass across the configurable hidden layers.",
                "[accent]Algorithm: Bellman target, replay memory, and target-net update flow.",
                f"[accent]Device target: {self.selected_device_preference.upper()} | Active runtime: {agent.device_label}",
            ]

        if fast_mode_requested and fast_mode_requested != fast_mode_effective:
            help_lines.append("[accent]Fast mode change is queued and will apply next episode.")
        elif fast_mode_effective and not lightweight:
            help_lines.append(
                f"[accent]Fast mode is active. The final {fast_tail_episodes} episode(s) are animated."
            )

        best_q = max(q_values)
        best_action_index = q_values.index(best_q)
        decision_summary = [
            f"Chosen move: {action_info['action_label']}",
            f"Selection mode: {action_info['decision_type']}",
            f"Greedy favorite: {agent.ACTION_LABELS[best_action_index]} ({best_q:+.3f})",
            f"Food cue: {agent.explain_food_view(state)}",
            f"Architecture: {agent.architecture_label}",
        ]
        if lightweight:
            decision_summary.append("Visualization: stripped fast pass")
        elif fast_mode_effective:
            decision_summary.append(f"Visualization: animated tail ({fast_tail_episodes} episode window)")

        metrics = [
            ("Run", f"{current_game_number}/{max(episode_goal, current_game_number)}"),
            ("Mode", context.get("mode_label", "Training")),
            ("Score", game.score),
            ("Best score", best_score),
            ("Episode reward", f"{context.get('episode_reward', 0.0):+.2f}"),
            ("Epsilon", f"{agent.epsilon:.3f}"),
            ("Loss", self._format_metric(train_info.get("loss"))),
            ("TD error", self._format_metric(train_info.get("td_error"))),
            ("Buffer fill", train_info.get("buffer_size", 0)),
            ("Warmup left", train_info.get("warmup_remaining", 0)),
            ("Target sync in", train_info.get("target_sync_remaining", "-")),
            ("Device", agent.device_label),
            ("Architecture", agent.architecture_label),
            ("Hidden layers", ",".join(str(size) for size in agent.hidden_layers)),
        ]

        graph_series = [
            {
                "label": "Deep score",
                "values": self.deep_scores,
                "visible": self.show_scores_toggle.value,
                "color": (80, 190, 255),
                "thickness": 2,
            },
            {
                "label": "Deep avg",
                "values": self.deep_average_history,
                "visible": self.show_avg_toggle.value,
                "color": (80, 230, 120),
                "thickness": 3,
            },
            {
                "label": "Deep best",
                "values": self.deep_best_history,
                "visible": self.show_best_toggle.value,
                "color": (255, 196, 68),
                "thickness": 2,
            },
            {
                "label": "Tabular score",
                "values": self.baseline_scores,
                "visible": show_baseline and self.show_scores_toggle.value and bool(self.baseline_scores),
                "color": (110, 140, 180),
                "thickness": 2,
            },
            {
                "label": "Tabular avg",
                "values": self.baseline_average_history,
                "visible": show_baseline and self.show_avg_toggle.value and bool(self.baseline_average_history),
                "color": (130, 190, 150),
                "thickness": 2,
            },
            {
                "label": "Tabular best",
                "values": self.baseline_best_history,
                "visible": show_baseline and self.show_best_toggle.value and bool(self.baseline_best_history),
                "color": (200, 165, 100),
                "thickness": 2,
            },
        ]

        comparison_lines = self._build_comparison_lines(
            show_baseline=show_baseline,
            fast_mode_effective=fast_mode_effective,
            fast_tail_episodes=fast_tail_episodes,
        )
        algorithm_sections = self._build_algorithm_sections(agent, action_info, context)
        control_panel = self._build_control_panel()
        control_sections = self._build_control_sections()

        return {
            "panel_title": "Snake Deep RL Lab",
            "view_mode": self.view_mode,
            "metrics_row_h": self.metrics_row_h,
            "view_buttons": [
                self.overview_button.draw_data(self.view_mode == "overview"),
                self.network_button.draw_data(self.view_mode == "network"),
                self.algorithm_button.draw_data(self.view_mode == "algorithm"),
            ],
            "metrics": metrics,
            "sliders": [
                self.speed_slider.draw_data(
                    f"{self.speed_mode_label} | {self.current_fps} fps | "
                    f"{'every ' + str(self.render_every_n_steps) + ' steps' if self.turbo_toggle.value else str(self.current_delay_ms) + ' ms delay'}"
                ),
                self.food_reward_slider.draw_data(f"{self.food_reward_slider.value:+.1f}"),
                self.death_reward_slider.draw_data(f"{self.death_reward_slider.value:+.1f}"),
                self.step_reward_slider.draw_data(f"{self.step_reward_slider.value:+.2f}"),
            ],
            "control_buttons": self._build_control_buttons(),
            "toggles": [
                toggle.draw_data()
                for toggle in self.toggles
                if self.started or toggle is not self.pause_toggle
            ],
            "inputs": [input_control.draw_data() for input_control in self.inputs],
            "show_arrows": self.show_arrows_toggle.value,
            "show_dangers": self.show_dangers_toggle.value,
            "show_graph": self.show_graph_toggle.value,
            "q_values": q_values,
            "action_labels": agent.ACTION_LABELS,
            "action_index": action_info["action_index"],
            "action_key": action_info["action_key"],
            "action_label": action_info["action_label"],
            "decision_type": action_info["decision_type"],
            "decision_summary": decision_summary,
            "candidate_points": candidate_points,
            "deadly_moves": deadly_moves,
            "state_lines": feature_lines,
            "help_lines": help_lines,
            "graph_series": graph_series,
            "graph_view_end": self.graph_view_end,
            "graph_view_size": self.graph_view_size,
            "graph_hover_index": self.graph_hover_index,
            "comparison_lines": comparison_lines,
            "algorithm_sections": algorithm_sections,
            "network_view": network_view,
            "control_panel": control_panel,
            "control_sections": control_sections,
        }

    def _build_comparison_lines(self, show_baseline, fast_mode_effective, fast_tail_episodes):
        if not show_baseline:
            if fast_mode_effective:
                return [
                    f"Fast mode is hiding the tabular baseline until the run leaves stripped mode.",
                    f"Only the final {fast_tail_episodes} episode(s) are animated in the current session.",
                    "Deep RL score history still updates after every completed episode.",
                ]
            return [
                "Tabular comparison is currently hidden for this session.",
                "Deep RL score history still updates after every completed episode.",
            ]

        lines = [
            "Bright lines are the Deep RL run using the neural network.",
            "Muted lines are the tabular baseline using a lookup table.",
        ]
        if self.baseline_scores:
            deep_best = max(self.deep_best_history) if self.deep_best_history else 0
            baseline_best = max(self.baseline_best_history) if self.baseline_best_history else 0
            lines.extend(
                [
                    f"Deep best score: {deep_best}",
                    f"Tabular best score: {baseline_best}",
                    "Deep RL may look noisier early because it is still fitting the value function.",
                ]
            )
        else:
            lines.extend(
                [
                    "Baseline lines will appear after the local tabular comparison is available.",
                    "The goal is to show memorized Q-values versus generalization through function approximation.",
                ]
            )
        return lines

    def _build_algorithm_sections(self, agent, action_info, context):
        train_info = context["train_info"]
        transition = context.get("transition", {})
        return [
            {
                "title": "RL Loop",
                "lines": [
                    "1. Observe the encoded state features.",
                    f"2. Choose {action_info['action_label']} with {action_info['decision_type']}.",
                    "3. Store (state, action, reward, next_state) in replay memory.",
                    "4. Sample a batch, compute Bellman targets, backpropagate, and sync the target net occasionally.",
                ],
            },
            {
                "title": "Current Transition",
                "lines": [
                    f"Reward: {transition.get('reward_text', 'n/a')}",
                    f"Episode score after move: {transition.get('score', 'n/a')}",
                    f"Done flag: {transition.get('done', 'n/a')}",
                    f"Replay buffer size: {train_info.get('buffer_size', 0)}",
                ],
            },
            {
                "title": "Bellman Update",
                "lines": [
                    f"Predicted Q: {self._format_metric(train_info.get('example_predicted_q'))}",
                    f"Target Q: {self._format_metric(train_info.get('example_target_q'))}",
                    f"TD error: {self._format_metric(train_info.get('example_td_error'))}",
                    f"Loss: {self._format_metric(train_info.get('loss'))}",
                ],
            },
            {
                "title": "Why It Helps",
                "lines": [
                    "Replay memory reduces correlation between updates.",
                    "The target network makes the learning target move more slowly.",
                    f"Epsilon is {agent.epsilon:.3f}, so the agent still explores when needed.",
                    f"The current MLP architecture is {agent.architecture_label}.",
                ],
            },
        ]

    def _build_control_panel(self):
        min_x = min(slider.track_rect.x for slider in self.sliders)
        max_right = max(slider.track_rect.right for slider in self.sliders)
        min_y = self.speed_slider.track_rect.y - 30
        max_bottom = max(toggle.rect.bottom for toggle in self.toggles)
        max_bottom = max(max_bottom, max(input_control.rect.bottom for input_control in self.inputs))
        max_bottom = max(max_bottom, max(button.rect.bottom for button in self.control_buttons))
        return {
            "x": min_x - 14,
            "y": min_y - 18,
            "w": (max_right - min_x) + 28,
            "h": (max_bottom - min_y) + 32,
            "title": "Controls",
        }

    def _build_control_sections(self):
        base_x = self.speed_slider.track_rect.x
        return [
            {"title": "Pace", "x": base_x, "y": self.speed_slider.track_rect.y - 36},
            {"title": "Rewards", "x": base_x, "y": self.food_reward_slider.track_rect.y - 36},
            {"title": "Session", "x": base_x, "y": self.show_arrows_toggle.rect.y - 18},
            {"title": "Compute", "x": base_x, "y": self.fast_mode_toggle.rect.y - 18},
            {"title": "History", "x": base_x, "y": self.show_scores_toggle.rect.y - 18},
        ]

    def _build_control_buttons(self):
        buttons = []
        if not self.started:
            buttons.append(
                self.start_button.draw_data(
                    active=True,
                    style="start",
                )
            )

        buttons.append(
            self.cpu_device_button.draw_data(
                active=self.selected_device_preference == "cpu",
                style="device",
            )
        )
        buttons.append(
            self.gpu_device_button.draw_data(
                active=self.selected_device_preference == "cuda",
                disabled=not self.cuda_available,
                style="device",
            )
        )
        return buttons

    def _format_metric(self, value):
        if value is None:
            return "warming up"
        return f"{value:.4f}"
