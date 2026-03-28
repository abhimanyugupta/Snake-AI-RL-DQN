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
        self.last_valid_value = str(value)
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
        try:
            if raw_value:
                value = int(raw_value)
            elif hasattr(self, 'last_valid_value') and self.last_valid_value.strip():
                value = int(self.last_valid_value)
            else:
                value = default
        except ValueError:
            value = default

        clamped_value = max(minimum, min(maximum, value))
        
        if not self.active:
            self.last_valid_value = str(clamped_value)
            self.text = str(clamped_value)
            
        return clamped_value

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
        initial_device_preference="cpu",
        cuda_available=False,
        require_manual_start=False,
        initial_fast_mode=False,
        initial_trainer_mode="single",
        initial_parallel_envs=8,
        initial_stall_threshold=150,
    ):
        initial_reward_config = initial_reward_config or {
            "food": 10.0,
            "death": -10.0,
            "step": 0.0,
        }

        padding = 20
        self.metrics_row_h = 28
        start_y = padding + 46 + 42 + 52 + (9 * self.metrics_row_h) + 20
        col_w = (game.sidebar_width - (padding * 2) - 20) // 2
        col_x = game.board_w + padding
        slider_width = col_w
        toggle_w = (col_w - 10) // 2
        dock_padding = 18
        dock_gap = 12
        dock_y = game.board_h + 14
        dock_h = max(340, game.logical_h - dock_y - 16)
        dock_w = max(260, game.board_w - dock_padding * 2)
        dock_control_h = 176
        
        self.bottom_dock_rect = pygame.Rect(dock_padding, dock_y, dock_w, dock_h)
        self.bottom_control_rect = pygame.Rect(dock_padding, dock_y, dock_w, dock_control_h)
        self.bottom_loss_rect = pygame.Rect(
            dock_padding,
            self.bottom_control_rect.bottom + dock_gap,
            dock_w,
            dock_h - dock_control_h - dock_gap,
        )
        dock_inner_x = self.bottom_control_rect.x + 12
        dock_inner_w = self.bottom_control_rect.width - 24
        dock_col_w = (dock_inner_w - 10) // 2
        dock_row_y = self.bottom_control_rect.y + 18

        self.view_mode = "overview"
        self.view_order = ["overview", "network", "algorithm"]
        self.cuda_available = bool(cuda_available)
        self.selected_trainer_mode = (
            "parallel" if str(initial_trainer_mode).lower() == "parallel" else "single"
        )
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
        self.results_button = ButtonControl("Results [R]", col_x + 470, padding + 44, 128, 30, "results")
        self.view_buttons = [
            self.overview_button,
            self.network_button,
            self.algorithm_button,
            self.results_button,
        ]

        y = start_y
        
        y += 18
        speed_ratio = self._speed_ratio_from_settings(initial_speed, initial_delay_ms)
        self.speed_slider = SliderControl("Training speed", 0.0, 1.0, speed_ratio, col_x, y, slider_width)
        y += 40
        
        loss_btn_w = (slider_width - 12) // 3
        self.show_scores_toggle = ToggleControl("Scores [S]", True, col_x, y, loss_btn_w, 24)
        self.show_avg_toggle = ToggleControl("Avg [M]", True, col_x + loss_btn_w + 6, y, loss_btn_w, 24)
        self.show_best_toggle = ToggleControl("Best [B]", True, col_x + (loss_btn_w + 6) * 2, y, loss_btn_w, 24)
        y += 32
        
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
        y += 32
        self.turbo_toggle = ToggleControl("Turbo [T]", False, col_x, y, toggle_w, 26)
        self.keep_open_toggle = ToggleControl("Keep open [K]", True, col_x + toggle_w + 10, y, toggle_w, 26)
        
        self.episode_input = TextInputControl(
            "Episode goal",
            initial_episode_goal,
            dock_inner_x,
            dock_row_y,
            dock_col_w,
            26,
        )
        self.parallel_env_input = TextInputControl(
            "Parallel envs",
            initial_parallel_envs,
            dock_inner_x + dock_col_w + 10,
            dock_row_y,
            dock_col_w,
            26,
        )
        self.stall_threshold_input = TextInputControl(
            "Stall threshold",
            initial_stall_threshold,
            dock_inner_x,
            dock_row_y + 50,
            dock_inner_w,
            22,
            max_length=4,
        )
        y += 32

        dock_row_y += 78
        self.fast_mode_toggle = ToggleControl(
            "Fast Mode [X]",
            bool(initial_fast_mode),
            dock_inner_x,
            dock_row_y,
            dock_inner_w,
            22,
        )
        y += 32

        dock_row_y += 26
        self.cpu_device_button = ButtonControl("CPU [C]", dock_inner_x, dock_row_y, dock_col_w, 22, "device_cpu")
        self.gpu_device_button = ButtonControl("GPU [U]", dock_inner_x + dock_col_w + 10, dock_row_y, dock_col_w, 22, "device_cuda")
        y += 32

        dock_row_y += 26
        self.single_trainer_button = ButtonControl("Single [J]", dock_inner_x, dock_row_y, dock_col_w, 22, "trainer_single")
        self.parallel_trainer_button = ButtonControl("Parallel [P]", dock_inner_x + dock_col_w + 10, dock_row_y, dock_col_w, 22, "trainer_parallel")
        y += 32

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
            self.fast_mode_toggle,
            self.show_scores_toggle,
            self.show_avg_toggle,
            self.show_best_toggle,
        ]
        self.inputs = [self.episode_input, self.parallel_env_input, self.stall_threshold_input]
        self.control_buttons = [
            self.start_button,
            self.cpu_device_button,
            self.gpu_device_button,
            self.single_trainer_button,
            self.parallel_trainer_button,
        ]

        self.initial_episode_goal = initial_episode_goal
        self.initial_parallel_envs = max(1, int(initial_parallel_envs))
        self.initial_stall_threshold = max(1, int(initial_stall_threshold))
        self.last_reward = 0.0
        self.deep_scores = []
        self.deep_average_history = []
        self.deep_best_history = []
        self.deep_episode_rewards = []
        self.deep_loss_history = []
        self.deep_loss_average_history = []
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
        self.results_graph_rects = []
        self.results_slider_rect = None
        self.results_slider_drag_active = False
        self.results_hover_index = None
        self.baseline_visible = True
        self.pending_trainer_mode = None
        self.results_ready = False

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

    def get_parallel_env_count(self):
        return self.parallel_env_input.get_int(default=self.initial_parallel_envs, minimum=1, maximum=256)

    def get_stall_threshold(self):
        return self.stall_threshold_input.get_int(
            default=self.initial_stall_threshold,
            minimum=1,
            maximum=9999,
        )

    def available_view_order(self):
        order = ["overview", "network", "algorithm"]
        if self.results_ready:
            order.append("results")
        return order

    def visible_view_buttons(self):
        buttons = [
            self.overview_button.draw_data(self.view_mode == "overview"),
            self.network_button.draw_data(self.view_mode == "network"),
            self.algorithm_button.draw_data(self.view_mode == "algorithm"),
        ]
        if self.results_ready:
            buttons.append(self.results_button.draw_data(self.view_mode == "results"))
        return buttons

    def active_view_button_controls(self):
        buttons = [self.overview_button, self.network_button, self.algorithm_button]
        if self.results_ready:
            buttons.append(self.results_button)
        return buttons

    def set_results_ready(self, ready=True, auto_focus=False):
        self.results_ready = bool(ready)
        if not self.results_ready and self.view_mode == "results":
            self.view_mode = "overview"
        elif self.results_ready and auto_focus:
            self.view_mode = "results"

    def export_deep_history(self):
        return {
            "scores": list(self.deep_scores),
            "moving_avg": list(self.deep_average_history),
            "best_scores": list(self.deep_best_history),
            "episode_rewards": list(self.deep_episode_rewards),
            "losses": list(self.deep_loss_history),
            "loss_moving_avg": list(self.deep_loss_average_history),
        }

    def export_baseline_history(self):
        return {
            "scores": list(self.baseline_scores),
            "moving_avg": list(self.baseline_average_history),
            "best_scores": list(self.baseline_best_history),
            "episode_rewards": list(self.baseline_episode_rewards),
        }

    def queue_or_set_trainer_mode(self, requested_mode):
        normalized = "parallel" if str(requested_mode).lower() == "parallel" else "single"
        if not self.started:
            self.selected_trainer_mode = normalized
            self.pending_trainer_mode = None
            return

        if normalized == self.selected_trainer_mode:
            self.pending_trainer_mode = None
            return

        if self.pending_trainer_mode == normalized:
            self.pending_trainer_mode = None
            return

        self.pending_trainer_mode = normalized

    def consume_pending_trainer_mode(self):
        pending = self.pending_trainer_mode
        if pending and pending != self.selected_trainer_mode:
            self.pending_trainer_mode = None
            self.selected_trainer_mode = pending
            return pending
        self.pending_trainer_mode = None
        return None

    def should_draw_frame(self, step_number, force=False):
        if force or not self.turbo_toggle.value:
            return True
        return step_number <= 1 or (step_number % self.render_every_n_steps == 0)

    def load_deep_history(self, history):
        self.deep_scores = list(history.get("scores", []))
        self.deep_average_history = list(history.get("moving_avg", []))
        self.deep_best_history = list(history.get("best_scores", []))
        self.deep_episode_rewards = list(history.get("episode_rewards", []))
        self.deep_loss_history = [float(value) for value in history.get("losses", [])]
        self.deep_loss_average_history = [
            float(value) for value in history.get("loss_moving_avg", [])
        ]

    def set_baseline_history(self, history):
        self.baseline_scores = list(history.get("scores", []))
        self.baseline_average_history = list(history.get("moving_avg", []))
        self.baseline_best_history = list(history.get("best_scores", []))
        self.baseline_episode_rewards = list(history.get("episode_rewards", []))

    def set_baseline_visibility(self, visible):
        self.baseline_visible = bool(visible)

    def record_deep_episode(self, score, episode_reward, loss=None):
        self.deep_scores.append(score)
        self.deep_episode_rewards.append(float(episode_reward))
        recent_scores = self.deep_scores[-20:]
        moving_average = sum(recent_scores) / len(recent_scores)
        self.deep_average_history.append(moving_average)
        best_score = max(self.deep_best_history[-1], score) if self.deep_best_history else score
        self.deep_best_history.append(best_score)
        loss_value = 0.0 if loss is None else float(loss)
        self.deep_loss_history.append(loss_value)
        recent_losses = self.deep_loss_history[-20:]
        self.deep_loss_average_history.append(sum(recent_losses) / len(recent_losses))

    def sync_graph_rect(self, game):
        data = game.dashboard_data
        if data and "_graph_rect" in data:
            self.graph_rect = data["_graph_rect"]
        else:
            self.graph_rect = None
        if data and "_results_graph_rects" in data:
            self.results_graph_rects = [rect for rect in data.get("_results_graph_rects", []) if rect]
        else:
            self.results_graph_rects = []
        if data and "_results_slider_rect" in data:
            self.results_slider_rect = data.get("_results_slider_rect")
        else:
            self.results_slider_rect = None

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

    def results_total_points(self):
        return max(
            len(self.deep_scores),
            len(self.deep_average_history),
            len(self.deep_best_history),
            len(self.deep_loss_history),
            len(self.deep_loss_average_history),
        )

    def handle_events(self, events):
        for event in events:
            consumed_by_input = False
            for input_control in self.inputs:
                if input_control.handle_event(event):
                    consumed_by_input = True
                    break

            if consumed_by_input:
                continue

            if self.view_mode == "results":
                if self._handle_results_graph_event(event):
                    continue
            elif self._handle_graph_event(event):
                continue

            for button in self.active_view_button_controls():
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

    def _handle_results_graph_event(self, event):
        total = self.results_total_points()
        plot_rects = [rect for rect in self.results_graph_rects if rect]
        slider_rect = self.results_slider_rect
        if total < 1 or (not plot_rects and slider_rect is None):
            return False

        if self.results_hover_index is None and total > 0:
            self.results_hover_index = total - 1

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if slider_rect and slider_rect.collidepoint(event.pos):
                self.results_slider_drag_active = True
                self._update_results_hover_from_slider(event.pos[0], total)
                return True
            for plot_rect in plot_rects:
                if plot_rect.collidepoint(event.pos):
                    self._update_results_hover_from_plot(plot_rect, event.pos[0], total)
                    return True

        if event.type == pygame.MOUSEMOTION:
            if self.results_slider_drag_active:
                self._update_results_hover_from_slider(event.pos[0], total)
                return True
            for plot_rect in plot_rects:
                if plot_rect.collidepoint(event.pos):
                    self._update_results_hover_from_plot(plot_rect, event.pos[0], total)
                    return True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.results_slider_drag_active:
            self.results_slider_drag_active = False
            self._update_results_hover_from_slider(event.pos[0], total)
            return True

        return False

    def _update_results_hover_from_plot(self, plot_rect, mouse_x, total):
        if total <= 0 or plot_rect.width <= 0:
            return
        ratio = (mouse_x - plot_rect.x) / max(1, plot_rect.width)
        self._set_results_hover_from_ratio(ratio, total)

    def _update_results_hover_from_slider(self, mouse_x, total):
        slider_rect = self.results_slider_rect
        if total <= 0 or slider_rect is None or slider_rect.width <= 0:
            return
        ratio = (mouse_x - slider_rect.x) / max(1, slider_rect.width)
        self._set_results_hover_from_ratio(ratio, total)

    def _set_results_hover_from_ratio(self, ratio, total):
        if total <= 0:
            self.results_hover_index = None
            return
        ratio = max(0.0, min(1.0, float(ratio)))
        self.results_hover_index = int(round(ratio * max(0, total - 1)))

    def _handle_shortcuts(self, key):
        if key == pygame.K_SPACE:
            if self.started:
                self.pause_toggle.toggle()
        elif key == pygame.K_TAB:
            order = self.available_view_order()
            current_view = self.view_mode if self.view_mode in order else "overview"
            current_index = order.index(current_view)
            self.view_mode = order[(current_index + 1) % len(order)]
        elif key == pygame.K_n:
            self.view_mode = "network"
        elif key == pygame.K_e:
            self.view_mode = "algorithm"
        elif key == pygame.K_r and self.results_ready:
            self.view_mode = "results"
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
        elif key == pygame.K_x:
            self.fast_mode_toggle.toggle()
        elif key == pygame.K_c:
            self._handle_control_button("device_cpu")
        elif key == pygame.K_u:
            self._handle_control_button("device_cuda")
        elif key == pygame.K_j:
            self._handle_control_button("trainer_single")
        elif key == pygame.K_p:
            self._handle_control_button("trainer_parallel")
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
            if self.view_mode == "results":
                total = self.results_total_points()
                if total > 0:
                    self.results_hover_index = total - 1
            else:
                total = self.graph_total_points()
                if total > 0:
                    self.graph_view_size = total
                    self.graph_view_end = None

    def _handle_control_button(self, mode_value):
        if mode_value == "start":
            self.started = True
            self.pause_toggle.value = False
        elif mode_value == "results" and self.results_ready:
            self.view_mode = "results"
        elif mode_value == "device_cpu":
            self.selected_device_preference = "cpu"
        elif mode_value == "device_cuda" and self.cuda_available:
            self.selected_device_preference = "cuda"
        elif mode_value == "trainer_single":
            self.queue_or_set_trainer_mode("single")
        elif mode_value == "trainer_parallel":
            self.queue_or_set_trainer_mode("parallel")

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
        if self.view_mode == "results" and not self.results_ready:
            self.view_mode = "overview"
        show_baseline = self.baseline_visible if show_baseline is None else bool(show_baseline)
        trainer_mode = str(context.get("trainer_mode", "single")).strip().lower()
        parallel_envs = int(max(1, context.get("parallel_envs", 1)))
        parallel_phase = str(context.get("parallel_phase", "single")).strip().lower()
        eval_tail_episodes = int(max(1, context.get("eval_tail_episodes", 3)))
        parallel_bulk_mode = trainer_mode == "parallel" and parallel_phase == "bulk"
        include_network = self.view_mode == "network" and not lightweight and not parallel_bulk_mode
        candidate_points = (
            game.get_relative_points() if not lightweight and not parallel_bulk_mode else {}
        )
        deadly_moves = (
            {key: game.is_collision(point) for key, point in candidate_points.items()}
            if candidate_points
            else {}
        )
        q_values = list(action_info.get("q_values", [0.0, 0.0, 0.0]))
        if len(q_values) < len(agent.ACTION_LABELS):
            q_values.extend([0.0] * (len(agent.ACTION_LABELS) - len(q_values)))
        train_info = context["train_info"]
        transition = context.get("transition") or {}
        fast_mode_requested = bool(context.get("fast_mode_requested", False))
        fast_mode_effective = bool(context.get("fast_mode_effective", False))
        fast_tail_episodes = int(context.get("fast_tail_episodes", 5))
        episodes_remaining = int(context.get("episodes_remaining", 0))
        completed_runs = len(self.deep_scores)
        completed_progress = min(completed_runs, max(0, int(episode_goal)))
        exploration = agent.exploration_status()
        exploration_mode = str(exploration.get("mode", "decay")).title()
        exploration_stall = (
            f"{exploration.get('plateau_counter', 0)}/"
            f"{exploration.get('plateau_patience', 0)}"
        )
        exploration_summary = (
            f"{exploration.get('reheat_count', 0)} | cd {exploration.get('cooldown_remaining', 0)}"
        )
        replay_status = agent.replay_status()
        replay_summary = (
            f"{replay_status.get('mode', 'Hybrid')} | "
            f"{replay_status.get('mix', '70U/30P')} | "
            f"n-step {replay_status.get('n_step', 1)} | "
            f"cap {replay_status.get('capacity', 0)} | "
            f"beta {replay_status.get('beta', 0.0):.2f}"
        )
        training_status_lines = self._build_training_status_lines(agent, train_info)

        if lightweight:
            network_view = {
                "message": (
                    "Fast mode hides the live network inspection while training is running. "
                    "Replay one of the newest 3 captured runs after completion to inspect it."
                )
            }
            feature_lines = [
                "Fast mode is hiding live board and dashboard rendering for this run.",
                "The newest 3 completed runs will be replayable after training finishes.",
                f"Episodes remaining in this session: {episodes_remaining}",
                f"Warmup remaining: {train_info.get('warmup_remaining', 0)}",
                f"Replay buffer fill: {train_info.get('buffer_size', 0)}",
                f"Episode steps: {train_info.get('episode_steps', 0)}",
                f"Env steps/sec: {self._format_rate(train_info.get('env_steps_per_sec'))}",
                f"Updates/sec: {self._format_rate(train_info.get('updates_per_sec'))}",
                f"Current device: {agent.device_label}",
            ]
            help_lines = [
                "[accent]Single + Fast Mode is the learn-mode speedup path: training runs hidden, replay comes afterward.",
                "[accent]Episode metrics, score history, checkpoints, and logs still update while rendering stays off.",
            ]
        elif parallel_bulk_mode:
            network_view = {
                "message": (
                    "Parallel bulk training is running aggregate worker updates. "
                    "The live network inspection returns during the rendered evaluation tail."
                )
            }
            feature_lines = [
                "Parallel mode is stepping many Snake environments in batches.",
                f"Parallel env count: {parallel_envs}",
                f"Rendered evaluation tail: last {eval_tail_episodes} run(s)",
                f"Warmup remaining: {train_info.get('warmup_remaining', 0)}",
                f"Replay buffer fill: {train_info.get('buffer_size', 0)}",
                f"Batch size: {train_info.get('batch_size', agent.batch_size)}",
                f"Episode steps: {train_info.get('episode_steps', 0)}",
                f"Env steps/sec: {self._format_rate(train_info.get('env_steps_per_sec'))}",
                f"Updates/sec: {self._format_rate(train_info.get('updates_per_sec'))}",
            ]
            help_lines = [
                "[accent]The board is paused on purpose here so it does not pretend to show one specific run.",
                "[accent]Completed runs are being counted across all parallel workers at the same time.",
                "[accent]Bulk training hides per-step heatmaps to keep throughput high.",
                "[accent]Network and Algorithm pages switch back to live teaching detail during the evaluation tail.",
                f"[accent]Device target: {self.selected_device_preference.upper()} | Active runtime: {agent.device_label}",
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
                    f"Episode steps: {train_info.get('episode_steps', 0)}",
                    f"Env steps/sec: {self._format_rate(train_info.get('env_steps_per_sec'))}",
                    f"Updates/sec: {self._format_rate(train_info.get('updates_per_sec'))}",
                ]
            )
            help_lines = [
                "[accent]Overview: board on the left, current decision on the right, history below.",
                "[accent]Network: live forward pass across the configurable hidden layers.",
                "[accent]Algorithm: Double-DQN target selection, 3-step returns, hybrid replay memory, and target-net update flow.",
                "[accent]Single mode is for learning and inspection; Parallel mode is for speed, especially on GPU.",
                f"[accent]Device target: {self.selected_device_preference.upper()} | Active runtime: {agent.device_label}",
            ]

        if fast_mode_requested and fast_mode_requested != fast_mode_effective:
            help_lines.append("[accent]Fast mode change is queued and will apply next episode.")
        elif fast_mode_effective and not lightweight:
            help_lines.append(
                "[accent]Fast mode is active. Training stays hidden and the newest 3 runs become replayable afterward."
            )
        if self.pending_trainer_mode:
            help_lines.append(
                f"[accent]Queued trainer mode: {self.pending_trainer_mode.capitalize()}. "
                "It will apply at the next safe boundary."
            )

        best_q = max(q_values)
        best_action_index = q_values.index(best_q)
        training_status_lines = self._build_training_status_lines(agent, train_info)
        if parallel_bulk_mode:
            decision_summary = [
                f"Completed runs: {completed_progress}/{max(1, int(episode_goal))}",
                f"Latest finished score: {transition.get('score', 'n/a')}",
                f"Latest reward: {transition.get('reward_text', 'n/a')}",
                f"Parallel envs: {parallel_envs}",
                f"Batch size: {train_info.get('batch_size', agent.batch_size)}",
                f"Env steps/sec: {self._format_rate(train_info.get('env_steps_per_sec'))}",
                f"Updates/sec: {self._format_rate(train_info.get('updates_per_sec'))}",
                f"Replay: {replay_summary}",
                *training_status_lines,
                f"Architecture: {agent.architecture_label}",
            ]
        else:
            decision_summary = [
                f"Chosen move: {action_info['action_label']}",
                f"Selection mode: {action_info['decision_type']}",
                f"Greedy favorite: {agent.ACTION_LABELS[best_action_index]} ({best_q:+.3f})",
                f"Food cue: {agent.explain_food_view(state)}",
                f"Replay: {replay_summary}",
                *training_status_lines,
                f"Architecture: {agent.architecture_label}",
            ]
        if lightweight:
            decision_summary.append("Visualization: hidden fast training")
        elif parallel_bulk_mode:
            decision_summary.append("Visualization: aggregate parallel bulk status")
        elif fast_mode_effective:
            decision_summary.append("Visualization: hidden training with replay afterward")

        if parallel_bulk_mode:
            metrics = [
                ("Completed runs", f"{completed_progress}/{max(1, int(episode_goal))}"),
                ("Mode", context.get("mode_label", "Parallel bulk")),
                ("Last finished score", transition.get("score", "n/a")),
                ("Best score", best_score),
                ("Last reward", transition.get("reward_text", "n/a")),
                ("Epsilon", f"{agent.epsilon:.3f}"),
                ("Explore", exploration_mode),
                ("Stall", exploration_stall),
                ("Env steps/sec", self._format_rate(train_info.get("env_steps_per_sec"))),
                ("Updates/sec", self._format_rate(train_info.get("updates_per_sec"))),
                ("Buffer fill", train_info.get("buffer_size", 0)),
                ("Batch size", train_info.get("batch_size", agent.batch_size)),
                ("Parallel envs", parallel_envs),
                ("Rendered later", f"Last {eval_tail_episodes} run(s)"),
                ("Reheats", exploration.get("reheat_count", 0)),
                ("Device", agent.device_label),
                ("Architecture", agent.architecture_label),
            ]
        else:
            metrics = [
                ("Run", f"{current_game_number}/{max(episode_goal, current_game_number)}"),
                ("Mode", context.get("mode_label", "Training")),
                ("Score", game.score),
                ("Best score", best_score),
                ("Episode reward", f"{context.get('episode_reward', 0.0):+.2f}"),
                ("Epsilon", f"{agent.epsilon:.3f}"),
                ("Explore", exploration_mode),
                ("Stall", exploration_stall),
                ("Loss", self._format_metric(train_info.get("loss"))),
                ("TD error", self._format_metric(train_info.get("td_error"))),
                ("Buffer fill", train_info.get("buffer_size", 0)),
                ("Warmup left", train_info.get("warmup_remaining", 0)),
                ("Target sync in", train_info.get("target_sync_remaining", "-")),
                ("Reheats", exploration_summary),
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
                "thickness": 1,
            },
            {
                "label": "Deep avg",
                "values": self.deep_average_history,
                "visible": self.show_avg_toggle.value,
                "color": (80, 230, 120),
                "thickness": 2,
            },
            {
                "label": "Deep best",
                "values": self.deep_best_history,
                "visible": self.show_best_toggle.value,
                "color": (255, 196, 68),
                "thickness": 1,
            },
            {
                "label": "Tabular score",
                "values": self.baseline_scores,
                "visible": show_baseline and self.show_scores_toggle.value and bool(self.baseline_scores),
                "color": (110, 140, 180),
                "thickness": 1,
            },
            {
                "label": "Tabular avg",
                "values": self.baseline_average_history,
                "visible": show_baseline and self.show_avg_toggle.value and bool(self.baseline_average_history),
                "color": (130, 190, 150),
                "thickness": 1,
            },
            {
                "label": "Tabular best",
                "values": self.baseline_best_history,
                "visible": show_baseline and self.show_best_toggle.value and bool(self.baseline_best_history),
                "color": (200, 165, 100),
                "thickness": 1,
            },
        ]
        loss_graph_series = [
            {
                "label": "Loss",
                "values": self.deep_loss_history,
                "visible": bool(self.deep_loss_history),
                "color": (255, 132, 86),
                "thickness": 1,
            },
            {
                "label": "Loss avg",
                "values": self.deep_loss_average_history,
                "visible": bool(self.deep_loss_average_history),
                "color": (255, 214, 110),
                "thickness": 2,
            },
        ]
        results_score_series = [
            {
                "label": "Score",
                "values": list(self.deep_scores),
                "visible": True,
                "color": (80, 190, 255),
                "thickness": 1,
            },
            {
                "label": "Avg (20)",
                "values": list(self.deep_average_history),
                "visible": True,
                "color": (80, 230, 120),
                "thickness": 2,
            },
            {
                "label": "Best",
                "values": list(self.deep_best_history),
                "visible": True,
                "color": (255, 196, 68),
                "thickness": 1,
            },
        ]
        results_loss_series = [
            {
                "label": "Loss",
                "values": list(self.deep_loss_history),
                "visible": True,
                "color": (255, 132, 86),
                "thickness": 1,
            },
            {
                "label": "Loss avg",
                "values": list(self.deep_loss_average_history),
                "visible": True,
                "color": (255, 214, 110),
                "thickness": 2,
            },
        ]
        results_total = max(
            len(self.deep_scores),
            len(self.deep_average_history),
            len(self.deep_best_history),
            len(self.deep_loss_history),
            len(self.deep_loss_average_history),
        )
        if results_total > 0:
            if self.results_hover_index is None:
                self.results_hover_index = results_total - 1
            else:
                self.results_hover_index = max(0, min(results_total - 1, int(self.results_hover_index)))
        else:
            self.results_hover_index = None
        results_summary_lines = [
            f"Completed runs: {completed_runs}",
            f"Best score: {max(self.deep_best_history) if self.deep_best_history else 0}",
            f"Final epsilon: {agent.epsilon:.3f}",
            (
                f"Exploration: {exploration_mode} | reheats {exploration.get('reheat_count', 0)}"
                f" | stall {exploration.get('plateau_counter', 0)}/{exploration.get('plateau_patience', 0)}"
                f" | cooldown {exploration.get('cooldown_remaining', 0)}"
            ),
            f"Device: {agent.device_label}",
            (
                "Trainer mode: Single (learn/inspect)"
                if trainer_mode == "single"
                else "Trainer mode: Parallel (speed/throughput)"
            ),
            f"Parallel envs: {parallel_envs}" if trainer_mode == "parallel" else "Parallel envs: n/a",
            f"Replay: {replay_summary}",
            *training_status_lines,
            *training_status_lines,
            *training_status_lines,
            *training_status_lines,
            f"n-step returns: {replay_status.get('n_step', 1)}",
            f"Architecture: {agent.architecture_label}",
            f"State features: {len(agent.STATE_LABELS)}",
            f"Latest loss: {self._format_metric(train_info.get('loss'))}",
        ]

        comparison_lines = self._build_comparison_lines(
            show_baseline=show_baseline,
            fast_mode_effective=fast_mode_effective,
            fast_tail_episodes=fast_tail_episodes,
            trainer_mode=trainer_mode,
            parallel_phase=parallel_phase,
            eval_tail_episodes=eval_tail_episodes,
        )
        algorithm_sections = self._build_algorithm_sections(agent, action_info, context)
        control_sections = self._build_control_sections()
        bottom_dock = self._build_bottom_dock_layout()
        board_mode = "parallel_bulk" if parallel_bulk_mode else "snake"
        board_panel = {}
        if parallel_bulk_mode:
            board_panel = {
                "title": "Parallel Bulk Training",
                "subtitle": f"{parallel_envs} environments are training at once.",
                "progress_ratio": completed_progress / max(1, int(episode_goal)),
                "progress_label": f"Completed runs {completed_progress}/{max(1, int(episode_goal))}",
                "lines": [
                    "This is aggregate progress, not one live run.",
                    "Many workers can finish between screen refreshes, so the count can jump quickly.",
                    "The training graph on the right is the authoritative live history.",
                    "Real snake animation returns during the evaluation tail and in post-run replays.",
                    f"Env steps/sec: {self._format_rate(train_info.get('env_steps_per_sec'))}",
                    f"Updates/sec: {self._format_rate(train_info.get('updates_per_sec'))}",
                ],
            }

        return {
            "panel_title": "Snake Deep RL Lab",
            "view_mode": self.view_mode,
            "metrics_row_h": self.metrics_row_h,
            "view_buttons": self.visible_view_buttons(),
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
                if (self.started or toggle is not self.pause_toggle)
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
            "control_sections": control_sections,
            "bottom_dock": bottom_dock,
            "loss_graph_series": loss_graph_series,
            "results_ready": self.results_ready,
            "results_score_series": results_score_series,
            "results_loss_series": results_loss_series,
            "results_summary_lines": results_summary_lines,
            "results_hover_index": self.results_hover_index,
            "board_mode": board_mode,
            "board_panel": board_panel,
            "decision_card_title": "Bulk Throughput" if parallel_bulk_mode else "Decision & Q-Values",
            "decision_card_mode": "bulk" if parallel_bulk_mode else "q_values",
        }

    def _build_comparison_lines(
        self,
        show_baseline,
        fast_mode_effective,
        fast_tail_episodes,
        *,
        trainer_mode="single",
        parallel_phase="single",
        eval_tail_episodes=3,
    ):
        if trainer_mode == "parallel":
            if parallel_phase == "eval":
                return [
                    "Parallel bulk training has finished and the short evaluation tail is being rendered.",
                    f"The final {eval_tail_episodes} evaluation run(s) are what become replayable after training.",
                    "The graph still updates on completed episodes instead of every worker step.",
                ]
            return [
                "Parallel mode is batching multiple environments for higher throughput.",
                "The graph updates on completed episodes, while the board switches to an aggregate status panel during bulk training.",
                f"The final {eval_tail_episodes} evaluation run(s) will be rendered and replayable.",
            ]

        if not show_baseline:
            if fast_mode_effective:
                return [
                    "Fast mode is hiding the live tabular comparison while training runs hidden.",
                    "Use the results screen and replayable last 3 runs after completion to inspect behavior.",
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
        trainer_mode = str(context.get("trainer_mode", "single")).strip().lower()
        parallel_phase = str(context.get("parallel_phase", "single")).strip().lower()
        parallel_envs = int(max(1, context.get("parallel_envs", 1)))
        eval_tail_episodes = int(max(1, context.get("eval_tail_episodes", 3)))
        replay_status = agent.replay_status()

        if trainer_mode == "parallel" and parallel_phase == "bulk":
            return [
                {
                    "title": "Parallel Bulk Training",
                    "lines": [
                        f"{parallel_envs} headless Snake environments are being stepped in parallel.",
                        "One batched policy forward pass chooses actions for all active workers.",
                        "Transitions are added to the replay buffer in batches before larger updates run.",
                    ],
                },
                {
                    "title": "Throughput Focus",
                    "lines": [
                        f"Batch size: {train_info.get('batch_size', agent.batch_size)}",
                        f"Env steps/sec: {self._format_rate(train_info.get('env_steps_per_sec'))}",
                        f"Updates/sec: {self._format_rate(train_info.get('updates_per_sec'))}",
                        "Network heatmaps and detailed algorithm cards are paused during bulk training.",
                    ],
                },
                {
                    "title": "Evaluation Tail",
                    "lines": [
                        f"The final {eval_tail_episodes} run(s) are rendered after bulk training completes.",
                        "Those runs use the learned policy and become replayable in the Recent Replays card.",
                        "This keeps the teaching UI without slowing down the main throughput path.",
                    ],
                },
                {
                    "title": "RL Update",
                    "lines": [
                        f"Replay buffer size: {train_info.get('buffer_size', 0)}",
                        f"Replay mode: {replay_status.get('mode', 'Hybrid')} {replay_status.get('mix', '70U/30P')} | beta {replay_status.get('beta', 0.0):.2f}",
                        f"n-step returns: {replay_status.get('n_step', 1)}",
                        f"Loss: {self._format_metric(train_info.get('loss'))}",
                        f"TD error: {self._format_metric(train_info.get('td_error'))}",
                        f"Epsilon: {agent.epsilon:.3f}",
                    ],
                },
            ]

        transition = context.get("transition") or {}

        return [
            {
                "title": "RL Loop",
                "lines": [
                    f"1. Observe {len(agent.STATE_LABELS)} encoded features, including local cues plus projected move safety.",
                    f"2. Choose {action_info['action_label']} with {action_info['decision_type']}.",
                    f"3. Aggregate {replay_status.get('n_step', 1)}-step returns before storing the transition in hybrid replay memory.",
                    "4. Sample a mixed 70U/30P batch, pick next actions with the policy net, evaluate them with the target net, then backpropagate the n-step target.",
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
                "title": "Double DQN + N-Step Update",
                "lines": [
                    f"Predicted Q: {self._format_metric(train_info.get('example_predicted_q'))}",
                    f"Target Q: {self._format_metric(train_info.get('example_target_q'))}",
                    f"TD error: {self._format_metric(train_info.get('example_td_error'))}",
                    f"Loss: {self._format_metric(train_info.get('loss'))}",
                    f"Replay beta: {replay_status.get('beta', 0.0):.2f}",
                    f"N-step horizon: {replay_status.get('n_step', 1)}",
                ],
            },
            {
                "title": "Why It Helps",
                "lines": [
                    "Hybrid replay keeps some surprise-driven sampling without letting noisy transitions dominate every batch.",
                    "Double DQN reduces over-optimistic targets by splitting action choice from action evaluation.",
                    f"{replay_status.get('n_step', 1)}-step returns push food-seeking and trap-avoidance credit back across several moves instead of only one step.",
                    "Projected move features add reachable-area, tail-reachability, and local-exit context before the snake commits to a turn.",
                    f"Epsilon is {agent.epsilon:.3f}, so the agent still explores when needed.",
                    f"The current MLP architecture is {agent.architecture_label}.",
                ],
            },
        ]

    def _build_bottom_dock_layout(self):
        return {
            "controls_rect": {
                "x": self.bottom_control_rect.x,
                "y": self.bottom_control_rect.y,
                "w": self.bottom_control_rect.width,
                "h": self.bottom_control_rect.height,
                "title": "Run Setup",
            },
            "loss_rect": {
                "x": self.bottom_loss_rect.x,
                "y": self.bottom_loss_rect.y,
                "w": self.bottom_loss_rect.width,
                "h": self.bottom_loss_rect.height,
                "title": "Loss Trend",
                "toggle_strip_bottom": self.show_scores_toggle.rect.bottom + 10,
            },
        }

    def _build_control_sections(self):
        base_x = self.speed_slider.track_rect.x
        return [
            {"title": "Pace", "x": base_x, "y": self.speed_slider.track_rect.y - 36},
            {"title": "Rewards", "x": base_x, "y": self.food_reward_slider.track_rect.y - 36},
            {"title": "Session", "x": base_x, "y": self.show_arrows_toggle.rect.y - 18},
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

        single_label = "Single [J]"
        parallel_label = "Parallel [P]"
        single_active = self.selected_trainer_mode == "single"
        parallel_active = self.selected_trainer_mode == "parallel"
        single_style = "default"
        parallel_style = "default"
        if self.started:
            if self.selected_trainer_mode == "single":
                single_label = "Single live [J]"
            else:
                parallel_label = "Parallel live [P]"
            if self.pending_trainer_mode == "single":
                single_label = "Single queued [J]"
                single_style = "queued"
            elif self.pending_trainer_mode == "parallel":
                parallel_label = "Parallel queued [P]"
                parallel_style = "queued"

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
        buttons.append(
            self.single_trainer_button.draw_data(
                active=single_active,
                style=single_style,
                label=single_label,
            )
        )
        buttons.append(
            self.parallel_trainer_button.draw_data(
                active=parallel_active,
                style=parallel_style,
                label=parallel_label,
            )
        )
        return buttons

    def _build_training_status_lines(self, agent, train_info):
        evaluation_status = agent.evaluation_status()
        current_lr = self._format_learning_rate(train_info.get("current_lr", agent.current_lr))
        latest_eval_avg = self._format_metric(evaluation_status.get("latest_eval_avg"))
        best_eval_avg = self._format_metric(evaluation_status.get("best_eval_avg"))
        target_update_mode = str(
            train_info.get("target_update_mode", agent.target_update_mode)
        ).strip().lower() or "soft"
        update_every_transitions = int(
            train_info.get("update_every_transitions", agent.update_every_transitions)
        )
        gradient_steps_per_update = int(
            train_info.get("gradient_steps_per_update", agent.gradient_steps_per_update)
        )
        update_label = "update" if gradient_steps_per_update == 1 else "updates"
        return [
            (
                f"Training status: LR {current_lr} | latest eval avg {latest_eval_avg} | "
                f"best eval avg {best_eval_avg}"
            ),
            (
                f"Target update: {target_update_mode} | replay ratio: "
                f"{update_every_transitions}T -> {gradient_steps_per_update} {update_label}"
            ),
        ]

    def _build_training_status_lines(self, agent, train_info):
        evaluation_status = agent.evaluation_status()
        current_lr = self._format_learning_rate(train_info.get("current_lr", agent.current_lr))
        latest_eval_avg = self._format_metric(evaluation_status.get("latest_eval_avg"))
        best_eval_avg = self._format_metric(evaluation_status.get("best_eval_avg"))
        target_update_mode = str(
            train_info.get("target_update_mode", agent.target_update_mode)
        ).strip().lower() or "soft"
        update_every_transitions = int(
            train_info.get("update_every_transitions", agent.update_every_transitions)
        )
        gradient_steps_per_update = int(
            train_info.get("gradient_steps_per_update", agent.gradient_steps_per_update)
        )
        update_label = "update" if gradient_steps_per_update == 1 else "updates"
        return [
            (
                f"Training status: LR {current_lr} | latest eval avg {latest_eval_avg} | "
                f"best eval avg {best_eval_avg}"
            ),
            (
                f"Target update: {target_update_mode} | replay ratio: "
                f"{update_every_transitions}T -> {gradient_steps_per_update} {update_label}"
            ),
        ]

    def _format_learning_rate(self, value):
        if value is None:
            return "warming up"
        return f"{float(value):.2e}"

    def _format_metric(self, value):
        if value is None:
            return "warming up"
        return f"{value:.4f}"

    def _format_rate(self, value):
        if value is None:
            return "0.0"
        return f"{float(value):.1f}"
