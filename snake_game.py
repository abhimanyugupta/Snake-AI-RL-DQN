import math
import os
import random
from dataclasses import dataclass
from enum import Enum

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

try:
    import pygame
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Missing dependency 'pygame'. Install requirements.txt before running the Deep RL project."
    ) from exc


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


@dataclass(frozen=True)
class Point:
    x: int
    y: int


def get_available_display_area():
    if os.name == "nt":
        try:
            work_area = wintypes.RECT()
            success = ctypes.windll.user32.SystemParametersInfoW(
                0x0030,
                0,
                ctypes.byref(work_area),
                0,
            )
            if success:
                return (
                    max(640, work_area.right - work_area.left),
                    max(720, work_area.bottom - work_area.top),
                )
        except Exception:
            pass

    desktop_sizes = pygame.display.get_desktop_sizes()
    if desktop_sizes:
        width, height = desktop_sizes[0]
        return max(640, int(width)), max(720, int(height) - 40)

    info = pygame.display.Info()
    return max(640, int(info.current_w)), max(720, int(info.current_h) - 40)


class SnakeLogicEnv:
    """Headless snake environment with no pygame dependency in its runtime path."""

    def __init__(self, w=640, h=560, block_size=20, speed=0):
        self.board_w = w
        self.board_h = h
        self.w = w
        self.h = h
        self.block_size = block_size
        self.speed = speed
        self.render = False
        self.sidebar_width = 0
        self.logical_w = self.board_w
        self.logical_h = self.board_h
        self.window_w = self.logical_w
        self.window_h = self.logical_h
        self.quit_requested = False
        self.dashboard_data = {}
        self.reward_config = {"food": 10.0, "death": -10.0, "step": 0.0}
        self.clock = None
        self.display = None
        self.title_font = None
        self.font = None
        self.small_font = None
        self.tiny_font = None
        self._overlay_surface = None
        self._glow_surface = None
        self._real_display = None
        self.reset()

    def scale_events(self, events):
        return events

    def handle_system_events(self, events):
        return None

    def draw(self):
        return None

    def close(self):
        return None

    def reset(self):
        """Reset the game so training can start a fresh episode."""
        start_x = (self.board_w // 2 // self.block_size) * self.block_size
        start_y = (self.board_h // 2 // self.block_size) * self.block_size

        self.direction = Direction.RIGHT
        self.head = Point(start_x, start_y)
        self.snake = [
            self.head,
            Point(start_x - self.block_size, start_y),
            Point(start_x - (2 * self.block_size), start_y),
        ]
        self.snake_body_set = set(self.snake[1:])

        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self._place_food()

    def set_dashboard_data(self, data):
        self.dashboard_data = dict(data or {})

    def set_reward_config(self, reward_config):
        self.reward_config = {
            "food": float(reward_config.get("food", 10.0)),
            "death": float(reward_config.get("death", -10.0)),
            "step": float(reward_config.get("step", 0.0)),
        }

    def get_relative_points(self):
        return {
            "straight": self._point_from_direction(self.head, self.direction),
            "right": self._point_from_direction(self.head, self._turn_right(self.direction)),
            "left": self._point_from_direction(self.head, self._turn_left(self.direction)),
        }

    def _place_food(self):
        """Place food on a free grid cell."""
        max_x = (self.board_w - self.block_size) // self.block_size
        max_y = (self.board_h - self.block_size) // self.block_size

        while True:
            x = random.randint(0, max_x) * self.block_size
            y = random.randint(0, max_y) * self.block_size
            self.food = Point(x, y)
            if self.food != self.head and self.food not in self.snake_body_set:
                break

    def play_step(self, action=None, events=None, draw_frame=True, apply_pacing=True):
        """
        Advance the game by one frame.

        - If action is None, the game uses keyboard input when supported by the subclass.
        - If action is [1, 0, 0], [0, 1, 0], or [0, 0, 1], the snake is
          controlled by the agent.
        """
        if self.quit_requested:
            return 0.0, True, self.score

        self.frame_iteration += 1
        events = events or []
        self.handle_system_events(events)
        if self.quit_requested:
            return 0.0, True, self.score

        if action is None:
            self._handle_human_input(events)
            self._move()
        else:
            self._move(action)

        previous_head = self.snake[0]
        self.snake.insert(0, self.head)
        self.snake_body_set.add(previous_head)
        reward = self.reward_config["step"]
        game_over = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            reward = self.reward_config["death"]
            game_over = True
        elif self.head == self.food:
            self.score += 1
            reward = self.reward_config["food"]
            self._place_food()
        else:
            tail = self.snake.pop()
            self.snake_body_set.discard(tail)

        if self.render and draw_frame:
            self.draw()

        if apply_pacing and self.speed > 0 and self.clock is not None:
            self.clock.tick(self.speed)
        return reward, game_over, self.score

    def is_collision(self, point=None):
        """Check whether a point hits a wall or the snake body."""
        if point is None:
            point = self.head

        if self._is_wall_collision(point):
            return True

        if point in self.snake_body_set:
            return True

        return False

    def raycast_free_steps(self, start, direction):
        steps = 0
        current = start
        while True:
            current = self._point_from_direction(current, direction)
            if self._is_wall_collision(current) or current in self.snake_body_set:
                return steps
            steps += 1

    def _is_wall_collision(self, point):
        return (
            point.x < 0
            or point.x >= self.board_w
            or point.y < 0
            or point.y >= self.board_h
        )

    def _handle_human_input(self, events):
        return None

    def _move(self, action=None):
        """
        Move the snake.

        The agent action is a one-hot list:
        [1, 0, 0] = keep going straight
        [0, 1, 0] = turn right
        [0, 0, 1] = turn left
        """
        if action is not None:
            action = list(action)
            if len(action) != 3 or sum(action) != 1:
                raise ValueError("Action must be a one-hot list like [1, 0, 0].")
            self.direction = self._direction_for_action(action)

        self.head = self._point_from_direction(self.head, self.direction)

    def _direction_for_action(self, action):
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_index = clockwise.index(self.direction)
        action_index = list(action).index(1)

        if action_index == 0:
            return clockwise[current_index]
        if action_index == 1:
            return clockwise[(current_index + 1) % 4]
        return clockwise[(current_index - 1) % 4]

    def _point_from_direction(self, point, direction):
        if direction == Direction.RIGHT:
            return Point(point.x + self.block_size, point.y)
        if direction == Direction.LEFT:
            return Point(point.x - self.block_size, point.y)
        if direction == Direction.UP:
            return Point(point.x, point.y - self.block_size)
        return Point(point.x, point.y + self.block_size)

    def _turn_right(self, direction):
        turns = {
            Direction.RIGHT: Direction.DOWN,
            Direction.DOWN: Direction.LEFT,
            Direction.LEFT: Direction.UP,
            Direction.UP: Direction.RIGHT,
        }
        return turns[direction]

    def _turn_left(self, direction):
        turns = {
            Direction.RIGHT: Direction.UP,
            Direction.UP: Direction.LEFT,
            Direction.LEFT: Direction.DOWN,
            Direction.DOWN: Direction.RIGHT,
        }
        return turns[direction]


class SnakeGameAI(SnakeLogicEnv):
    """Playable snake game that also exposes helper methods for RL training."""

    def __init__(
        self,
        w=640,
        h=560,
        block_size=20,
        speed=12,
        render=True,
        sidebar_width=860,
        window_h=None,
    ):
        super().__init__(w=w, h=h, block_size=block_size, speed=speed)
        self.render = render
        self.sidebar_width = sidebar_width if render else 0
        requested_window_h = window_h if window_h is not None else self.board_h
        requested_window_w = self.board_w + self.sidebar_width

        self.logical_w = requested_window_w
        self.logical_h = max(self.board_h, requested_window_h)
        self.window_w = self.logical_w
        self.window_h = self.logical_h

        pygame.init()
        if render:
            os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        else:
            # Keep the headless wrapper aligned with the board-only logic footprint.
            self.sidebar_width = 0

        if self.render:
            available_w, available_h = get_available_display_area()
            max_window_w = max(self.board_w, available_w - 40)
            max_window_h = max(self.board_h, available_h - 140)
            self.window_w = min(self.logical_w, max_window_w)
            self.window_h = min(self.logical_h, max_window_h)
            self.sidebar_width = max(0, self.logical_w - self.board_w)
            
            self._real_display = pygame.display.set_mode((self.window_w, self.window_h), pygame.RESIZABLE)
            self.display = pygame.Surface((self.logical_w, self.logical_h))
            pygame.display.set_caption("Snake Deep RL Lab")
            self.title_font = pygame.font.SysFont("arial", 26, bold=True)
            self.font = pygame.font.SysFont("arial", 20)
            self.small_font = pygame.font.SysFont("arial", 16)
            self.tiny_font = pygame.font.SysFont("arial", 14)
            # Pre-allocate reusable alpha surfaces to avoid per-frame allocation
            self._overlay_surface = pygame.Surface((self.board_w, self.board_h), pygame.SRCALPHA)
            glow_size = self.block_size * 3
            self._glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
        else:
            self.display = None
            self.title_font = None
            self.font = None
            self.small_font = None
            self.tiny_font = None
            self._overlay_surface = None
            self._glow_surface = None

        self.clock = pygame.time.Clock()

    def scale_events(self, events):
        if not hasattr(self, "_real_display") or self._real_display is None:
            return events
            
        rw, rh = self._real_display.get_size()
        lw, lh = self.logical_w, self.logical_h
        
        scaled_events = []
        for e in events:
            if e.type == pygame.VIDEORESIZE:
                self.window_w, self.window_h = e.size
                scaled_events.append(e)
            elif e.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
                rx, ry = e.pos
                sx, sy = int(rx * lw / rw), int(ry * lh / rh)
                new_dict = dict(e.__dict__)
                new_dict['pos'] = (sx, sy)
                scaled_events.append(pygame.event.Event(e.type, new_dict))
            else:
                scaled_events.append(e)
        return scaled_events

    def reset(self):
        """Reset the game so training can start a fresh episode."""
        start_x = (self.board_w // 2 // self.block_size) * self.block_size
        start_y = (self.board_h // 2 // self.block_size) * self.block_size

        self.direction = Direction.RIGHT
        self.head = Point(start_x, start_y)
        self.snake = [
            self.head,
            Point(start_x - self.block_size, start_y),
            Point(start_x - (2 * self.block_size), start_y),
        ]
        self.snake_body_set = set(self.snake[1:])

        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self._place_food()

    def set_dashboard_data(self, data):
        self.dashboard_data = dict(data or {})

    def set_reward_config(self, reward_config):
        self.reward_config = {
            "food": float(reward_config.get("food", 10.0)),
            "death": float(reward_config.get("death", -10.0)),
            "step": float(reward_config.get("step", 0.0)),
        }

    def handle_system_events(self, events):
        for event in events:
            if event.type == pygame.QUIT:
                self.quit_requested = True

    def draw(self):
        if self.render:
            self._draw_scene()

    def get_relative_points(self):
        return {
            "straight": self._point_from_direction(self.head, self.direction),
            "right": self._point_from_direction(self.head, self._turn_right(self.direction)),
            "left": self._point_from_direction(self.head, self._turn_left(self.direction)),
        }

    def _place_food(self):
        """Place food on a free grid cell."""
        max_x = (self.board_w - self.block_size) // self.block_size
        max_y = (self.board_h - self.block_size) // self.block_size

        while True:
            x = random.randint(0, max_x) * self.block_size
            y = random.randint(0, max_y) * self.block_size
            self.food = Point(x, y)
            if self.food != self.head and self.food not in self.snake_body_set:
                break

    def play_step(self, action=None, events=None, draw_frame=True, apply_pacing=True):
        """
        Advance the game by one frame.

        - If action is None, the game uses keyboard input.
        - If action is [1, 0, 0], [0, 1, 0], or [0, 0, 1], the snake is
          controlled by the agent.
        """
        if self.quit_requested:
            return 0.0, True, self.score

        self.frame_iteration += 1

        if events is None:
            events = pygame.event.get() if self.render else []
            events = self.scale_events(events)

        self.handle_system_events(events)
        if self.quit_requested:
            return 0.0, True, self.score

        if action is None:
            self._handle_human_input(events)
            self._move()
        else:
            self._move(action)

        previous_head = self.snake[0]
        self.snake.insert(0, self.head)
        self.snake_body_set.add(previous_head)
        reward = self.reward_config["step"]
        game_over = False

        # End the game if the snake hits a wall, itself, or loops for too long.
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            reward = self.reward_config["death"]
            game_over = True
        elif self.head == self.food:
            self.score += 1
            reward = self.reward_config["food"]
            self._place_food()
        else:
            tail = self.snake.pop()
            self.snake_body_set.discard(tail)

        if self.render and draw_frame:
            self._draw_scene()

        if apply_pacing and self.speed > 0:
            self.clock.tick(self.speed)
        return reward, game_over, self.score

    def is_collision(self, point=None):
        """Check whether a point hits a wall or the snake body."""
        if point is None:
            point = self.head

        if self._is_wall_collision(point):
            return True

        if point in self.snake_body_set:
            return True

        return False

    def raycast_free_steps(self, start, direction):
        steps = 0
        current = start
        while True:
            current = self._point_from_direction(current, direction)
            if self._is_wall_collision(current) or current in self.snake_body_set:
                return steps
            steps += 1

    def _is_wall_collision(self, point):
        return (
            point.x < 0
            or point.x >= self.board_w
            or point.y < 0
            or point.y >= self.board_h
        )

    def _handle_human_input(self, events):
        """Allow the player to move with arrow keys or WASD."""
        for event in events:
            if event.type != pygame.KEYDOWN:
                continue

            if event.key in (pygame.K_LEFT, pygame.K_a):
                self._change_direction(Direction.LEFT)
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                self._change_direction(Direction.RIGHT)
            elif event.key in (pygame.K_UP, pygame.K_w):
                self._change_direction(Direction.UP)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self._change_direction(Direction.DOWN)

    def _change_direction(self, new_direction):
        opposite = {
            Direction.RIGHT: Direction.LEFT,
            Direction.LEFT: Direction.RIGHT,
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
        }

        if new_direction != opposite[self.direction]:
            self.direction = new_direction

    def _move(self, action=None):
        """
        Move the snake.

        The agent action is a one-hot list:
        [1, 0, 0] = keep going straight
        [0, 1, 0] = turn right
        [0, 0, 1] = turn left
        """
        if action is not None:
            action = list(action)
            if len(action) != 3 or sum(action) != 1:
                raise ValueError("Action must be a one-hot list like [1, 0, 0].")
            self.direction = self._direction_for_action(action)

        self.head = self._point_from_direction(self.head, self.direction)

    def _direction_for_action(self, action):
        clockwise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_index = clockwise.index(self.direction)
        action_index = list(action).index(1)

        if action_index == 0:
            return clockwise[current_index]
        if action_index == 1:
            return clockwise[(current_index + 1) % 4]
        return clockwise[(current_index - 1) % 4]

    def _point_from_direction(self, point, direction):
        if direction == Direction.RIGHT:
            return Point(point.x + self.block_size, point.y)
        if direction == Direction.LEFT:
            return Point(point.x - self.block_size, point.y)
        if direction == Direction.UP:
            return Point(point.x, point.y - self.block_size)
        return Point(point.x, point.y + self.block_size)

    def _turn_right(self, direction):
        turns = {
            Direction.RIGHT: Direction.DOWN,
            Direction.DOWN: Direction.LEFT,
            Direction.LEFT: Direction.UP,
            Direction.UP: Direction.RIGHT,
        }
        return turns[direction]

    def _turn_left(self, direction):
        turns = {
            Direction.RIGHT: Direction.UP,
            Direction.UP: Direction.LEFT,
            Direction.LEFT: Direction.DOWN,
            Direction.DOWN: Direction.RIGHT,
        }
        return turns[direction]

    def _draw_scene(self):
        if not self.render or self.display is None:
            return

        self.display.fill((18, 18, 18))
        self._draw_board_background()
        if self.dashboard_data.get("board_mode") == "parallel_bulk":
            self._draw_parallel_bulk_board(self.dashboard_data.get("board_panel", {}))
        else:
            self._draw_danger_overlays()
            self._draw_snake_and_food()
            self._draw_action_arrows()
        self._draw_sidebar()
        self._draw_overlay_message()
        if hasattr(self, "_real_display") and self._real_display is not None:
            pygame.transform.smoothscale(self.display, self._real_display.get_size(), self._real_display)
        pygame.display.flip()

    def _draw_board_background(self):
        board_rect = pygame.Rect(0, 0, self.board_w, self.board_h)
        # Deep space premium navy/slate background
        pygame.draw.rect(self.display, (16, 18, 24), board_rect)

        # Subtle grid lines
        for x in range(0, self.board_w, self.block_size):
            pygame.draw.line(self.display, (28, 32, 40), (x, 0), (x, self.board_h), 1)
        for y in range(0, self.board_h, self.block_size):
            pygame.draw.line(self.display, (28, 32, 40), (0, y), (self.board_w, y), 1)

        # Premium outer board border
        pygame.draw.rect(self.display, (50, 60, 80), board_rect, width=2)
        pygame.draw.rect(self.display, (35, 45, 60), board_rect.inflate(4, 4), width=2)

    def _draw_snake_and_food(self):
        # --- Draw Premium Food (Pulsing Glow) ---
        pulse = (math.sin(pygame.time.get_ticks() / 200.0) + 1) / 2  # 0 to 1
        food_center = (self.food.x + self.block_size // 2, self.food.y + self.block_size // 2)
        
        # Glow effect (only drawn if not in turbo mode to save FPS)
        toggles = self.dashboard_data.get("toggles", []) if self.dashboard_data else []
        is_turbo = toggles[4].get("value", False) if len(toggles) > 4 else False
        if not is_turbo and self._glow_surface is not None:
            self._glow_surface.fill((0, 0, 0, 0))  # Clear cached surface
            glow_radius = int(self.block_size * 0.8 + pulse * self.block_size * 0.4)
            pygame.draw.circle(self._glow_surface, (255, 60, 60, 40 + int(pulse * 30)), (self.block_size * 1.5, self.block_size * 1.5), glow_radius)
            self.display.blit(self._glow_surface, (self.food.x - self.block_size, self.food.y - self.block_size))

        # Core food apple
        food_rect = pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size)
        pygame.draw.rect(self.display, (255, 70, 70), food_rect, border_radius=self.block_size // 2)
        pygame.draw.rect(self.display, (255, 180, 180), food_rect.inflate(-8, -8), border_radius=self.block_size // 2)

        # --- Draw Premium Snake (Gradients & Joints) ---
        n = len(self.snake)
        for index, part in enumerate(reversed(self.snake)):
            # Reversed so head is drawn last and on top
            true_idx = n - 1 - index
            
            # Gradient: Head is bright neon green/cyan, tail is dark teal
            ratio = true_idx / max(1, n - 1)
            r = int(50 * (1 - ratio) + 20 * ratio)
            g = int(240 * (1 - ratio) + 120 * ratio)
            b = int(140 * (1 - ratio) + 180 * ratio)
            color = (r, g, b)
            
            rect = pygame.Rect(part.x, part.y, self.block_size, self.block_size)
            
            if true_idx == 0:
                # Head
                pygame.draw.rect(self.display, (255, 255, 255), rect, border_radius=6)
                pygame.draw.rect(self.display, color, rect.inflate(-4, -4), border_radius=4)
                
                # Draw Eyes
                eye_color = (20, 20, 30)
                cx, cy = part.x + self.block_size // 2, part.y + self.block_size // 2
                offset = 4
                if self.direction == Direction.RIGHT:
                    pygame.draw.circle(self.display, eye_color, (cx + offset, cy - offset), 2)
                    pygame.draw.circle(self.display, eye_color, (cx + offset, cy + offset), 2)
                elif self.direction == Direction.LEFT:
                    pygame.draw.circle(self.display, eye_color, (cx - offset, cy - offset), 2)
                    pygame.draw.circle(self.display, eye_color, (cx - offset, cy + offset), 2)
                elif self.direction == Direction.UP:
                    pygame.draw.circle(self.display, eye_color, (cx - offset, cy - offset), 2)
                    pygame.draw.circle(self.display, eye_color, (cx + offset, cy - offset), 2)
                elif self.direction == Direction.DOWN:
                    pygame.draw.circle(self.display, eye_color, (cx - offset, cy + offset), 2)
                    pygame.draw.circle(self.display, eye_color, (cx + offset, cy + offset), 2)
                    
            else:
                # Body segment
                pygame.draw.rect(self.display, color, rect.inflate(-2, -2), border_radius=4)
                
            # Connect the joints with a circle for a continuous tube look
            if true_idx > 0:
                prev_part = self.snake[true_idx - 1]
                joint_x = (part.x + prev_part.x) // 2 + self.block_size // 2
                joint_y = (part.y + prev_part.y) // 2 + self.block_size // 2
                
                # Use the color of the segment closer to the head for the joint
                joint_ratio = (true_idx - 0.5) / max(1, n - 1)
                jr = int(50 * (1 - joint_ratio) + 20 * joint_ratio)
                jg = int(240 * (1 - joint_ratio) + 120 * joint_ratio)
                jb = int(140 * (1 - joint_ratio) + 180 * joint_ratio)
                
                pygame.draw.circle(self.display, (jr, jg, jb), (joint_x, joint_y), self.block_size // 2 - 1)

    def _draw_danger_overlays(self):
        data = self.dashboard_data
        if not data or not data.get("show_dangers") or self._overlay_surface is None:
            return

        self._overlay_surface.fill((0, 0, 0, 0))  # Clear cached surface
        candidate_points = data.get("candidate_points", {})
        deadly_moves = data.get("deadly_moves", {})

        for key, point in candidate_points.items():
            if not deadly_moves.get(key):
                continue

            draw_point = self._clamp_point_to_board(point)
            rect = pygame.Rect(draw_point.x, draw_point.y, self.block_size, self.block_size)
            pygame.draw.rect(self._overlay_surface, (220, 70, 70, 130), rect, border_radius=5)
            pygame.draw.rect(self._overlay_surface, (255, 180, 180, 180), rect, width=2, border_radius=5)

            label_surface = self.small_font.render(key[0].upper(), True, (255, 255, 255))
            self._overlay_surface.blit(label_surface, (rect.x + 4, rect.y + 2))

        self.display.blit(self._overlay_surface, (0, 0))

    def _draw_action_arrows(self):
        data = self.dashboard_data
        if not data or not data.get("show_arrows"):
            return

        candidate_points = data.get("candidate_points", {})
        deadly_moves = data.get("deadly_moves", {})
        action_key = data.get("action_key")
        q_values = data.get("q_values", [0.0, 0.0, 0.0])
        decision_type = data.get("decision_type", "")

        start = (self.head.x + self.block_size // 2, self.head.y + self.block_size // 2)
        key_order = ["straight", "right", "left"]
        key_labels = {"straight": "S", "right": "R", "left": "L"}

        for index, key in enumerate(key_order):
            point = candidate_points.get(key)
            if point is None:
                continue

            end_point = self._clamp_point_to_board(point)
            end = (end_point.x + self.block_size // 2, end_point.y + self.block_size // 2)

            color = (110, 160, 255)
            if deadly_moves.get(key):
                color = (255, 90, 90)
            if key == action_key:
                color = (255, 196, 68) if decision_type == "explore" else (95, 236, 124)
            if decision_type == "policy preview" and key == action_key:
                color = (120, 210, 255)

            self._draw_arrow(start, end, color)

            label = f"{key_labels[key]} {q_values[index]:.2f}"
            label_surface = self.tiny_font.render(label, True, color)
            label_rect = label_surface.get_rect(center=(end[0], end[1] - 14))
            self.display.blit(label_surface, label_rect)

    def _draw_arrow(self, start, end, color):
        pygame.draw.line(self.display, color, start, end, 4)

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return

        ux = dx / length
        uy = dy / length
        size = 10
        left = (end[0] - ux * size - uy * size / 2, end[1] - uy * size + ux * size / 2)
        right = (end[0] - ux * size + uy * size / 2, end[1] - uy * size - ux * size / 2)
        pygame.draw.polygon(self.display, color, [end, left, right])

    def _draw_sidebar(self):
        if self.sidebar_width <= 0:
            return

        data = self.dashboard_data or {}
        panel_x = self.board_w
        panel_rect = pygame.Rect(panel_x, 0, self.sidebar_width, self.window_h)

        pygame.draw.rect(self.display, (22, 23, 28), panel_rect)
        pygame.draw.line(self.display, (15, 15, 18), (panel_x - 1, 0), (panel_x - 1, self.window_h), 2)
        pygame.draw.line(self.display, (45, 48, 56), (panel_x, 0), (panel_x, self.window_h), 1)

        padding = 20
        col_gap = 20
        col_w = (self.sidebar_width - (padding * 2) - col_gap) // 2

        left_col = pygame.Rect(panel_x + padding, padding, col_w, self.window_h - padding * 2)
        right_col = pygame.Rect(panel_x + padding + col_w + col_gap, padding, col_w, self.window_h - padding * 2)

        title = data.get("panel_title", "Snake Deep RL Lab")
        title_surface = self.title_font.render(title, True, (250, 252, 255))
        self.display.blit(title_surface, (left_col.x, left_col.y))

        title_h = title_surface.get_height() + 20
        left_col.y += title_h
        left_col.height -= title_h
        right_col.y += title_h
        right_col.height -= title_h

        view_buttons = data.get("view_buttons", [])
        if view_buttons:
            self._draw_view_buttons(view_buttons)
            tabs_bottom = max(button["y"] + button["h"] for button in view_buttons) + 14
            if tabs_bottom > left_col.y:
                consumed = tabs_bottom - left_col.y
                left_col.y += consumed
                left_col.height -= consumed
                right_col.y += consumed
                right_col.height -= consumed

        if not data:
            self._draw_text_card(
                pygame.Rect(left_col.x, left_col.y, left_col.width + right_col.width + col_gap, 170),
                "Deep RL Lab",
                [
                    "Run train.py to open the Deep RL dashboard.",
                    "Run visualizer.py later to inspect a saved checkpoint.",
                    "This folder is independent from the tabular project.",
                ],
            )
            return

        if data.get("view_mode") == "algorithm":
            full_rect = pygame.Rect(left_col.x, left_col.y, left_col.width + right_col.width + col_gap, left_col.height)
            self._draw_explainer_page(full_rect, data)
            return
        if data.get("view_mode") == "network":
            full_rect = pygame.Rect(left_col.x, left_col.y, left_col.width + right_col.width + col_gap, left_col.height)
            self._draw_network_page(full_rect, data)
            return
        if data.get("view_mode") == "results":
            full_rect = pygame.Rect(left_col.x, left_col.y, left_col.width + right_col.width + col_gap, left_col.height)
            self._draw_results_page(full_rect, data)
            return

        overview_rect = pygame.Rect(
            left_col.x,
            left_col.y,
            left_col.width + right_col.width + col_gap,
            left_col.height,
        )
        metrics_rows = max(1, (len(data.get("metrics", [])) + 1) // 2)
        metrics_row_h = int(data.get("metrics_row_h", 36))
        metrics_h = 52 + metrics_rows * metrics_row_h
        metrics_rect = pygame.Rect(overview_rect.x, overview_rect.y, overview_rect.width, metrics_h)
        self._draw_metrics_card(metrics_rect, data.get("metrics", []), row_h=metrics_row_h)

        content_y = metrics_rect.bottom + 15
        left_content = pygame.Rect(
            left_col.x,
            content_y,
            left_col.width,
            overview_rect.bottom - content_y,
        )
        right_content = pygame.Rect(
            right_col.x,
            content_y,
            right_col.width,
            overview_rect.bottom - content_y,
        )

        bottom_dock = data.get("bottom_dock")
        if bottom_dock:
            self._draw_bottom_dock(bottom_dock, data)

        self._draw_controls(data)

        comparison_lines = data.get("comparison_lines", [])
        if comparison_lines:
            comparison_top = max(
                left_content.y + 14,
                self._overview_sidebar_controls_bottom(data, left_content) + 16,
            )
            available_h = left_content.bottom - comparison_top
            if available_h >= 96:
                comparison_rect = pygame.Rect(
                    left_content.x,
                    comparison_top,
                    left_content.width,
                    available_h,
                )
                self._draw_text_card(comparison_rect, "Comparison Notes", comparison_lines)

        bulk_card = data.get("decision_card_mode") == "bulk"
        if bulk_card:
            q_h = max(174, min(220, int(right_content.height * 0.38)))
        else:
            q_h = max(146, min(176, int(right_content.height * 0.30)))
        q_rect = pygame.Rect(right_content.x, right_content.y, right_content.width, q_h)
        if data.get("q_values") is not None:
            self._draw_q_values(q_rect, data)

        right_content.y += q_h + 15
        right_content.height -= q_h + 15

        replay_data = data.get("recent_replays")
        graph_min_h = 170
        replay_h = 0
        if replay_data:
            replay_estimate = self._estimate_recent_replays_card_height(
                replay_data,
                right_content.width,
            )
            replay_min_h = 96 if replay_data.get("buttons") else 68
            replay_budget = max(0, right_content.height - graph_min_h - 15)
            replay_h = min(replay_estimate, replay_budget)
            if replay_h < replay_min_h:
                replay_h = 0

        graph_h = right_content.height - (replay_h + 15 if replay_h else 0)
        graph_h = max(120, graph_h)
        graph_rect = pygame.Rect(right_content.x, right_content.y, right_content.width, graph_h)
        data["_graph_rect"] = graph_rect
        self._draw_graph(graph_rect, data)
        
        right_content.y += graph_h + (15 if replay_h else 0)
        right_content.height -= graph_h

        if replay_h:
            replay_rect = pygame.Rect(
                right_content.x,
                right_content.y,
                right_content.width,
                replay_h,
            )
            self._draw_recent_replays_card(replay_rect, replay_data)

    def _draw_bottom_dock(self, dock_data, data):
        controls_rect_data = dock_data.get("controls_rect")
        if controls_rect_data:
            controls_rect = pygame.Rect(
                controls_rect_data["x"],
                controls_rect_data["y"],
                controls_rect_data["w"],
                controls_rect_data["h"],
            )
            self._draw_card_background(controls_rect)

        loss_rect_data = dock_data.get("loss_rect")
        if loss_rect_data:
            loss_rect = pygame.Rect(
                loss_rect_data["x"],
                loss_rect_data["y"],
                loss_rect_data["w"],
                loss_rect_data["h"],
            )
            self._draw_loss_graph(loss_rect, data)
            self._draw_inline_toggles(data.get("graph_toggles", []))

    def _draw_inline_toggles(self, toggles):
        for toggle in toggles or []:
            rect = pygame.Rect(toggle["x"], toggle["y"], toggle["w"], toggle["h"])
            base_color = (65, 180, 100) if toggle["value"] else (50, 52, 60)
            border_color = (45, 140, 75) if toggle["value"] else (40, 42, 48)
            pygame.draw.rect(self.display, base_color, rect, border_radius=6)
            pygame.draw.rect(self.display, border_color, rect, width=2, border_radius=6)
            text_color = (255, 255, 255) if toggle["value"] else (180, 185, 195)
            label_surface = self.tiny_font.render(toggle["label"], True, text_color)
            label_rect = label_surface.get_rect(center=rect.center)
            self.display.blit(label_surface, label_rect)

    def _overview_sidebar_controls_bottom(self, data, column_rect):
        bottom = column_rect.y
        left = column_rect.x - 4
        right = column_rect.right + 4

        for slider in data.get("sliders", []):
            track_rect = pygame.Rect(
                slider["track_x"],
                slider["track_y"],
                slider["track_w"],
                slider["track_h"],
            )
            if track_rect.right >= left and track_rect.x <= right:
                bottom = max(bottom, track_rect.bottom)

        for item in list(data.get("toggles", [])) + list(data.get("control_buttons", [])):
            rect = pygame.Rect(item["x"], item["y"], item["w"], item["h"])
            if rect.right >= left and rect.x <= right:
                bottom = max(bottom, rect.bottom)

        for section in data.get("control_sections", []):
            section_x = int(section.get("x", -10_000))
            if left <= section_x <= right:
                bottom = max(bottom, int(section.get("y", bottom)) + 18)

        return bottom

    def _draw_card_background(self, rect):
        """Draw a premium dark card with subtle borders."""
        pygame.draw.rect(self.display, (30, 32, 38), rect, border_radius=8)
        pygame.draw.rect(self.display, (55, 60, 70), rect, width=1, border_radius=8)
        # Subtle drop shadow at bottom
        pygame.draw.line(self.display, (15, 16, 20), (rect.x + 4, rect.bottom), (rect.right - 4, rect.bottom), 2)

    def _draw_view_buttons(self, view_buttons):
        for button in view_buttons:
            rect = pygame.Rect(button["x"], button["y"], button["w"], button["h"])
            active = button.get("active", False)
            fill = (70, 155, 220) if active else (44, 48, 56)
            border = (100, 210, 255) if active else (68, 72, 82)
            text_color = (255, 255, 255) if active else (188, 194, 202)
            pygame.draw.rect(self.display, fill, rect, border_radius=8)
            pygame.draw.rect(self.display, border, rect, width=2, border_radius=8)
            label_surface = self.small_font.render(button["label"], True, text_color)
            label_rect = label_surface.get_rect(center=rect.center)
            self.display.blit(label_surface, label_rect)

    def _draw_text_card(self, rect, title, lines):
        self._draw_card_background(rect)
        title_surface = self.small_font.render(title, True, (250, 252, 255))
        self.display.blit(title_surface, (rect.x + 12, rect.y + 10))

        y = rect.y + 36
        max_width = rect.width - 24
        for line in lines:
            color = (196, 202, 212)
            text = str(line)
            if text.startswith("[accent]"):
                text = text.replace("[accent]", "", 1).strip()
                color = (130, 205, 255)
            wrapped_lines = self._wrap_text(text, self.tiny_font, max_width)
            for wrapped in wrapped_lines:
                surface = self.tiny_font.render(wrapped, True, color)
                self.display.blit(surface, (rect.x + 12, y))
                y += 18

    def _draw_metrics_card(self, rect, metrics, row_h=36):
        self._draw_card_background(rect)
        title_surface = self.small_font.render("Training Snapshot", True, (250, 252, 255))
        self.display.blit(title_surface, (rect.x + 12, rect.y + 10))

        inner_x = rect.x + 12
        inner_y = rect.y + 38
        inner_w = rect.width - 24
        col_gap = 10
        col_w = (inner_w - col_gap) // 2

        for index, (label, value) in enumerate(metrics):
            col = index % 2
            row = index // 2
            card_rect = pygame.Rect(
                inner_x + col * (col_w + col_gap),
                inner_y + row * row_h,
                col_w,
                row_h - 4,
            )
            pygame.draw.rect(self.display, (22, 25, 31), card_rect, border_radius=8)
            pygame.draw.rect(self.display, (52, 58, 68), card_rect, width=1, border_radius=8)
            label_surface = self.tiny_font.render(str(label).upper(), True, (134, 142, 154))
            value_surface = self.tiny_font.render(str(value), True, (238, 242, 249))
            
            label_y = card_rect.y + (card_rect.height - label_surface.get_height()) // 2
            value_y = card_rect.y + (card_rect.height - value_surface.get_height()) // 2
            
            self.display.blit(label_surface, (card_rect.x + 8, label_y))
            value_x = card_rect.right - value_surface.get_width() - 8
            self.display.blit(value_surface, (value_x, value_y))

    def _estimate_text_card_height(self, lines, width):
        base = 48
        line_count = 0
        for line in lines:
            wrapped = self._wrap_text(str(line).replace("[accent]", "").strip(), self.tiny_font, width - 24)
            line_count += max(1, len(wrapped))
        return base + line_count * 18 + 8

    def _wrap_text(self, text, font, max_width):
        words = text.split()
        if not words:
            return [""]

        lines = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _draw_tooltip(self, x, y, lines):
        surfaces = [self.tiny_font.render(line, True, (250, 252, 255)) for line in lines]
        width = max(surface.get_width() for surface in surfaces) + 12
        height = sum(surface.get_height() for surface in surfaces) + 10
        rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.display, (35, 38, 48), rect, border_radius=4)
        pygame.draw.rect(self.display, (100, 110, 130), rect, width=1, border_radius=4)

        current_y = y + 4
        for surface in surfaces:
            self.display.blit(surface, (x + 6, current_y))
            current_y += surface.get_height()

    def _draw_network_page(self, rect, data):
        network = data.get("network_view", {})
        if not network:
            self._draw_text_card(rect, "Network View", ["Waiting for network data..."])
            return
        if network.get("message"):
            lines = [str(network["message"])]
            if network.get("details"):
                lines.extend(str(item) for item in network.get("details", []))
            self._draw_text_card(rect, "Network View", lines)
            return

        diagram_h = int(rect.height * 0.58)
        diagram_rect = pygame.Rect(rect.x, rect.y, rect.width, diagram_h)
        self._draw_network_diagram(diagram_rect, network)

        bottom_y = diagram_rect.bottom + 14
        bottom_h = rect.bottom - bottom_y
        col_gap = 12
        col_w = (rect.width - col_gap * 2) // 3

        input_lines = [
            f"{node['label']}: {node['value']:+.2f}"
            for node in network.get("top_inputs", [])
        ]
        hidden_lines = []
        for layer in network.get("hidden_highlights", []):
            hidden_lines.append(f"[accent]{layer.get('title', 'Hidden layer')}")
            hidden_lines.extend(
                f"{item['label']}: {item['value']:+.3f}"
                for item in layer.get("items", [])
            )

        output_lines = [
            f"{node['label']}: {node['value']:+.3f}{' <- chosen' if node.get('chosen') else ''}"
            for node in network.get("output_nodes", [])
        ]
        output_lines.extend(
            f"{item['label']}: {item['value']}"
            for item in network.get("layer_stats", [])[:4]
        )
        stat_lines = [
            f"{item['label']}: {item['value']}"
            for item in network.get("layer_stats", [])
        ]

        self._draw_text_card(
            pygame.Rect(rect.x, bottom_y, col_w, bottom_h),
            "Top Input Features",
            input_lines,
        )
        self._draw_text_card(
            pygame.Rect(rect.x + col_w + col_gap, bottom_y, col_w, bottom_h),
            "Hidden Layer Highlights",
            hidden_lines or stat_lines,
        )
        self._draw_text_card(
            pygame.Rect(rect.x + (col_w + col_gap) * 2, bottom_y, col_w, bottom_h),
            "Output Summary",
            output_lines,
        )

    def _draw_network_diagram(self, rect, network):
        self._draw_card_background(rect)
        title = self.small_font.render("Network In Operation", True, (250, 252, 255))
        subtitle = self.tiny_font.render(
            "Live forward pass through sampled activations from the current state.",
            True,
            (150, 198, 255),
        )
        self.display.blit(title, (rect.x + 12, rect.y + 10))
        self.display.blit(subtitle, (rect.x + 12, rect.y + 34))

        architecture = self.tiny_font.render(
            f"Architecture: {network.get('architecture_label', 'n/a')}",
            True,
            (200, 206, 216),
        )
        self.display.blit(architecture, (rect.right - 18 - architecture.get_width(), rect.y + 14))

        dominant_path = network.get("dominant_path")
        header_bottom = rect.y + 56
        if dominant_path:
            path_surface = self.tiny_font.render(
                f"Dominant path: {dominant_path}",
                True,
                (255, 214, 110),
            )
            self.display.blit(path_surface, (rect.x + 12, rect.y + 52))
            
            legend_x = rect.x + 12 + path_surface.get_width() + 20
            if legend_x > rect.right - 270:
                # If path text is long, push legend down
                self._draw_network_legend(rect.right - 270, rect.y + 70)
                header_bottom += 36
            else:
                self._draw_network_legend(rect.right - 270, rect.y + 50)
                header_bottom += 18
        else:
            self._draw_network_legend(rect.right - 270, rect.y + 50)

        plot_rect = pygame.Rect(rect.x + 16, header_bottom + 8, rect.width - 32, rect.bottom - header_bottom - 24)
        pygame.draw.rect(self.display, (20, 22, 26), plot_rect, border_radius=8)
        pygame.draw.rect(self.display, (45, 50, 58), plot_rect, width=1, border_radius=8)

        layers = network.get("layers", [])
        connection_blocks = network.get("connection_blocks", [])
        if not layers:
            waiting = self.tiny_font.render("Waiting for live forward-pass data...", True, (130, 136, 146))
            self.display.blit(waiting, (plot_rect.x + 14, plot_rect.y + 14))
            return

        gap = 10
        layer_top_h = int(plot_rect.height * 0.64)
        layer_area = pygame.Rect(plot_rect.x, plot_rect.y, plot_rect.width, layer_top_h)
        connection_area = pygame.Rect(
            plot_rect.x,
            layer_area.bottom + 10,
            plot_rect.width,
            plot_rect.bottom - layer_area.bottom - 10,
        )

        layer_count = max(1, len(layers))
        layer_w = max(48, (layer_area.width - gap * (layer_count - 1)) // layer_count)
        total_layer_w = (layer_w * layer_count) + (gap * (layer_count - 1))
        start_x = layer_area.x + max(0, (layer_area.width - total_layer_w) // 2)

        for index, layer in enumerate(layers):
            layer_rect = pygame.Rect(
                start_x + index * (layer_w + gap),
                layer_area.y,
                layer_w,
                layer_area.height,
            )
            self._draw_layer_heatmap_card(layer_rect, layer)

        if connection_blocks and connection_area.height > 50:
            block_gap = 10
            block_count = max(1, len(connection_blocks))
            block_w = max(52, (connection_area.width - block_gap * (block_count - 1)) // block_count)
            total_block_w = (block_w * block_count) + (block_gap * (block_count - 1))
            block_x = connection_area.x + max(0, (connection_area.width - total_block_w) // 2)
            for index, block in enumerate(connection_blocks):
                block_rect = pygame.Rect(
                    block_x + index * (block_w + block_gap),
                    connection_area.y,
                    block_w,
                    connection_area.height,
                )
                self._draw_connection_heatmap_block(block_rect, block)

    def _draw_layer_heatmap_card(self, rect, layer):
        pygame.draw.rect(self.display, (26, 29, 35), rect, border_radius=8)
        accent = self._layer_accent_color(layer.get("kind", "hidden"))
        pygame.draw.rect(self.display, accent, (rect.x + 1, rect.y + 1, rect.width - 2, 5), border_radius=8)
        pygame.draw.rect(self.display, (55, 60, 70), rect, width=1, border_radius=8)

        title_surface = self.tiny_font.render(layer.get("title", "Layer"), True, (235, 240, 248))
        self.display.blit(title_surface, (rect.x + 8, rect.y + 8))

        stats = layer.get("stats", {})
        meta_text = f"{layer.get('sampled_count', 0)}/{layer.get('size', 0)} shown"
        meta_surface = self.tiny_font.render(meta_text, True, (132, 192, 245))
        self.display.blit(meta_surface, (rect.x + 8, rect.y + 24))

        activity_text = f"active {stats.get('active_count', 0)} | max {stats.get('max_activation', 0.0):.2f}"
        activity_surface = self.tiny_font.render(activity_text, True, (150, 156, 168))
        self.display.blit(activity_surface, (rect.x + 8, rect.y + 40))

        heatmap = layer.get("heatmap", {})
        cells = heatmap.get("cells", [])
        if not cells:
            return

        columns = max(1, int(heatmap.get("columns", 1)))
        rows = max(1, math.ceil(len(cells) / columns))
        grid_rect = pygame.Rect(rect.x + 8, rect.y + 62, rect.width - 16, rect.height - 70)
        cell_gap = 6
        cell_w = max(22, (grid_rect.width - cell_gap * (columns - 1)) // columns)
        cell_h = max(26, (grid_rect.height - cell_gap * (rows - 1)) // rows)

        for index, cell in enumerate(cells):
            row = index // columns
            col = index % columns
            cell_rect = pygame.Rect(
                grid_rect.x + col * (cell_w + cell_gap),
                grid_rect.y + row * (cell_h + cell_gap),
                cell_w,
                cell_h,
            )
            self._draw_activation_cell(cell_rect, cell)

    def _draw_activation_cell(self, rect, cell):
        strength = max(0.0, min(1.0, float(cell.get("strength", 0.0))))
        chosen = bool(cell.get("chosen", False))
        positive = bool(cell.get("positive", True))
        value = float(cell.get("value", 0.0))
        rank = int(cell.get("rank", 999))

        if chosen:
            fill = (190 + int(65 * strength), 150 + int(40 * strength), 60)
            border = (255, 214, 110)
        elif positive:
            fill = (
                30,
                90 + int(100 * strength),
                110 + int(110 * strength),
            )
            border = (70, 190, 240)
        else:
            fill = (
                110 + int(110 * strength),
                48,
                58,
            )
            border = (235, 110, 118)
        if rank == 0 and not chosen:
            border = (235, 240, 248)

        pygame.draw.rect(self.display, fill, rect, border_radius=6)
        pygame.draw.rect(self.display, border, rect, width=2 if chosen else 1, border_radius=6)

        label = cell.get("label", "")
        # Adjust label length depending on cell width
        while label and self.tiny_font.size(label)[0] > (rect.width - 8):
            label = label[:-1]
            
        label_surface = self.tiny_font.render(label, True, (245, 248, 252))
        value_surface = self.tiny_font.render(f"{value:+.2f}", True, (245, 248, 252))
        
        if rect.height < 32:
            if rect.width > 60:
                self.display.blit(label_surface, (rect.x + 4, rect.y + (rect.height - label_surface.get_height()) // 2))
                self.display.blit(value_surface, (rect.right - value_surface.get_width() - 4, rect.y + (rect.height - value_surface.get_height()) // 2))
            else:
                self.display.blit(value_surface, (rect.x + (rect.width - value_surface.get_width()) // 2, rect.y + (rect.height - value_surface.get_height()) // 2))
        else:
            self.display.blit(label_surface, (rect.x + 6, rect.y + 5))
            self.display.blit(value_surface, (rect.x + 6, rect.bottom - 17))

    def _draw_connection_heatmap_block(self, rect, block):
        pygame.draw.rect(self.display, (24, 26, 31), rect, border_radius=8)
        pygame.draw.rect(self.display, (55, 60, 70), rect, width=1, border_radius=8)

        title_surface = self.tiny_font.render(block.get("short_title", "Connection"), True, (232, 238, 246))
        self.display.blit(title_surface, (rect.x + 8, rect.y + 8))

        rows = block.get("rows", [])
        if not rows:
            return

        matrix_rect = pygame.Rect(rect.x + 8, rect.y + 28, rect.width - 16, max(30, rect.height - 58))
        row_count = len(rows)
        col_count = len(rows[0]) if rows else 1
        gap = 4
        cell_w = max(10, (matrix_rect.width - gap * (col_count - 1)) // max(1, col_count))
        cell_h = max(10, (matrix_rect.height - gap * (row_count - 1)) // max(1, row_count))

        for row_index, row in enumerate(rows):
            for col_index, edge in enumerate(row):
                cell_rect = pygame.Rect(
                    matrix_rect.x + col_index * (cell_w + gap),
                    matrix_rect.y + row_index * (cell_h + gap),
                    cell_w,
                    cell_h,
                )
                strength = max(0.0, min(1.0, float(edge.get("strength", 0.0))))
                positive = bool(edge.get("positive", True))
                if positive:
                    color = (
                        32,
                        80 + int(120 * strength),
                        120 + int(110 * strength),
                    )
                else:
                    color = (
                        120 + int(110 * strength),
                        54,
                        64,
                    )
                pygame.draw.rect(self.display, color, cell_rect, border_radius=4)

        summary = block.get("summary", {})
        summary_text = (
            f"mean |w| {summary.get('mean_abs', 0.0):.2f} | "
            f"+ {summary.get('positive_ratio', 0.0) * 100:.0f}%"
        )
        summary_surface = self.tiny_font.render(summary_text, True, (150, 156, 168))
        self.display.blit(summary_surface, (rect.x + 8, rect.bottom - 18))

    def _draw_network_legend(self, x, y):
        items = [
            ((70, 190, 240), "positive"),
            ((235, 110, 118), "negative"),
            ((255, 214, 110), "chosen"),
            ((235, 240, 248), "top node"),
        ]
        cursor_x = x
        for color, label in items:
            pygame.draw.circle(self.display, color, (cursor_x + 5, y + 7), 5)
            label_surface = self.tiny_font.render(label, True, (168, 174, 184))
            self.display.blit(label_surface, (cursor_x + 14, y))
            cursor_x += 22 + label_surface.get_width()

    def _layer_accent_color(self, kind):
        if kind == "input":
            return (80, 190, 255)
        if kind == "output":
            return (255, 196, 68)
        return (80, 220, 180)

    def _draw_controls(self, data):
        for section in data.get("control_sections", []):
            label_surface = self.tiny_font.render(section["title"].upper(), True, (132, 192, 245))
            self.display.blit(label_surface, (section["x"], section["y"]))

        for slider in data.get("sliders", []):
            label_surface = self.tiny_font.render(
                f"{slider['label']}: {slider['value_text']}",
                True,
                (210, 215, 225),
            )
            self.display.blit(label_surface, (slider["x"], slider["y"]))

            track_rect = pygame.Rect(
                slider["track_x"], slider["track_y"], slider["track_w"], slider["track_h"]
            )
            fill_w = max(0, min(track_rect.width, int(track_rect.width * slider["ratio"])))
            fill_rect = pygame.Rect(track_rect.x, track_rect.y, fill_w, track_rect.height)

            # Premium slider track
            pygame.draw.rect(self.display, (45, 48, 55), track_rect, border_radius=6)
            pygame.draw.rect(self.display, (30, 32, 36), track_rect, width=1, border_radius=6)
            
            # Premium slider fill (gradient-like via inner rect, but solid for now)
            pygame.draw.rect(self.display, (80, 190, 255), fill_rect, border_radius=6)
            
            # Premium knob (larger, layered for glow effect)
            knob_center = (slider["knob_x"], slider["knob_y"])
            pygame.draw.circle(self.display, (40, 140, 220), knob_center, slider["knob_radius"] + 2)
            pygame.draw.circle(self.display, (250, 252, 255), knob_center, slider["knob_radius"])

        for toggle in data.get("toggles", []):
            rect = pygame.Rect(toggle["x"], toggle["y"], toggle["w"], toggle["h"])
            # Premium toggle buttons
            base_color = (65, 180, 100) if toggle["value"] else (50, 52, 60)
            border_color = (45, 140, 75) if toggle["value"] else (40, 42, 48)
            
            pygame.draw.rect(self.display, base_color, rect, border_radius=6)
            pygame.draw.rect(self.display, border_color, rect, width=2, border_radius=6)

            text_color = (255, 255, 255) if toggle["value"] else (180, 185, 195)
            label_surface = self.tiny_font.render(toggle["label"], True, text_color)
            # Center text in rect
            text_rect = label_surface.get_rect(center=rect.center)
            self.display.blit(label_surface, text_rect)

        for button in data.get("control_buttons", []):
            rect = pygame.Rect(button["x"], button["y"], button["w"], button["h"])
            active = bool(button.get("active", False))
            disabled = bool(button.get("disabled", False))
            style = button.get("style", "default")

            if disabled:
                base_color = (44, 46, 52)
                border_color = (58, 60, 68)
                text_color = (120, 124, 132)
            elif style == "start":
                base_color = (75, 150, 230) if active else (48, 52, 60)
                border_color = (110, 205, 255) if active else (60, 65, 75)
                text_color = (255, 255, 255) if active else (190, 195, 205)
            elif style == "device":
                base_color = (74, 126, 192) if active else (50, 52, 60)
                border_color = (112, 185, 255) if active else (40, 42, 48)
                text_color = (255, 255, 255) if active else (180, 185, 195)
            elif style == "queued":
                base_color = (110, 86, 34)
                border_color = (245, 190, 80)
                text_color = (255, 244, 220)
            else:
                base_color = (56, 60, 70) if active else (50, 52, 60)
                border_color = (100, 210, 255) if active else (40, 42, 48)
                text_color = (255, 255, 255) if active else (180, 185, 195)

            pygame.draw.rect(self.display, base_color, rect, border_radius=6)
            pygame.draw.rect(self.display, border_color, rect, width=2, border_radius=6)

            label_surface = self.tiny_font.render(button["label"], True, text_color)
            text_rect = label_surface.get_rect(center=rect.center)
            self.display.blit(label_surface, text_rect)

        for input_box in data.get("inputs", []):
            label_surface = self.tiny_font.render(input_box["label"], True, (210, 215, 225))
            self.display.blit(label_surface, (input_box["x"], input_box["y"] - 20))

            rect = pygame.Rect(input_box["x"], input_box["y"], input_box["w"], input_box["h"])
            # Premium text input
            fill_color = (20, 22, 26) if input_box.get("active") else (28, 30, 36)
            border_color = (100, 210, 255) if input_box.get("active") else (60, 65, 75)
            pygame.draw.rect(self.display, fill_color, rect, border_radius=6)
            pygame.draw.rect(self.display, border_color, rect, width=2, border_radius=6)

            value_text = input_box.get("text") or input_box.get("hint", "")
            value_color = (250, 252, 255) if input_box.get("text") else (120, 125, 135)
            value_surface = self.small_font.render(value_text, True, value_color)
            
            # Simple cursor blink effect
            if input_box.get("active") and pygame.time.get_ticks() % 1000 < 500:
                cursor_x = rect.x + 10 + value_surface.get_width()
                pygame.draw.line(self.display, (250, 252, 255), (cursor_x, rect.y + 6), (cursor_x, rect.bottom - 6), 2)
            
            self.display.blit(value_surface, (rect.x + 10, rect.y + 4))

    def _draw_q_values(self, rect, data):
        self._draw_card_background(rect)

        y = rect.y + 12
        heading = self.small_font.render(
            data.get("decision_card_title", "Decision & Q-Values"),
            True,
            (250, 252, 255),
        )
        self.display.blit(heading, (rect.x + 12, y))
        y += 28

        if data.get("decision_card_mode") == "bulk":
            max_width = rect.width - 24
            for line in data.get("decision_summary", []):
                wrapped_lines = self._wrap_text(str(line), self.tiny_font, max_width)
                for wrapped in wrapped_lines:
                    line_surface = self.tiny_font.render(wrapped, True, (190, 198, 210))
                    self.display.blit(line_surface, (rect.x + 12, y))
                    y += 16
                    if y > rect.bottom - 20:
                        return
            return

        for line in data.get("decision_summary", [])[:3]:
            line_surface = self.tiny_font.render(str(line), True, (190, 198, 210))
            self.display.blit(line_surface, (rect.x + 12, y))
            y += 17

        y += 8

        q_values = data.get("q_values", [0.0, 0.0, 0.0])
        action_labels = data.get("action_labels", ["Straight", "Right", "Left"])
        action_index = data.get("action_index")
        decision_type = data.get("decision_type", "")

        max_abs = max(1.0, max(abs(value) for value in q_values))
        
        # Calculate layut dynamically to fit rect
        label_w = 60
        val_w = 45
        bar_x = rect.x + 12 + label_w
        bar_w = rect.width - 24 - label_w - val_w
        
        for index, label in enumerate(action_labels):
            bar_y = y + index * 26
            value = q_values[index]
            fill_w = int((abs(value) / max_abs) * bar_w)
            
            # Premium colors
            color = (80, 230, 120) if value >= 0 else (255, 100, 100)
            if index == action_index:
                color = (255, 190, 60) if decision_type == "explore" else color
                if decision_type == "policy preview":
                    color = (100, 200, 255)

            label_surface = self.tiny_font.render(label, True, (200, 205, 215))
            self.display.blit(label_surface, (rect.x + 12, bar_y + 2))
            
            # Track
            pygame.draw.rect(self.display, (20, 22, 26), (bar_x, bar_y + 6, bar_w, 10), border_radius=5)
            # Fill
            pygame.draw.rect(self.display, color, (bar_x, bar_y + 6, fill_w, 10), border_radius=5)

            value_surface = self.tiny_font.render(f"{value:.2f}", True, (240, 245, 255))
            self.display.blit(value_surface, (bar_x + bar_w + 8, bar_y + 2))

    def _estimate_recent_replays_card_height(self, replay_data, width):
        base = 44
        line_count = 0
        for line in replay_data.get("lines", []):
            wrapped = self._wrap_text(str(line), self.tiny_font, width - 24)
            line_count += max(1, len(wrapped))
        footer = replay_data.get("footer", "")
        if footer:
            line_count += max(1, len(self._wrap_text(str(footer), self.tiny_font, width - 24)))
        button_rows = 1 if replay_data.get("buttons") else 0
        return base + line_count * 16 + button_rows * 34 + 10

    def _draw_recent_replays_card(self, rect, replay_data):
        self._draw_card_background(rect)
        title_surface = self.small_font.render(
            replay_data.get("title", "Recent Replays"),
            True,
            (250, 252, 255),
        )
        self.display.blit(title_surface, (rect.x + 12, rect.y + 10))

        y = rect.y + 36
        max_width = rect.width - 24
        lines = list(replay_data.get("lines", []))
        max_lines = 3 if replay_data.get("buttons") else 2
        for line in lines[:max_lines]:
            wrapped_lines = self._wrap_text(str(line), self.tiny_font, max_width)
            for wrapped in wrapped_lines:
                line_surface = self.tiny_font.render(wrapped, True, (196, 202, 212))
                self.display.blit(line_surface, (rect.x + 12, y))
                y += 16

        buttons = replay_data.get("buttons", [])
        if buttons:
            y += 4
            button_gap = 8
            button_h = 28
            button_count = max(1, len(buttons))
            button_w = max(88, (rect.width - 24 - button_gap * (button_count - 1)) // button_count)
            button_x = rect.x + 12
            for button in buttons:
                button_rect = pygame.Rect(button_x, y, button_w, button_h)
                button["x"] = button_rect.x
                button["y"] = button_rect.y
                button["w"] = button_rect.width
                button["h"] = button_rect.height

                pygame.draw.rect(self.display, (58, 116, 188), button_rect, border_radius=6)
                pygame.draw.rect(self.display, (120, 205, 255), button_rect, width=2, border_radius=6)
                label_surface = self.tiny_font.render(button.get("label", "Replay"), True, (255, 255, 255))
                label_rect = label_surface.get_rect(center=button_rect.center)
                self.display.blit(label_surface, label_rect)
                button_x += button_w + button_gap
            y += button_h + 6

        footer = replay_data.get("footer")
        if footer and y < rect.bottom - 14:
            footer_lines = self._wrap_text(str(footer), self.tiny_font, max_width)
            for wrapped in footer_lines[:1]:
                footer_surface = self.tiny_font.render(wrapped, True, (132, 192, 245))
                self.display.blit(footer_surface, (rect.x + 12, y))
                y += 16

    def _draw_state_block(self, rect, data):
        lines = list(data.get("state_lines", [])) + list(data.get("help_lines", []))
        self._draw_text_card(rect, "Top Active Features", lines)

    def _draw_explainer_page(self, rect, data):
        sections = data.get("algorithm_sections", [])
        if not sections:
            self._draw_text_card(rect, "How Deep RL Works", ["Waiting for live training data..."])
            return

        cols = 2 if rect.width >= 620 else 1
        rows = max(1, math.ceil(len(sections) / cols))
        gap = 12
        card_w = (rect.width - gap * (cols - 1)) // cols
        card_h = (rect.height - gap * (rows - 1)) // rows

        for index, section in enumerate(sections):
            row = index // cols
            col = index % cols
            card_rect = pygame.Rect(
                rect.x + col * (card_w + gap),
                rect.y + row * (card_h + gap),
                card_w,
                card_h,
            )
            self._draw_text_card(card_rect, section.get("title", "Section"), section.get("lines", []))

    def _draw_results_page(self, rect, data):
        data["_graph_rect"] = None
        summary_lines = data.get("results_summary_lines", [])
        score_series = data.get("results_score_series", [])
        loss_series = data.get("results_loss_series", [])

        summary_h = max(142, 54 + len(summary_lines) * 18)
        score_h = max(160, (rect.height - summary_h - 24) // 2)
        summary_rect = pygame.Rect(rect.x, rect.y, rect.width, summary_h)
        score_rect = pygame.Rect(rect.x, summary_rect.bottom + 12, rect.width, score_h)
        loss_rect = pygame.Rect(rect.x, score_rect.bottom + 12, rect.width, rect.bottom - score_rect.bottom - 12)

        self._draw_text_card(summary_rect, "Full Run Summary", summary_lines)
        score_total = max((len(item.get("values", [])) for item in score_series), default=0)
        loss_total = max((len(item.get("values", [])) for item in loss_series), default=0)
        self._draw_series_graph(
            score_rect,
            "Full Score History",
            score_series,
            view_size=max(2, score_total or 2),
            view_end=score_total or None,
            hover_index=None,
            empty_message="Score history will appear once completed runs are available.",
            min_value=0.0,
            min_ceiling=1.0,
            show_window_bar=False,
        )
        self._draw_series_graph(
            loss_rect,
            "Full Loss History",
            loss_series,
            view_size=max(2, loss_total or 2),
            view_end=loss_total or None,
            hover_index=None,
            empty_message="Loss history will appear once training updates have been recorded.",
            min_value=0.0,
            min_ceiling=0.02,
            show_window_bar=False,
        )

    def _draw_graph(self, rect, data):
        if not data.get("show_graph"):
            data["_graph_rect"] = None
            return
        self._draw_series_graph(
            rect,
            "Training Comparison",
            data.get("graph_series", []),
            view_size=data.get("graph_view_size", 60),
            view_end=data.get("graph_view_end"),
            hover_index=data.get("graph_hover_index"),
            empty_message="Waiting for data...",
            min_value=0.0,
            min_ceiling=1.0,
        )

    def _draw_loss_graph(self, rect, data):
        self._draw_series_graph(
            rect,
            "Loss Trend",
            data.get("loss_graph_series", []),
            view_size=data.get("graph_view_size", 60),
            view_end=data.get("graph_view_end"),
            hover_index=None,
            empty_message="Loss values will appear after completed episodes.",
            min_value=0.0,
            min_ceiling=0.02,
            header_extra=max(
                0,
                int(data.get("bottom_dock", {}).get("loss_rect", {}).get("toggle_strip_bottom", rect.y)) - rect.y - 26,
            ),
        )

    def _draw_series_graph(
        self,
        rect,
        title_text,
        raw_series,
        *,
        view_size=60,
        view_end=None,
        hover_index=None,
        empty_message="Waiting for data...",
        min_value=0.0,
        min_ceiling=1.0,
        show_window_bar=True,
        header_extra=0,
    ):
        self._draw_card_background(rect)

        title = self.tiny_font.render(title_text, True, (250, 252, 255))
        self.display.blit(title, (rect.x + 12, rect.y + 12))

        series = [
            item
            for item in raw_series
            if item.get("visible", True) and len(item.get("values", [])) > 0
        ]
        if not series:
            no_data = self.tiny_font.render(empty_message, True, (120, 125, 135))
            self.display.blit(no_data, (rect.x + 12, rect.y + 40))
            return

        legend_x = rect.x + 12
        legend_y = rect.y + 32 + max(0, int(header_extra))
        legend_max_x = rect.right - 16
        for item in series:
            color = tuple(item.get("color", (200, 200, 200)))
            label_surface = self.tiny_font.render(item.get("label", "Series"), True, (205, 210, 220))
            needed_w = 22 + label_surface.get_width()
            if legend_x + needed_w > legend_max_x:
                legend_x = rect.x + 12
                legend_y += 18
            pygame.draw.rect(self.display, color, (legend_x, legend_y + 4, 10, 10), border_radius=3)
            self.display.blit(label_surface, (legend_x + 16, legend_y))
            legend_x += 22 + label_surface.get_width()

        total = max(len(item.get("values", [])) for item in series)
        if total < 2:
            no_data = self.tiny_font.render(empty_message, True, (120, 125, 135))
            self.display.blit(no_data, (rect.x + 12, rect.y + 40))
            return

        view_size, view_end, view_start = self._resolve_graph_window(total, view_size, view_end)
        plot_rect = pygame.Rect(
            rect.x + 46,
            legend_y + 24,
            rect.width - 60,
            rect.height - (legend_y - rect.y) - 54,
        )

        pygame.draw.rect(self.display, (20, 22, 26), plot_rect, border_radius=4)
        pygame.draw.rect(self.display, (40, 42, 48), plot_rect, width=1, border_radius=4)

        min_value = float(min_value)
        max_value = self._resolve_graph_max_value(
            series,
            view_start,
            view_end,
            min_value=min_value,
            min_ceiling=min_ceiling,
        )
        value_span = max(1e-6, max_value - min_value)

        tick_count = 4
        for tick in range(tick_count + 1):
            ratio = tick / tick_count
            y = plot_rect.bottom - int(ratio * plot_rect.height)
            tick_value = min_value + value_span * ratio
            pygame.draw.line(
                self.display,
                (44, 48, 56),
                (plot_rect.x, y),
                (plot_rect.right, y),
                1,
            )
            label_surface = self.tiny_font.render(
                self._format_axis_value(tick_value),
                True,
                (150, 156, 168),
            )
            self.display.blit(
                label_surface,
                (rect.x + 8, y - (label_surface.get_height() // 2)),
            )

        pygame.draw.line(
            self.display,
            (80, 86, 98),
            (plot_rect.x, plot_rect.y),
            (plot_rect.x, plot_rect.bottom),
            1,
        )
        pygame.draw.line(
            self.display,
            (80, 86, 98),
            (plot_rect.x, plot_rect.bottom),
            (plot_rect.right, plot_rect.bottom),
            1,
        )

        for item in series:
            values = item.get("values", [])[view_start:view_end]
            if len(values) >= 2:
                self._draw_graph_line(
                    plot_rect,
                    values,
                    min_value,
                    max_value,
                    tuple(item.get("color", (200, 200, 200))),
                    thickness=item.get("thickness", 2),
                )

        if hover_index is not None and view_start <= hover_index < view_end:
            local_idx = hover_index - view_start
            px = plot_rect.x + int(local_idx / max(1, view_size - 1) * plot_rect.width)
            pygame.draw.line(self.display, (255, 255, 255, 120), (px, plot_rect.y), (px, plot_rect.bottom), 1)

            tooltip_lines = [f"Episode {hover_index + 1}"]
            for item in series:
                values = item.get("values", [])
                if hover_index < len(values):
                    value = values[hover_index]
                    color = tuple(item.get("color", (200, 200, 200)))
                    normalized = (value - min_value) / value_span
                    point_y = plot_rect.bottom - int(normalized * plot_rect.height)
                    point_y = max(plot_rect.top, min(plot_rect.bottom, point_y))
                    pygame.draw.circle(self.display, color, (px, point_y), 4)
                    tooltip_lines.append(
                        f"{item.get('label', 'Series')}: {self._format_axis_value(value)}"
                    )
            tip_x = min(px + 8, plot_rect.right - 170)
            tip_y = plot_rect.y + 8
            self._draw_tooltip(tip_x, tip_y, tooltip_lines)

        if show_window_bar:
            bar_y = rect.bottom - 18
            bar_rect = pygame.Rect(rect.x + 15, bar_y, rect.width - 30, 8)
            pygame.draw.rect(self.display, (30, 32, 38), bar_rect, border_radius=4)

            if total > view_size:
                thumb_ratio = view_size / total
                thumb_w = max(12, int(bar_rect.width * thumb_ratio))
                thumb_x = bar_rect.x + int((view_start / total) * bar_rect.width)
                thumb_rect = pygame.Rect(thumb_x, bar_y, thumb_w, 8)
                pygame.draw.rect(self.display, (80, 190, 255), thumb_rect, border_radius=4)

        range_text = f"Episodes {view_start + 1}-{view_end} of {total}"
        range_surf = self.tiny_font.render(range_text, True, (140, 145, 155))
        self.display.blit(range_surf, (rect.right - 15 - range_surf.get_width(), rect.y + 12))
        start_label = self.tiny_font.render(str(view_start + 1), True, (150, 156, 168))
        end_label = self.tiny_font.render(str(view_end), True, (150, 156, 168))
        self.display.blit(start_label, (plot_rect.x - start_label.get_width() // 2, plot_rect.bottom + 2))
        self.display.blit(end_label, (plot_rect.right - end_label.get_width() // 2, plot_rect.bottom + 2))

    def _draw_parallel_bulk_board(self, panel):
        title = panel.get("title", "Parallel Bulk Training")
        subtitle = panel.get("subtitle", "")
        progress_ratio = max(0.0, min(1.0, float(panel.get("progress_ratio", 0.0))))
        progress_label = panel.get("progress_label", "")
        lines = list(panel.get("lines", []))

        card_w = min(self.board_w - 72, 520)
        card_h = min(self.board_h - 96, 320)
        rect = pygame.Rect(
            (self.board_w - card_w) // 2,
            (self.board_h - card_h) // 2,
            card_w,
            card_h,
        )
        self._draw_card_background(rect)

        title_surface = self.title_font.render(title, True, (250, 252, 255))
        self.display.blit(title_surface, (rect.x + 18, rect.y + 18))

        subtitle_surface = self.small_font.render(subtitle, True, (130, 205, 255))
        self.display.blit(subtitle_surface, (rect.x + 18, rect.y + 52))

        y = rect.y + 88
        bar_rect = pygame.Rect(rect.x + 18, y, rect.width - 36, 14)
        pygame.draw.rect(self.display, (30, 34, 40), bar_rect, border_radius=7)
        fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, int(bar_rect.width * progress_ratio), bar_rect.height)
        pygame.draw.rect(self.display, (80, 190, 255), fill_rect, border_radius=7)
        pygame.draw.rect(self.display, (55, 60, 70), bar_rect, width=1, border_radius=7)
        y += 22

        if progress_label:
            progress_surface = self.small_font.render(progress_label, True, (235, 240, 248))
            self.display.blit(progress_surface, (rect.x + 18, y))
            y += 28

        max_width = rect.width - 36
        for line in lines:
            wrapped_lines = self._wrap_text(str(line), self.small_font, max_width)
            for wrapped in wrapped_lines:
                line_surface = self.small_font.render(wrapped, True, (196, 202, 212))
                self.display.blit(line_surface, (rect.x + 18, y))
                y += 22
                if y > rect.bottom - 24:
                    return

    def _resolve_graph_window(self, total, view_size, view_end):
        if view_end is None:
            view_end = total
        view_size = max(2, min(total, int(view_size)))
        view_end = max(view_size, min(total, int(view_end)))
        view_start = max(0, view_end - view_size)
        return view_size, view_end, view_start

    def _resolve_graph_max_value(self, series, view_start, view_end, *, min_value=0.0, min_ceiling=1.0):
        max_value = float(min_ceiling)
        for item in series:
            values = item.get("values", [])[view_start:view_end]
            if values:
                max_value = max(max_value, max(values))
        if max_value <= min_value:
            max_value = min_value + max(0.001, float(min_ceiling))
        else:
            max_value += max(0.001, (max_value - min_value) * 0.08)
        return max_value

    def _draw_graph_line(self, rect, values, min_value, max_value, color, thickness=2):
        if len(values) < 2:
            return

        value_span = max(1e-6, max_value - min_value)
        points = []
        for index, value in enumerate(values):
            ratio_x = index / (len(values) - 1)
            ratio_y = (value - min_value) / value_span
            x = rect.x + int(ratio_x * rect.width)
            y = rect.bottom - int(ratio_y * rect.height)
            
            # Keep inside plot area
            y = max(rect.top, min(rect.bottom, y))
            points.append((x, y))

        if len(points) >= 2:
            # Draw premium smoothed line
            pygame.draw.lines(self.display, color, False, points, thickness)

    def _format_axis_value(self, value):
        if value >= 10:
            return f"{value:.0f}"
        if value >= 1:
            return f"{value:.1f}"
        if value >= 0.1:
            return f"{value:.2f}"
        if value >= 0.01:
            return f"{value:.3f}"
        return f"{value:.4f}"

    def _draw_overlay_message(self):
        title = self.dashboard_data.get("overlay_title")
        subtitle = self.dashboard_data.get("overlay_subtitle")
        if not title:
            return

        overlay = pygame.Surface((self.board_w, self.board_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 140))
        self.display.blit(overlay, (0, 0))

        title_surface = self.title_font.render(title, True, (255, 255, 255))
        subtitle_surface = self.small_font.render(subtitle or "", True, (235, 235, 235))
        title_rect = title_surface.get_rect(center=(self.board_w // 2, self.board_h // 2 - 14))
        subtitle_rect = subtitle_surface.get_rect(center=(self.board_w // 2, self.board_h // 2 + 16))
        self.display.blit(title_surface, title_rect)
        self.display.blit(subtitle_surface, subtitle_rect)

        for button in self.dashboard_data.get("overlay_buttons", []):
            rect = pygame.Rect(button["x"], button["y"], button["w"], button["h"])
            pygame.draw.rect(self.display, (58, 116, 188), rect, border_radius=8)
            pygame.draw.rect(self.display, (120, 205, 255), rect, width=2, border_radius=8)
            label_surface = self.small_font.render(button.get("label", "Replay"), True, (255, 255, 255))
            label_rect = label_surface.get_rect(center=rect.center)
            self.display.blit(label_surface, label_rect)

    def _clamp_point_to_board(self, point):
        max_x = self.board_w - self.block_size
        max_y = self.board_h - self.block_size
        return Point(
            min(max(point.x, 0), max_x),
            min(max(point.y, 0), max_y),
        )

    def show_game_over_screen(self):
        """Simple restart screen for manual play."""
        if not self.render:
            return False

        self.set_dashboard_data(
            {
                "overlay_title": "Game Over",
                "overlay_subtitle": "Press R to restart or Q to quit",
            }
        )

        while not self.quit_requested:
            events = pygame.event.get()
            events = self.scale_events(events)
            self.handle_system_events(events)

            for event in events:
                if event.type != pygame.KEYDOWN:
                    continue
                if event.key in (pygame.K_r, pygame.K_RETURN):
                    self.set_dashboard_data({})
                    return True
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    self.quit_requested = True
                    return False

            self._draw_scene()
            self.clock.tick(10)

        return False

    def close(self):
        pygame.quit()


def run_human_game():
    game = SnakeGameAI(render=True, speed=12)

    try:
        while not game.quit_requested:
            game.set_dashboard_data(
                {
                    "panel_title": "Manual Play",
                    "metrics": [
                        ("Score", game.score),
                        ("Model", "Human control"),
                    ],
                    "state_lines": [
                        "Use Arrow keys or WASD to move.",
                        "Run train.py for the RL dashboard.",
                    ],
                    "help_lines": [],
                }
            )
            _, game_over, _ = game.play_step()
            if game_over:
                if game.show_game_over_screen():
                    game.reset()
                else:
                    break
    finally:
        game.close()


if __name__ == "__main__":
    run_human_game()
