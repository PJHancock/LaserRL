
"""The original discrete grid optics prototype.

This is an independent toy model, not the implementation behind Astra's puzzle.

Physics/model assumptions
-------------------------
* Light travels one grid cell per tick in straight lines.
* A '/' mirror maps directions up->right, right->up, down->left, left->down.
* A '\\' mirror maps directions up->left, left->up, down->right, right->down.
* The ray stops at a wall, board edge, or repeated (cell, direction) state.
* A target is lit if any ray reaches its cell. Targets do not block the ray.

Actions can toggle a mirror or move it one cell up, right, down, or left. Moves into
occupied or blocked cells are invalid and leave the mirror in place. The observation
is a plain Python dictionary. The continuous environment is re-exported below.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Directions are clockwise: up, right, down, left.
UP, RIGHT, DOWN, LEFT = range(4)
DELTAS = ((0, -1), (1, 0), (0, 1), (-1, 0))
SLASH_TURN = (RIGHT, UP, LEFT, DOWN)
BACKSLASH_TURN = (LEFT, DOWN, RIGHT, UP)
MOVE_DELTAS = {1: (0, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 0)}
COMMAND_NAMES = ("toggle", "up", "right", "down", "left")
Cell = Tuple[int, int]


@dataclass(frozen=True)
class Level:
    width: int
    height: int
    source: Cell
    direction: int
    targets: Tuple[Cell, ...]
    mirror_cells: Tuple[Cell, ...]
    walls: Tuple[Cell, ...] = ()
    initial_mirrors: Tuple[int, ...] = ()  # 0='/', 1='\\'

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("width and height must be positive")
        if self.direction not in range(4):
            raise ValueError("direction must be UP, RIGHT, DOWN, or LEFT")
        cells = (self.source,) + self.targets + self.mirror_cells + self.walls
        if any(not self._inside(c) for c in cells):
            raise ValueError("all level cells must be inside the board")
        if self.source in self.walls:
            raise ValueError("source cannot be a wall")
        if len(set(self.mirror_cells)) != len(self.mirror_cells):
            raise ValueError("starting mirror cells must be distinct")
        if self.source in self.mirror_cells:
            raise ValueError("source cannot overlap a mirror")
        if set(self.mirror_cells) & set(self.walls):
            raise ValueError("mirror cells cannot overlap walls")
        if set(self.mirror_cells) & set(self.targets):
            raise ValueError("mirror cells cannot overlap targets")
        if self.initial_mirrors and len(self.initial_mirrors) != len(self.mirror_cells):
            raise ValueError("initial_mirrors must have one value per mirror cell")
        if any(o not in (0, 1) for o in self.initial_mirrors):
            raise ValueError("mirror orientations must be 0 ('/') or 1 ('\\')")

    def _inside(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height


class OpticsGridEnv:
    """Move and rotate mirrors to direct a beam through all targets.

    Each mirror has five actions: toggle, move up, move right, move down, or move
    left. Action ``mirror_index * 5 + command_index`` selects the command, where
    command indices follow ``COMMAND_NAMES``. For example, ``action_id(0, "right")``
    moves mirror 0 one cell right.

    Moves cannot enter the source, a target, a wall, or another mirror. An invalid
    move is a no-op with a small extra penalty.

    Args:
        level: A Level instance. If omitted, a small demo level is used.
        max_steps: Maximum actions before the episode terminates.

    Usage::

        env = OpticsGridEnv()
        obs = env.reset()
        obs, reward, done, info = env.step(env.action_id(0, "toggle"))
        print(env.render())

    For a modern Gymnasium wrapper, map reset() to ``(obs, {})`` and step() to
    ``(obs, reward, terminated, truncated, info)`` as appropriate for your setup.
    """

    def __init__(self, level: Optional[Level] = None, max_steps: int = 30):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.level = level or demo_level()
        self.max_steps = max_steps
        self.steps = 0
        self.orientations: List[int] = []
        self.mirror_positions: List[Cell] = []
        self._last_rays: List[Cell] = []
        self._last_lit: set[Cell] = set()
        self.reset()

    def reset(self, seed: Optional[int] = None) -> Dict[str, object]:
        """Start a new episode; seed is accepted for wrapper compatibility."""
        del seed  # This deterministic environment has no randomness.
        self.steps = 0
        self.mirror_positions = list(self.level.mirror_cells)
        if self.level.initial_mirrors:
            self.orientations = list(self.level.initial_mirrors)
        else:
            self.orientations = [0] * len(self.level.mirror_cells)
        self._trace()
        return self.observe()

    @property
    def action_count(self) -> int:
        """Number of discrete actions: five commands per mirror."""
        return 5 * len(self.mirror_positions)

    def action_id(self, mirror_index: int, command: str) -> int:
        """Encode a mirror command as an integer action for ``step``."""
        if not 0 <= mirror_index < len(self.mirror_positions):
            raise ValueError("mirror_index is out of range")
        if command not in COMMAND_NAMES:
            raise ValueError(f"command must be one of {COMMAND_NAMES}")
        return 5 * mirror_index + COMMAND_NAMES.index(command)

    def step(self, action: int):
        """Apply an encoded mirror command; return (observation, reward, done, info)."""
        if not isinstance(action, int) or not 0 <= action < self.action_count:
            raise ValueError(f"action must be an integer in [0, {self.action_count})")

        was_solved = self.solved
        mirror_index, command_index = divmod(action, 5)
        command = COMMAND_NAMES[command_index]
        valid = True
        if command_index == 0:
            self.orientations[mirror_index] ^= 1
        else:
            x, y = self.mirror_positions[mirror_index]
            dx, dy = MOVE_DELTAS[command_index]
            destination = (x + dx, y + dy)
            occupied = set(self.mirror_positions)
            occupied.discard((x, y))
            blocked = set(self.level.walls) | set(self.level.targets) | {self.level.source} | occupied
            if (not self.level._inside(destination)) or destination in blocked:
                valid = False
            else:
                self.mirror_positions[mirror_index] = destination

        self.steps += 1
        self._trace()
        solved = self.solved
        reward = 1.0 if solved else (-0.05 if not valid else -0.01)
        done = solved or self.steps >= self.max_steps
        info = {
            "solved": solved,
            "action_valid": valid,
            "command": command,
            "mirror_index": mirror_index,
            "lit_targets": tuple(sorted(self._last_lit & set(self.level.targets))),
            "steps": self.steps,
            "already_solved": was_solved,
        }
        return self.observe(), reward, done, info

    @property
    def solved(self) -> bool:
        return set(self.level.targets).issubset(self._last_lit)

    def observe(self) -> Dict[str, object]:
        """Return a serializable observation with board geometry and beam trace."""
        return {
            "width": self.level.width,
            "height": self.level.height,
            "source": self.level.source,
            "direction": self.level.direction,
            "targets": self.level.targets,
            "mirror_cells": tuple(self.mirror_positions),
            "mirror_orientations": tuple(self.orientations),
            "walls": self.level.walls,
            "ray_cells": tuple(self._last_rays),
            "lit_targets": tuple(t for t in self.level.targets if t in self._last_lit),
            "solved": self.solved,
            "steps": self.steps,
        }

    def _trace(self) -> None:
        """Trace the single beam and cache visited cells and lit targets."""
        mirror_at = {cell: self.orientations[i] for i, cell in enumerate(self.mirror_positions)}
        walls = set(self.level.walls)
        targets = set(self.level.targets)
        x, y = self.level.source
        direction = self.level.direction
        seen_states = set()
        rays: List[Cell] = []
        lit: set[Cell] = set()

        # State-based loop detection handles cycles without an arbitrary ray limit.
        while True:
            dx, dy = DELTAS[direction]
            x, y = x + dx, y + dy
            cell = (x, y)
            if not (0 <= x < self.level.width and 0 <= y < self.level.height) or cell in walls:
                break
            state = (cell, direction)
            if state in seen_states:
                break
            seen_states.add(state)
            rays.append(cell)
            if cell in targets:
                lit.add(cell)
            if cell in mirror_at:
                direction = (SLASH_TURN if mirror_at[cell] == 0 else BACKSLASH_TURN)[direction]

        self._last_rays = rays
        self._last_lit = lit

    def render(self) -> str:
        """Render a compact text board for logs and debugging."""
        grid = [["." for _ in range(self.level.width)] for _ in range(self.level.height)]
        for x, y in self.level.walls:
            grid[y][x] = "#"
        for i, (x, y) in enumerate(self.mirror_positions):
            grid[y][x] = "/" if self.orientations[i] == 0 else "\\"
        for x, y in self.level.targets:
            grid[y][x] = "*" if (x, y) in self._last_lit else "T"
        for x, y in self._last_rays:
            if grid[y][x] == ".":
                grid[y][x] = ":"
        sx, sy = self.level.source
        grid[sy][sx] = {UP: "^", RIGHT: ">", DOWN: "v", LEFT: "<"}[self.level.direction]
        return "\n".join(" ".join(row) for row in grid)


def demo_level() -> Level:
    """A tiny level: toggle mirror 0 to light the target at (5, 1)."""
    return Level(
        width=7,
        height=5,
        source=(0, 2),
        direction=RIGHT,
        targets=((5, 1),),
        mirror_cells=((2, 2), (2, 1)),
        walls=((4, 3),),
        initial_mirrors=(1, 0),
    )


# The continuous PPO environment lives separately from this grid prototype.
from optics_env import ContinuousOpticsEnv  # noqa: E402,F401
