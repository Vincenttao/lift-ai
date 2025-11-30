from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    import gymnasium as gym
    from gymnasium import spaces as gym_spaces

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None

import numpy as np

from core.environment import LiftSimEnvironment, SimulationConfig


class LiftGymEnv(gym.Env if gym else object):  # type: ignore[misc]
    """
    Gymnasium wrapper for LiftSim.
    Provides fixed-shape observation/action interfaces; requires gymnasium to be installed.
    """

    metadata = {"render.modes": []}

    def __init__(self, config: Optional[SimulationConfig] = None) -> None:
        if gym is None or spaces is None:
            raise ImportError("gymnasium is required for LiftGymEnv; install gymnasium to use this wrapper.")
        self.config = config or SimulationConfig()
        self.sim = LiftSimEnvironment(config=self.config)
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    def _build_observation_space(self) -> "gym_spaces.Dict":
        assert spaces is not None  # for type checkers
        return spaces.Dict(
            {
                "elevator_floor": spaces.MultiDiscrete([self.config.floors] * self.config.elevators),
                "elevator_direction": spaces.MultiDiscrete([3] * self.config.elevators),  # -1/0/1 shifted to 0/1/2
                "door_state": spaces.MultiDiscrete([2] * self.config.elevators),
                "is_full": spaces.MultiBinary(self.config.elevators),
                "hall_call_up": spaces.MultiBinary(self.config.floors),
                "hall_call_down": spaces.MultiBinary(self.config.floors),
                "time": spaces.Discrete(self.config.horizon_s + 1),
            }
        )

    def _build_action_space(self) -> "gym_spaces.MultiDiscrete":
        assert spaces is not None  # for type checkers
        # One discrete choice per elevator: target floor (0=no-op)
        return spaces.MultiDiscrete([self.config.floors + 1] * self.config.elevators)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        obs = self.sim.reset(seed=seed)
        return self._observe_from_state(obs), {}

    def step(self, action):
        # Normalize action coming from VecEnv (may be nested array of shape (1, n))
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if not isinstance(action, (list, tuple)):
            action = [action]
        if len(action) == 1 and isinstance(action[0], (list, tuple)):
            action = list(action[0])
        action = self._mask_actions(action)
        obs, reward, terminated, info = self.sim.step(list(action))
        obs = self._observe_from_state(obs)
        truncated = False
        return obs, reward, terminated, truncated, info

    def _observe_from_state(self, state: dict):
        floors = [max(0, min(self.config.floors - 1, f - 1)) for f in state["elevator_floor"]]
        directions = [d + 1 for d in state["elevator_direction"]]  # shift -1/0/1 to 0/1/2
        door_state = state.get("door_state", [0] * self.config.elevators)
        is_full = state.get("is_full", [0] * self.config.elevators)
        hall_up = np.array(state.get("hall_call_up", [0] * self.config.floors), dtype=np.int64)
        hall_down = np.array(state.get("hall_call_down", [0] * self.config.floors), dtype=np.int64)
        return {
            "elevator_floor": floors,
            "elevator_direction": directions,
            "door_state": door_state,
            "is_full": is_full,
            "hall_call_up": hall_up,
            "hall_call_down": hall_down,
            "time": state["time"],
        }

    def _mask_actions(self, actions: list) -> list:
        """
        Replace invalid actions with no-op (0) using the environment-provided valid action set.
        """
        valid = self.sim.valid_actions()
        masked = []
        for act, allowed in zip(actions, valid):
            masked.append(act if act in allowed else 0)
        return masked
