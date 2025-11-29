from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    gym = None
    spaces = None

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

    def _build_observation_space(self) -> "spaces.Dict":
        return spaces.Dict(
            {
                "elevator_floor": spaces.MultiDiscrete([self.config.floors] * self.config.elevators),
                "elevator_load": spaces.MultiDiscrete([self.config.capacity + 1] * self.config.elevators),
                "time": spaces.Discrete(self.config.horizon_s + 1),
            }
        )

    def _build_action_space(self) -> "spaces.MultiDiscrete":
        # One discrete choice per elevator: target floor (0=no-op)
        return spaces.MultiDiscrete([self.config.floors + 1] * self.config.elevators)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.sim.config.seed = seed
        self.sim.reset()
        obs = self._observe()
        return obs, {}

    def step(self, action):
        if not isinstance(action, (list, tuple)):
            action = [action]
        # Placeholder: integrate actions into dispatcher/queue updates.
        self.sim.step()
        obs = self._observe()
        reward = 0.0
        terminated = self.sim.time >= self.config.horizon_s
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _observe(self):
        floors = [max(0, min(self.config.floors - 1, e.current_floor - 1)) for e in self.sim.elevators]
        loads = [min(self.config.capacity, len(e.passengers)) for e in self.sim.elevators]
        return {
            "elevator_floor": floors,
            "elevator_load": loads,
            "time": self.sim.time,
        }

