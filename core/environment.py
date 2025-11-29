from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Dict, List, Optional, Sequence

from .dispatcher import Dispatcher, ETADispatcher
from .elevator import Elevator
from .metrics import MetricsCollector
from .passenger import Passenger


@dataclass
class SimulationConfig:
    floors: int = 18
    elevators: int = 2
    capacity: int = 12
    floor_height_m: float = 3.0
    speed_mps: float = 3.0
    dwell_time_s: int = 5
    spawn_prob: float = 0.2
    horizon_s: int = 3600
    seed: Optional[int] = None

    @property
    def seconds_per_floor(self) -> float:
        return self.floor_height_m / self.speed_mps


class LiftSimEnvironment:
    """Discrete-time lift simulation environment (non-gym wrapper)."""

    def __init__(self, config: SimulationConfig, dispatcher: Optional[Dispatcher] = None) -> None:
        self.config = config
        self.dispatcher = dispatcher or ETADispatcher(dwell_time=config.dwell_time_s)
        self.rng = Random(config.seed)
        self.time: int = 0
        self.passengers: Dict[int, Passenger] = {}
        self.elevators: List[Elevator] = [
            Elevator(id=i + 1, capacity=config.capacity) for i in range(config.elevators)
        ]
        self.metrics = MetricsCollector()
        self._next_passenger_id = 1

    def reset(self) -> None:
        self.time = 0
        self.passengers.clear()
        self.elevators = [Elevator(id=i + 1, capacity=self.config.capacity) for i in range(self.config.elevators)]
        self.metrics = MetricsCollector()
        self._next_passenger_id = 1
        self.rng.seed(self.config.seed)

    def step(self) -> bool:
        """Advance one second. Returns True if simulation continues, False if horizon reached."""
        if self.time >= self.config.horizon_s:
            return False
        self._spawn_passengers()
        self._dispatch_new()
        self._move_elevators()
        self.time += 1
        return True

    def _spawn_passengers(self) -> None:
        if self.rng.random() > self.config.spawn_prob:
            return
        origin = self.rng.randint(1, self.config.floors)
        dest = origin
        while dest == origin:
            dest = self.rng.randint(1, self.config.floors)
        pid = self._next_passenger_id
        self._next_passenger_id += 1
        self.passengers[pid] = Passenger(
            id=pid,
            appear_time=self.time,
            origin_floor=origin,
            dest_floor=dest,
        )

    def _dispatch_new(self) -> None:
        new_passengers = [p for p in self.passengers.values() if p.assigned_elevator is None]
        if new_passengers:
            self.dispatcher.assign(self.time, new_passengers, self.elevators)

    def _move_elevators(self) -> None:
        # Placeholder: to be filled with full movement, dwell, and boarding logic.
        for elevator in self.elevators:
            elevator.step(dt=1)

