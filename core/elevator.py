from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class DoorState(str, Enum):
    OPEN = "open"
    CLOSED = "closed"


class MoveState(str, Enum):
    IDLE = "idle"
    MOVING = "moving"
    DWELL = "dwell"


@dataclass
class Elevator:
    """Represents a single elevator car."""

    id: int
    capacity: int = 12
    current_floor: int = 1
    direction: int = 0  # -1 down, 0 idle, +1 up
    door_state: DoorState = DoorState.CLOSED
    move_state: MoveState = MoveState.IDLE
    target_queue: List[int] = field(default_factory=list)
    passengers: List[int] = field(default_factory=list)  # holds passenger ids
    dwell_remaining: int = 0  # seconds to finish current stop

    def is_full(self) -> bool:
        return len(self.passengers) >= self.capacity

    def enqueue_stop(self, floor: int) -> None:
        if floor not in self.target_queue:
            self.target_queue.append(floor)

    def step(self, dt: int = 1) -> None:
        """
        Advance elevator state by dt seconds.
        Placeholder: concrete movement/stop logic is defined in the environment loop.
        """
        if self.move_state == MoveState.DWELL:
            self.dwell_remaining = max(0, self.dwell_remaining - dt)
            if self.dwell_remaining == 0:
                self.move_state = MoveState.IDLE

