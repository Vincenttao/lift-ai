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

    def enqueue_stop(self, floor: int, force: bool = False) -> None:
        """
        Add a stop to the queue. When full and force is False, ignore new stops to honor
        the "no new boarding requests while full" rule; existing onboard destinations
        should already be in the queue.
        """
        if self.is_full() and not force:
            return
        if floor not in self.target_queue:
            self.target_queue.append(floor)

    def begin_dwell(self, dwell_time: int) -> None:
        self.move_state = MoveState.DWELL
        self.door_state = DoorState.OPEN
        self.dwell_remaining = dwell_time
        self.direction = 0

    def tick_dwell(self, dt: int = 1) -> None:
        self.dwell_remaining = max(0, self.dwell_remaining - dt)
        if self.dwell_remaining == 0:
            self.move_state = MoveState.IDLE
            self.door_state = DoorState.CLOSED

    def move_toward(self, target_floor: int) -> None:
        if target_floor == self.current_floor:
            return
        self.move_state = MoveState.MOVING
        self.direction = 1 if target_floor > self.current_floor else -1
        self.current_floor += self.direction
