from __future__ import annotations

import abc
from typing import Iterable, List, Sequence

from .elevator import Elevator
from .passenger import Passenger


class Dispatcher(abc.ABC):
    """Abstract dispatcher. Implementations produce assignments for new passengers."""

    @abc.abstractmethod
    def assign(self, time: int, passengers: Iterable[Passenger], elevators: Sequence[Elevator]) -> None:
        """
        Assign passengers to elevators by mutating passenger.assigned_elevator and
        optionally enqueuing stops on elevators.
        """
        raise NotImplementedError


class ETADispatcher(Dispatcher):
    """Baseline ETA-first dispatcher placeholder."""

    dwell_time: int

    def __init__(self, dwell_time: int = 5) -> None:
        self.dwell_time = dwell_time

    def assign(self, time: int, passengers: Iterable[Passenger], elevators: Sequence[Elevator]) -> None:
        # Greedy min distance (ETA proxy) based on current floor; ignores queue ordering.
        for p in passengers:
            if p.assigned_elevator is not None:
                continue
            best_elevator = self._pick_elevator(elevators, p.origin_floor)
            if best_elevator is None:
                continue
            p.assigned_elevator = best_elevator.id
            if not best_elevator.is_full():
                best_elevator.enqueue_stop(p.origin_floor)

    def _pick_elevator(self, elevators: Sequence[Elevator], origin_floor: int) -> Elevator | None:
        candidates: List[Elevator] = [e for e in elevators if not e.is_full()]
        if not candidates:
            return None
        return min(candidates, key=lambda e: abs(e.current_floor - origin_floor))
