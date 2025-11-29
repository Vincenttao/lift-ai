from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Passenger:
    """Passenger lifecycle properties."""

    id: int
    appear_time: int
    origin_floor: int
    dest_floor: int
    assigned_elevator: Optional[int] = None
    board_time: Optional[int] = None
    arrive_time: Optional[int] = None

    @property
    def waiting_time(self) -> Optional[int]:
        if self.board_time is None:
            return None
        return self.board_time - self.appear_time

    @property
    def ride_time(self) -> Optional[int]:
        if self.board_time is None or self.arrive_time is None:
            return None
        return self.arrive_time - self.board_time

