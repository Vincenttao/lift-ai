from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MetricsCollector:
    wait_times: List[int] = field(default_factory=list)
    ride_times: List[int] = field(default_factory=list)
    completed: int = 0
    rejected: int = 0

    def record_completion(self, wait_time: int, ride_time: int) -> None:
        self.wait_times.append(wait_time)
        self.ride_times.append(ride_time)
        self.completed += 1

    def record_rejection(self) -> None:
        self.rejected += 1

    def summary(self) -> dict:
        return {
            "completed": self.completed,
            "rejected": self.rejected,
            "avg_wait": self._avg(self.wait_times),
            "p95_wait": self._percentile(self.wait_times, 0.95),
            "avg_ride": self._avg(self.ride_times),
        }

    @staticmethod
    def _avg(values: List[int]) -> float | None:
        return sum(values) / len(values) if values else None

    @staticmethod
    def _percentile(values: List[int], q: float) -> float | None:
        if not values:
            return None
        values_sorted = sorted(values)
        idx = int((len(values_sorted) - 1) * q)
        return float(values_sorted[idx])

