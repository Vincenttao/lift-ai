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
    speed_mps: float = 1.5
    dwell_time_s: int = 5
    spawn_prob: float = 0.05
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
        self.time: int = 0  # seconds elapsed
        self.passengers: Dict[int, Passenger] = {}
        self.elevators: List[Elevator] = [
            Elevator(id=i + 1, capacity=config.capacity) for i in range(config.elevators)
        ]
        self.metrics = MetricsCollector()
        self._next_passenger_id = 1
        self._finalized = False

    def reset(self, seed: Optional[int] = None) -> Dict[str, object]:
        self.time = 0
        self.passengers.clear()
        self.elevators = [Elevator(id=i + 1, capacity=self.config.capacity) for i in range(self.config.elevators)]
        self.metrics = MetricsCollector()
        self._next_passenger_id = 1
        seed_to_use = self.config.seed if seed is None else seed
        self.rng.seed(seed_to_use)
        self._finalized = False
        return self.observe()

    def step(self, actions: Optional[List[int]] = None):
        """
        Advance one second.
        actions: list of target floors (length = elevators), 0 means no-op. Floors are 1-based.
        Returns: obs, reward (placeholder 0.0), terminated, info.
        """
        if self.time >= self.config.horizon_s:
            added_unserved = self._finalize_unserved()
            reward = self._compute_reward(
                delta_completed=0,
                delta_rejected=0,
                delta_unserved=added_unserved,
                completed_passengers=[],
            )
            return self.observe(), reward, True, {"action_mask": self.valid_actions()}

        prev_completed = self.metrics.completed
        prev_rejected = self.metrics.rejected
        prev_unserved = self.metrics.unserved
        completed_passengers: List[Passenger] = []

        self._spawn_passengers()
        self._dispatch_new()
        if actions is not None:
            self.apply_actions(actions)
        self._move_elevators(completed_passengers)
        self.time += 1
        terminated = self.time >= self.config.horizon_s
        added_unserved = self._finalize_unserved() if terminated else 0
        reward = self._compute_reward(
            delta_completed=self.metrics.completed - prev_completed,
            delta_rejected=self.metrics.rejected - prev_rejected,
            delta_unserved=added_unserved,
            completed_passengers=completed_passengers,
        )
        return self.observe(), reward, terminated, {"action_mask": self.valid_actions()}

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

    def apply_actions(self, actions: Sequence[int]) -> None:
        if len(actions) != len(self.elevators):
            raise ValueError("Actions length must equal number of elevators.")
        for elevator, target in zip(self.elevators, actions):
            if target is None or target == 0:
                continue
            if 1 <= target <= self.config.floors and not elevator.is_full():
                elevator.enqueue_stop(int(target))

    def _move_elevators(self, completed: List[Passenger]) -> None:
        for elevator in self.elevators:
            if elevator.move_state == elevator.move_state.DWELL:
                elevator.tick_dwell(dt=1)
                continue
            if not elevator.target_queue:
                elevator.direction = 0
                elevator.move_state = elevator.move_state.IDLE
                continue

            target = elevator.target_queue[0]
            if elevator.current_floor == target:
                # Arrived: unload/load and dwell
                self._process_stop(elevator, completed)
                elevator.target_queue.pop(0)
                elevator.begin_dwell(self.config.dwell_time_s)
                continue

            # Move one floor toward target
            elevator.move_toward(target)

    def _process_stop(self, elevator: Elevator, completed: List[Passenger]) -> None:
        # Drop off passengers
        to_drop = [pid for pid in elevator.passengers if self.passengers[pid].dest_floor == elevator.current_floor]
        for pid in to_drop:
            passenger = self.passengers[pid]
            passenger.arrive_time = self.time
            elevator.passengers.remove(pid)
            if passenger.waiting_time is not None and passenger.ride_time is not None:
                self.metrics.record_completion(passenger.waiting_time, passenger.ride_time)
                completed.append(passenger)

        # Pick up waiting passengers assigned to this elevator
        waiting = [
            p for p in self.passengers.values()
            if p.assigned_elevator == elevator.id and p.board_time is None and p.origin_floor == elevator.current_floor
        ]
        for p in waiting:
            if elevator.is_full():
                self.metrics.record_rejection()
                # Re-queue passenger for redispatch so they can board later.
                p.assigned_elevator = None
                continue
            p.board_time = self.time
            elevator.passengers.append(p.id)
            elevator.enqueue_stop(p.dest_floor)

    def observe(self) -> Dict[str, object]:
        floors = [e.current_floor for e in self.elevators]
        directions = [e.direction for e in self.elevators]
        doors = [0 if e.door_state == e.door_state.CLOSED else 1 for e in self.elevators]
        is_full = [1 if e.is_full() else 0 for e in self.elevators]
        hall_up, hall_down = self._hall_calls()
        return {
            "time": self.time,
            "elevator_floor": floors,
            "elevator_direction": directions,
            "door_state": doors,
            "is_full": is_full,
            "hall_call_up": hall_up,
            "hall_call_down": hall_down,
        }

    def valid_actions(self) -> List[List[int]]:
        """
        Returns per-elevator list of valid target floors (including 0=no-op).
        Full elevators only allow no-op until space frees up.
        """
        valid: List[List[int]] = []
        for e in self.elevators:
            if e.is_full():
                valid.append([0])
            else:
                valid.append([0] + list(range(1, self.config.floors + 1)))
        return valid

    def _finalize_unserved(self) -> int:
        if self._finalized:
            return 0
        unserved = sum(1 for p in self.passengers.values() if p.arrive_time is None)
        if unserved > 0:
            self.metrics.record_unserved(unserved)
        self._finalized = True
        return unserved

    def _hall_calls(self) -> tuple[list[int], list[int]]:
        up = [0 for _ in range(self.config.floors)]
        down = [0 for _ in range(self.config.floors)]
        for p in self.passengers.values():
            if p.board_time is not None:
                continue
            idx = p.origin_floor - 1
            if p.dest_floor > p.origin_floor:
                up[idx] += 1
            else:
                down[idx] += 1
        # Convert counts to binary call presence (observable call buttons)
        up = [1 if c > 0 else 0 for c in up]
        down = [1 if c > 0 else 0 for c in down]
        return up, down

    def _compute_reward(
        self,
        delta_completed: int,
        delta_rejected: int,
        delta_unserved: int,
        completed_passengers: List[Passenger],
    ) -> float:
        waiting_count = sum(1 for p in self.passengers.values() if p.board_time is None)
        reward = 0.0
        reward -= 0.015 * waiting_count  # lighter per-step backlog penalty
        reward -= 2.0 * delta_rejected
        reward -= 20.0 * delta_unserved
        for p in completed_passengers:
            wait = p.waiting_time or 0
            ride = p.ride_time or 0
            reward += 9.0  # stronger base positive for serving a passenger
            reward -= 0.005 * wait  # softer penalty for long waits
            reward -= 0.002 * ride  # mild penalty for long rides
        return reward
