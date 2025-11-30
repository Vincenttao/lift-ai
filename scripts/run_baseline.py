"""
Run a one-off baseline simulation with the default ETA dispatcher and print metrics.

Usage:
    PYTHONPATH=. python scripts/run_baseline.py
"""

from core.environment import LiftSimEnvironment, SimulationConfig


def main() -> None:
    cfg = SimulationConfig(
        floors=18,
        elevators=2,
        capacity=12,
        floor_height_m=3.0,
        speed_mps=1.5,
        dwell_time_s=5,
        spawn_prob=0.1,
        horizon_s=3600,
        seed=42,
    )
    env = LiftSimEnvironment(cfg)
    env.reset()
    terminated = False
    steps = 0
    while not terminated:
        _, _, terminated, _ = env.step()
        steps += 1
    print(f"Simulated steps: {steps}")
    print(env.metrics.summary())


if __name__ == "__main__":
    main()
