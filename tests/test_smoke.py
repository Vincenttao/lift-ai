from core.environment import LiftSimEnvironment, SimulationConfig


def test_environment_runs_steps():
    config = SimulationConfig(horizon_s=10, spawn_prob=0.0, seed=123)
    env = LiftSimEnvironment(config)
    env.reset()
    steps = 0
    while env.step():
        steps += 1
    assert steps == config.horizon_s
    summary = env.metrics.summary()
    # No passengers spawned; metrics should be empty/zero.
    assert summary["completed"] == 0
    assert summary["avg_wait"] is None

