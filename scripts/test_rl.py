"""
Load a trained PPO policy and evaluate one episode on LiftGymEnv.

Usage:
    PYTHONPATH=. python scripts/test_rl.py --model artifacts/ppo_liftsim --spawn_prob 0.05 --seed 999
    # 如需对比基线 ETA 策略
    PYTHONPATH=. python scripts/test_rl.py --baseline --spawn_prob 0.05 --seed 999
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from core.environment import SimulationConfig
from gym_env.env import LiftGymEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a trained PPO policy on LiftSim.")
    parser.add_argument("--model", type=Path, help="Path to saved PPO model.")
    parser.add_argument("--spawn_prob", type=float, default=0.05, help="Passenger spawn probability per second.")
    parser.add_argument("--seed", type=int, default=999, help="Environment seed for evaluation.")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Run ETA baseline instead of a trained model (ignores --model).",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both baseline and model (requires --model) with the same config/seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(spawn_prob=args.spawn_prob, seed=args.seed)

    def run_baseline() -> tuple[float, dict]:
        env = LiftGymEnv(config)
        obs, info = env.reset()
        terminated = False
        truncated = False
        reward_sum = 0.0
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step([0] * config.elevators)
            reward_sum += reward
        metrics = env.sim.metrics.summary()
        env.close()
        return reward_sum, metrics

    def run_model(model_path: Path) -> tuple[float, dict]:
        env = LiftGymEnv(config)
        model = PPO.load(str(model_path), env=env)
        obs, info = env.reset()
        terminated = False
        truncated = False
        reward_sum = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
        metrics = env.sim.metrics.summary()
        env.close()
        return reward_sum, metrics

    if args.compare:
        if not args.model:
            raise SystemExit("Please provide --model when using --compare.")
        base_reward, base_metrics = run_baseline()
        model_reward, model_metrics = run_model(args.model)
        print("=== Baseline (ETA) ===")
        print("reward:", base_reward)
        print("metrics:", base_metrics)
        print("=== Model ===")
        print("reward:", model_reward)
        print("metrics:", model_metrics)
        return

    if args.baseline:
        base_reward, base_metrics = run_baseline()
        print("Baseline episode reward:", base_reward)
        print("Baseline metrics:", base_metrics)
        return

    if not args.model:
        raise SystemExit("Please provide --model path or use --baseline/--compare.")

    model_reward, model_metrics = run_model(args.model)
    print("Episode reward:", model_reward)
    print("Metrics:", model_metrics)


if __name__ == "__main__":
    main()
