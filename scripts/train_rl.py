"""
Train a PPO agent on LiftGymEnv with configurable spawn probability and seed.

Usage:
    PYTHONPATH=. python scripts/train_rl.py --spawn_prob 0.05 --timesteps 200000 --seed 42 --out artifacts/ppo_liftsim
"""

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from core.environment import SimulationConfig
from gym_env.env import LiftGymEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on LiftSim.")
    parser.add_argument("--spawn_prob", type=float, default=0.05, help="Passenger spawn probability per second.")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total training timesteps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for env and agent.")
    parser.add_argument("--tag", type=str, default="", help="Optional tag for run naming.")
    parser.add_argument("--out", type=Path, default=Path("artifacts"), help="Base output directory.")
    parser.add_argument("--eval_freq", type=int, default=50_000, help="Evaluate every N steps; 0 to disable.")
    parser.add_argument("--eval_episodes", type=int, default=3, help="Episodes per evaluation run.")
    parser.add_argument("--lr", type=float, default=1e-4, help="PPO learning rate.")
    parser.add_argument(
        "--n_steps",
        type=int,
        default=4096,
        help="Rollout steps per env per update (total batch = n_steps * num_envs).",
    )
    parser.add_argument("--batch_size", type=int, default=1024, help="PPO batch size.")
    parser.add_argument("--device", type=str, default="auto", help="Device for PPO ('auto', 'cuda', 'cpu').")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs for sampling.")
    parser.add_argument(
        "--vec_env",
        type=str,
        default="subproc",
        choices=["subproc", "dummy"],
        help="Vector env type: subproc (multiprocess) or dummy (single process).",
    )
    parser.add_argument("--no_tb", action="store_true", help="Disable TensorBoard logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_name = f"ppo_liftsim_{timestamp}"
    if args.tag:
        run_name += f"_{args.tag}"
    out_dir = args.out / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(spawn_prob=args.spawn_prob, seed=args.seed)

    def make_env_fn():
        def _init():
            return LiftGymEnv(config)

        return _init

    if args.vec_env == "subproc" and args.num_envs > 1:
        env = SubprocVecEnv([make_env_fn() for _ in range(args.num_envs)])
    else:
        env = DummyVecEnv([make_env_fn() for _ in range(args.num_envs)])
    env = VecMonitor(env)
    tb_log_dir = None
    if not args.no_tb:
        if importlib.util.find_spec("tensorboard") is not None:
            tb_log_dir = out_dir / "tb"
        else:
            print("tensorboard not installed, disabling TensorBoard logging.")
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(tb_log_dir) if tb_log_dir else None,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=args.device,
    )

    callback = None
    if args.eval_freq > 0:
        eval_env = VecMonitor(DummyVecEnv([make_env_fn()]))
        callback = EvalCallback(
            eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
        )
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(str(out_dir / "model"))
    # Save run metadata for comparison
    meta = {
        "spawn_prob": args.spawn_prob,
        "timesteps": args.timesteps,
        "seed": args.seed,
        "tag": args.tag,
        "timestamp": timestamp,
        "run_name": run_name,
        "lr": args.lr,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "device": args.device,
        "tensorboard": False if args.no_tb else bool(tb_log_dir),
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    env.close()
    print(f"Model and metadata saved to {out_dir}")


if __name__ == "__main__":
    main()
