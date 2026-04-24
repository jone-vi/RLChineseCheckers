"""
Evaluate the latest (or a specific) checkpoint against the heuristic agent.

Designed to be run in a separate terminal while PPO training is in progress.
Auto-detects the latest checkpoint — run it repeatedly to track improvement.

Usage:
    python src/eval.py                        # latest checkpoint, 30 games
    python src/eval.py --games 50             # more games for accuracy
    python src/eval.py --ckpt checkpoints/stage1_supervised.pt
    watch -n 120 python src/eval.py           # re-run every 2 minutes
"""

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.models.network import ChineseCheckersNet
from src.training.heuristic import HeuristicAgent


def _latest_checkpoint() -> pathlib.Path | None:
    ckpt_dir = _ROOT / "checkpoints"
    ppo = sorted(ckpt_dir.glob("ppo_step_*.pt"))
    if ppo:
        return ppo[-1]
    stage1 = ckpt_dir / "stage1_supervised.pt"
    return stage1 if stage1.exists() else None


def evaluate(ckpt_path: pathlib.Path, n_games: int, temperature: float = 0.3) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net = ChineseCheckersNet()
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    step = ckpt.get("step", f"epoch {ckpt.get('epoch', '?')}")
    print(f"Checkpoint : {ckpt_path.name}  ({step})")
    print(f"Games      : {n_games}")
    print()

    heuristic = HeuristicAgent()
    env = ChineseCheckersEnv(n_players=2)

    wins = 0
    episode_lengths = []
    t0 = time.perf_counter()

    for g in range(n_games):
        _, info = env.reset()
        steps = 0
        for _ in range(800):
            if env._turn_idx == 0:
                action = net.act(env, temperature=temperature)
            else:
                action = heuristic.act(env)
            _, _, done, trunc, info = env.step(action)
            steps += 1
            if done or trunc:
                if info["rewards"]["red"] > info["rewards"]["blue"]:
                    wins += 1
                episode_lengths.append(steps)
                break
        # Progress dot every 10 games
        if (g + 1) % 10 == 0:
            print(f"  {g + 1}/{n_games} games  ({wins} wins so far)")

    elapsed = time.perf_counter() - t0
    win_rate = wins / n_games
    avg_len = np.mean(episode_lengths) if episode_lengths else 0

    print()
    print(f"Win rate   : {wins}/{n_games} = {win_rate:.1%}")
    print(f"Avg length : {avg_len:.0f} steps")
    print(f"Time       : {elapsed:.1f}s")
    print()

    if win_rate >= 0.60:
        print(">> Excellent — well above Stage 1 baseline (40%)")
    elif win_rate >= 0.40:
        print(">> Good — at or above Stage 1 baseline")
    elif win_rate >= 0.20:
        print(">> Learning — below Stage 1 baseline, PPO still converging")
    else:
        print(">> Weak — may need more PPO steps or check entropy collapse")


def main():
    p = argparse.ArgumentParser(description="Evaluate latest checkpoint vs heuristic")
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (default: latest)")
    p.add_argument("--games", type=int, default=30, help="Number of eval games")
    p.add_argument("--temperature", type=float, default=0.3)
    args = p.parse_args()

    ckpt_path = pathlib.Path(args.ckpt) if args.ckpt else _latest_checkpoint()
    if ckpt_path is None:
        print("No checkpoint found. Run supervised.py first.")
        sys.exit(1)

    evaluate(ckpt_path, n_games=args.games, temperature=args.temperature)


if __name__ == "__main__":
    main()
