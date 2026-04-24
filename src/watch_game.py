"""
Watch a game rendered in the terminal.

By default plays heuristic vs heuristic.
Use --net to pit the latest trained checkpoint against the heuristic.

Usage:
    python src/watch_game.py                          # heuristic vs heuristic
    python src/watch_game.py --net                    # net (red) vs heuristic (blue)
    python src/watch_game.py --net --ckpt checkpoints/stage1_supervised.pt
    python src/watch_game.py --delay 0.05             # faster rendering
"""

import argparse
import pathlib
import sys
import time

import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.training.heuristic import HeuristicAgent


def _latest_checkpoint() -> pathlib.Path | None:
    ckpt_dir = _ROOT / "checkpoints"
    ppo = sorted(ckpt_dir.glob("ppo_step_*.pt"))
    if ppo:
        return ppo[-1]
    stage1 = ckpt_dir / "stage1_supervised.pt"
    return stage1 if stage1.exists() else None


def _load_net(ckpt_path: pathlib.Path):
    from src.models.network import ChineseCheckersNet
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net = ChineseCheckersNet()
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", action="store_true", help="Use trained net as player 0 (red)")
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (default: latest)")
    p.add_argument("--delay", type=float, default=0.15, help="Seconds between moves")
    p.add_argument("--max-steps", type=int, default=800)
    args = p.parse_args()

    heuristic = HeuristicAgent()
    env = ChineseCheckersEnv(n_players=2, render_mode="ansi")

    net = None
    if args.net:
        ckpt_path = pathlib.Path(args.ckpt) if args.ckpt else _latest_checkpoint()
        if ckpt_path is None:
            print("No checkpoint found. Run supervised.py first.")
            sys.exit(1)
        net = _load_net(ckpt_path)
        print(f"Checkpoint: {ckpt_path.name}")
        print("Red = NET   Blue = HEURISTIC\n")
    else:
        print("Red = HEURISTIC   Blue = HEURISTIC\n")

    _, info = env.reset()

    for step in range(args.max_steps):
        env.render()
        time.sleep(args.delay)

        if net is not None and env._turn_idx == 0:
            action = net.act(env, temperature=0.3)
            actor = "NET"
        else:
            action = heuristic.act(env)
            actor = "HEU"

        _, reward, terminated, truncated, info = env.step(action)
        print(f"step {step + 1:>3}  [{actor}]  reward={reward:+.3f}")

        if terminated or truncated:
            env.render()
            if net is not None:
                winner = "NET" if info["rewards"]["red"] > info["rewards"]["blue"] else "HEURISTIC"
            else:
                winner = "red" if info["rewards"]["red"] > info["rewards"]["blue"] else "blue"
            print(f"\nDONE after {step + 1} steps — winner: {winner}")
            break


if __name__ == "__main__":
    main()
