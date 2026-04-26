"""
Stage 1 data generation.

Runs heuristic self-play games and saves (observation, action, outcome) tuples
to an HDF5 file for supervised pre-training.

Dataset layout (data/stage1.h5):
    obs      float32 [N, 1089]  — board observation from acting player's POV
    actions  int16   [N]        — encoded action = pin_id * 121 + canonical_dest_idx
    outcomes float32 [N]        — game outcome for acting player in [-1, 1]

Usage:
    python src/training/data_gen.py
    python src/training/data_gen.py --n-games 500 --out data/stage1_debug.h5
"""

import argparse
import pathlib
import random
import sys
import time

import h5py
import numpy as np
from tqdm import tqdm

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.training.heuristic import HeuristicAgent

# ---------------------------------------------------------------------------
# Diversity constants
# ---------------------------------------------------------------------------
_WARMUP_NONE   = 0    # 50 % — standard start, collect from move 0
_WARMUP_SHORT  = 6    # 30 % — 6 random moves (≈ 3 per player) before heuristic
_WARMUP_LONG   = 20   # 20 % — 20 random moves (≈ 10 per player) before heuristic


def _warmup(env: ChineseCheckersEnv, n_random_moves: int) -> bool:
    """
    Apply n_random_moves random legal moves to diversify the starting position.
    Returns False if the game terminates during warmup (discard this game).
    """
    for _ in range(n_random_moves):
        mask  = env._build_action_mask()
        legal = np.where(mask)[0]
        if len(legal) == 0:
            return False
        action = int(np.random.choice(legal))
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            return False
    return True


def _pick_warmup(rng: random.Random) -> int:
    """Return warmup length according to diversity schedule."""
    r = rng.random()
    if r < 0.50:
        return _WARMUP_NONE
    elif r < 0.80:
        return _WARMUP_SHORT
    else:
        return _WARMUP_LONG


def run_one_game(
    env:    ChineseCheckersEnv,
    agent:  HeuristicAgent,
    rng:    random.Random,
    sample_rate: float,
) -> tuple[list, bool]:
    """
    Run one complete heuristic self-play game.

    Returns a list of (obs, action, outcome) tuples sampled at `sample_rate`
    from the full game trajectory.  Returns an empty list if the game ends
    during the diversity warmup phase.
    """
    obs, info = env.reset()

    # --- Diversity warmup (random moves, no data collected) ---
    n_warmup = _pick_warmup(rng)
    if n_warmup > 0:
        ok = _warmup(env, n_warmup)
        if not ok:
            return [], False
        # Rebuild obs from new position
        obs = env._build_observation()
        info = {
            "action_mask":    env._build_action_mask(),
            "current_player": env._active_colours[env._turn_idx],
            "rewards":        {c: 0.0 for c in env._active_colours},
            "no_legal_moves": False,
        }

    # --- Collect (obs, action, player) tuples during play ---
    trajectory: list = []          # list of (obs_copy, action, colour)

    for _ in range(1000):          # hard cap; well above any real game length
        current = info["current_player"]
        action  = agent.act(env)
        trajectory.append((obs.copy(), action, current))
        obs, _, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    if not trajectory:
        return [], False

    # --- Assign outcomes ---
    final_rewards = info["rewards"]         # {colour: float} — last step's rewards

    # Supervised value targets should encode the competitive result, not the
    # current PPO reward shaping.  Winners get +1.  If the game ends with a
    # winner, all non-winners get the same negative rank target.  If it ends
    # without a winner, rank players by remaining normalized distance.
    winner = next((c for c, r in final_rewards.items() if r >= 10.0), None)
    if winner is not None:
        loser_target = -1.0 / max(1, len(env._active_colours) - 1)
        outcome_map = {
            c: 1.0 if c == winner else loser_target
            for c in env._active_colours
        }
        is_draw = False
    else:
        ranked = sorted(
            env._active_colours,
            key=lambda c: env._prev_dist_sq[c] / max(env._d_max[c], 1.0),
        )
        if len(ranked) == 1:
            outcome_map = {ranked[0]: 0.0}
        else:
            outcome_map = {
                c: 1.0 - 2.0 * rank / (len(ranked) - 1)
                for rank, c in enumerate(ranked)
            }
        is_draw = True

    # --- Sample 5 % ---
    n_sample = max(1, round(len(trajectory) * sample_rate))
    sampled  = rng.sample(trajectory, min(n_sample, len(trajectory)))

    result = []
    for (o, a, colour) in sampled:
        result.append((o, a, outcome_map[colour]))
    return result, is_draw


def generate(
    n_games:     int   = 20_000,
    sample_rate: float = 0.05,
    out_path:    str   = "data/stage1.h5",
    player_mix:  dict  = None,   # {n_players: fraction}, e.g. {2: 0.8, 4: 0.1, 6: 0.1}
    seed:        int   = 42,
):
    """
    Generate n_games heuristic self-play games and write examples to HDF5.

    player_mix controls how many games are played at each player count.
    Fractions are normalised so they don't need to sum exactly to 1.0.
    Default (player_mix=None) uses 2-player only for backwards compatibility.

    Returns the path to the written file.
    """
    if player_mix is None:
        player_mix = {2: 1.0}

    # Normalise fractions and compute per-count game counts.
    total_frac  = sum(player_mix.values())
    counts      = {}
    assigned    = 0
    items       = sorted(player_mix.items())          # deterministic order
    for i, (np_, frac) in enumerate(items):
        if i == len(items) - 1:
            counts[np_] = n_games - assigned          # absorb rounding remainder
        else:
            counts[np_] = round(n_games * frac / total_frac)
            assigned   += counts[np_]

    rng = random.Random(seed)
    np.random.seed(seed)

    # Pre-allocate storage — generous upper bound, trimmed at end.
    max_examples = n_games * 200
    obs_buf      = np.empty((max_examples, ChineseCheckersEnv.OBS_SIZE), dtype=np.float32)
    act_buf      = np.empty(max_examples,  dtype=np.int16)
    out_buf      = np.empty(max_examples,  dtype=np.float32)
    ptr          = 0
    skipped      = 0
    draws        = 0
    t0           = time.perf_counter()

    out_file = _ROOT / out_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    total_done = 0
    with tqdm(total=n_games, unit="game", desc="Generating") as bar:
        for n_players, n_batch in counts.items():
            agent = HeuristicAgent()
            env   = ChineseCheckersEnv(n_players=n_players)
            bar.set_description(f"Generating ({n_players}p)")

            games_done = 0
            while games_done < n_batch:
                tuples, is_draw = run_one_game(env, agent, rng, sample_rate)
                if not tuples:
                    skipped += 1
                    continue

                if is_draw:
                    draws += 1

                for (o, a, outcome) in tuples:
                    if ptr >= max_examples:
                        break
                    obs_buf[ptr] = o
                    act_buf[ptr] = a
                    out_buf[ptr] = outcome
                    ptr += 1

                games_done += 1
                total_done += 1
                bar.update(1)
                bar.set_postfix(examples=ptr, skipped=skipped, draws=draws)

    elapsed = time.perf_counter() - t0
    n = ptr

    # Shuffle before writing (removes per-game and per-player-count ordering)
    idx = np.random.permutation(n)

    mix_str = ",".join(f"{k}p×{v}" for k, v in counts.items())
    print(f"\nWriting {n:,} examples to {out_file} …")
    with h5py.File(out_file, "w") as f:
        f.create_dataset("obs",      data=obs_buf[idx[:n]],   compression="gzip", compression_opts=4)
        f.create_dataset("actions",  data=act_buf[idx[:n]],   compression="gzip", compression_opts=4)
        f.create_dataset("outcomes", data=out_buf[idx[:n]],   compression="gzip", compression_opts=4)
        f.attrs["n_games"]     = n_games
        f.attrs["sample_rate"] = sample_rate
        f.attrs["player_mix"]  = str(counts)
        f.attrs["seed"]        = seed
        f.attrs["action_encoding"] = "canonical_destination_v2"

    draw_pct = 100.0 * draws / max(1, total_done)
    print(f"Done. {n:,} examples | mix: {mix_str} | "
          f"{skipped} skipped | draws: {draws} ({draw_pct:.1f}%) | "
          f"{elapsed:.1f}s ({n_games/elapsed:.1f} games/s)")
    return str(out_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_mix(s: str) -> dict:
    """
    Parse a player-mix string into {n_players: fraction}.
    Format: "2:0.8,4:0.1,6:0.1"  or  "2:8,4:1,6:1"  (raw counts, auto-normalised)
    """
    result = {}
    for part in s.split(","):
        k, v = part.strip().split(":")
        result[int(k)] = float(v)
    return result


def _parse():
    p = argparse.ArgumentParser(description="Stage 1 data generation")
    p.add_argument("--n-games",     type=int,   default=20_000)
    p.add_argument("--sample-rate", type=float, default=0.05)
    p.add_argument("--out",         type=str,   default="data/stage1.h5")
    p.add_argument("--player-mix",  type=str,   default="2:0.4,3:0.15,4:0.15,5:0.15,6:0.1",
                   help='Player count mix, e.g. "2:0.4,3:0.15,4:0.15,5:0.15,6:0.1"')
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    generate(
        n_games     = args.n_games,
        sample_rate = args.sample_rate,
        out_path    = args.out,
        player_mix  = _parse_mix(args.player_mix),
        seed        = args.seed,
    )
