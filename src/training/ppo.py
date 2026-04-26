"""
Stage 2 PPO self-play training.

Loads Stage 1 weights and refines via PPO with:
- 8 parallel mixed 2-6-player environments
- 128-step rollouts per env → 1024+ transitions per update
- GAE advantage estimation (γ=0.99, λ=0.95)
- Clipped PPO objective (ε=0.2), entropy bonus, value loss
- Whole-game live self-play plus capped frozen-pool games
- Pool promotion gated by 2-6p win and draw/limit rates
- Checkpoints every 100K steps

Exploration schedule per game:
  all moves   : temperature 0.3 (matches eval; prevents high-T early-move cycles)

Usage:
    python -m src.training.ppo
    python -m src.training.ppo --max-steps 50000 --device cpu   # smoke test
    python -m src.training.ppo --max-steps 1000 --debug          # dense logging
"""

import argparse
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.env.chinese_checkers_env import ChineseCheckersEnv, COLOUR_ORDER
from src.models.network import ChineseCheckersNet
from src.training.heuristic import HeuristicAgent
from src.training.opponent_pool import OpponentPool

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
ACTION_ENCODING = "canonical_destination_v2"
N_ENVS = 8
ROLLOUT_STEPS = 128          # steps per env per update → ~1024 transitions
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.1               # fine-tuning from pretrained policy: 0.1, not 0.2
CLIP_EPS_START = 0.003       # very tight start: Stage 1 policy is near-deterministic (ent≈0.13),
                             # and Adam's first step on the policy head is ≈±LR per parameter
                             # (no prior statistics → sign-normalised step), which is large enough
                             # to cause a catastrophic entropy jump (0.13→1.0) at clip=0.01.
                             # 0.003 limits per-step ratio change to 0.3%, preventing the first-
                             # minibatch explosion. Ramps to 0.1 over POLICY_RAMPUP_ROLLOUTS.
POLICY_RAMPUP_ROLLOUTS = 50  # rollouts after value warmup before encoder is unfrozen and clip hits CLIP_EPS
                             # Phase sequence:
                             #   warmup (WARMUP_ROLLOUTS): encoder+policy frozen, value head calibrates
                             #   rampup (POLICY_RAMPUP_ROLLOUTS): policy head unfrozen, encoder still frozen,
                             #     clip ramps 0.003→0.1  — prevents encoder-coupling drift (encoder changes
                             #     shift policy head outputs even without policy gradient, bypassing clip+KL)
                             #   full training: everything unfrozen, clip=0.1
                             # 400 (doubled from 200) so the very tight CLIP_EPS_START=0.003 has enough
                             # rollouts to ramp to a useful clip value while staying conservative early.
ENT_COEF = 0.0               # Stage 1 already has sufficient diversity (ent≈0.13); no bonus needed.
                             # Any push toward exploration is pure instability when fine-tuning from
                             # a supervised prior — the policy should refine, not explore.
ENT_PENALTY_COEF = 5.0       # at entropy=1.7 (above ceiling 1.5): penalty=5.0×0.20=1.0, which is
                             # 3-5× the typical policy_loss of 0.05-0.1 — strongly enforces the ceiling.
MAX_ENT = 1.5                # hard ceiling; above this, recovery mode skips policy gradient entirely.
                             # Raised from 0.60 to 1.5: with WARMUP_ROLLOUTS=50 (51K steps), the policy
                             # head has had no time to lower its Stage 1 entropy before policy updates
                             # begin.  Measured post-warmup entropy is 1.07–1.65, so MAX_ENT=0.60
                             # fires recovery on 11/16 minibatches from rollout 1 — identical to the
                             # broken 0.35 setting.  Recovery concentrates the policy on Stage 1's
                             # current high-probability actions (which include cyclic moves), causing
                             # win→0% in 3 rollouts.  At 1.5, recovery only triggers if entropy
                             # explosively increases above the Stage 1 baseline, which PPO with
                             # KL stopping (TARGET_KL=0.01) and small initial clip (0.003) prevents.
                             # Entropy decreases naturally toward 0.4–0.6 over 30–50 rollouts as the
                             # policy learns to distinguish win vs cycle trajectories.  The floor (0.22)
                             # still prevents collapse.
ENT_FLOOR = 0.22             # minimum entropy (raised from 0.15); prevents the near-deterministic
                             # regime (ent≈0.13) where the masking nonlinearity is unstable.  The
                             # masking maps 1210 raw logits to ~50 valid actions; a tiny logit shift
                             # can reroute large probability mass across the mask boundary, causing a
                             # 0.15→0.69 entropy jump in one minibatch even with clip=0.012.  At
                             # ent=0.22 the policy is still very concentrated (exp(0.22)=1.25 effective
                             # branching factor vs exp(0.15)=1.16) but far enough from the boundary
                             # that gradient steps don't trigger the nonlinearity catastrophically.
ENT_FLOOR_COEF = 3.0         # at ent=0.15 (below floor 0.22): penalty=3.0×0.07=0.21 ≈ 4× policy_loss.
                             # Stronger than before (was 2.0) to compensate for the raised floor.
VF_COEF = 0.5
LR = 1e-4                    # was 3e-4 — lower LR preserves Stage 1 knowledge during fine-tuning
N_EPOCHS = 1                 # policy epochs per rollout — deliberately 1 to prevent large policy drift
N_VALUE_EXTRA = 3            # additional value-head-only epochs per rollout (encoder+policy frozen).
                             # v_loss oscillated 0.07-0.17 throughout rampup because the policy changes every
                             # rollout and 1 epoch of value updates can't track it; extra epochs hold it low,
                             # producing accurate advantages and preventing the noisy-grad→entropy-drift loop.
MINIBATCH = 64              # increased from 64; larger batches dilute rare high-negative-advantage
                             # loss samples (2% of episodes), reducing per-step gradient variance
                             # and halving update frequency per rollout (~4 vs ~8 with 64).
                             # Stays well below B≈512 so each rollout still gets meaningful updates.
TARGET_KL = 0.01             # stop epoch early if policy changes too much
SELFPLAY_GAME_RATIO = 0.00   # whole-game live self-play.  All seats use the current net,
                             # and all seats' trajectories train the shared policy.
                             # Reduced from 0.50: in N-player self-play all N seats generate
                             # transitions, so 50% self-play by game count → ~86% of batch
                             # transitions come from self-play.  If the policy cycles, 86% of
                             # training data is cyclic.  At 0.30 it's ~72%, and pool/heuristic
                             # games (which never cycle) form a 28% stable baseline signal.
MAX_FROZEN_NET_OPPONENTS = 1 # in pool games, cap stochastic frozen nets per game.
                             # The remaining seats use the heuristic, avoiding the
                             # multiplayer compounding of draw-prone checkpoint policies.
WARMUP_ROLLOUTS = 50         # ~200K steps: freeze policy, let value head calibrate on real game dynamics
                             # before policy updates begin.  200 (doubled from 100) because:
                             # (a) the first pool update at step ~100K changes opponent dynamics, causing
                             #     v_loss to spike — the head needs time to recalibrate on the new games
                             # (b) at 100 rollouts, v_loss was still 0.10+ when policy updates started,
                             #     producing noisy advantages that drove the entropy explosion
CYCLE_TERMINAL_PENALTY = 1.0  # applied to live players whose own piece positions cycled
                              # (any pos_count >= 2 in _player_pos_counts).  Per-player
                              # attribution avoids penalising for cycles triggered entirely
                              # by an opponent.  Combined with the per-move revisit shaping
                              # penalty in the env, this makes the cycle signal non-zero
                              # even in pool games (where all_live=False).
MAX_STEPS = 5_000_000
REWARD_SCALE = 10.0          # divide raw rewards; terminal win→+1, loss→-1
CKPT_EVERY = 100_000
EVAL_GAMES = 50
EVAL_GAMES_PER_COUNT = 8
EVAL_PLAYER_COUNTS = (2, 3, 4, 5, 6)
PROMOTE_MIN_WIN_BY_COUNT = {2: 0.50, 3: 0.30, 4: 0.20, 5: 0.15, 6: 0.12}
PROMOTE_MAX_DRAW_RATE = 0.25  # max(cycle + limit) for every evaluated player count.
POST_PROMO_WARMUP = 20       # rollouts of value-head-only update after each pool promotion.
PLAYER_MIX: dict[int, float] = {2: 0.35, 3: 0.13, 4: 0.20, 5: 0.12, 6: 0.20}
                               # distribution of N-player game counts per episode.
                               # Tilted toward 2p (simple, strong win signal) and 4p/6p.
                             # When a new checkpoint enters the pool, game dynamics shift suddenly:
                             # the learning policy now faces a competitive opponent, so typical
                             # returns drop (e.g. 0.99→0.50).  The value head, calibrated on the
                             # old high-win games (V(s)≈0.9), produces systematically wrong
                             # advantages for the new regime, pushing the policy in the wrong
                             # direction and cascading into cycles.  20 rollouts (~20K steps) lets
                             # v_loss converge to the new distribution before policy updates resume.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_temperature(move_count: int) -> float:
    """Fixed temperature matching eval behavior — prevents high-T early moves from causing cycles."""
    return 0.3


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float = GAMMA,
    lam: float = GAE_LAMBDA,
):
    """
    Standard single-trajectory GAE.

    For 2-player games with interleaved P0/P1 experiences, the simplified
    non-negated formulation is used (Gu & Adhikari 2024).  Terminal rewards
    dominate, so bootstrapping bias across player boundaries is small.

    Returns:
        advantages [T], returns [T]
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        next_val = last_value if t == T - 1 else values[t + 1]
        next_done = 0.0 if t == T - 1 else float(dones[t + 1])
        delta = rewards[t] + gamma * next_val * (1.0 - float(dones[t])) - values[t]
        gae = delta + gamma * lam * (1.0 - float(dones[t])) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def evaluate(
    net: ChineseCheckersNet,
    opponent,
    n_games: int = EVAL_GAMES,
    device: str = "cpu",
) -> float:
    """
    Evaluate net (player 0, red) vs opponent (player 1, blue).
    Returns win rate for the learning network.
    """
    metrics = _evaluate_match(
        net,
        opponent_factory=lambda _seat: opponent,
        n_players=2,
        n_games=n_games,
        device=device,
    )
    return metrics["win_rate"]


def _term_type(info: dict, truncated: bool) -> str:
    rewards = info.get("rewards", {})
    if truncated:
        return "limit"
    if rewards.get("red", 0.0) >= 10.0:
        return "win"
    if any(r >= 10.0 for c, r in rewards.items() if c != "red"):
        return "loss"
    return "cycle"


def _evaluate_match(
    net: ChineseCheckersNet,
    opponent_factory,
    n_players: int,
    n_games: int,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate red/current net against a fixed opponent factory."""
    net.eval()
    env = ChineseCheckersEnv(n_players=n_players)
    counts = {"win": 0, "loss": 0, "cycle": 0, "limit": 0}
    progress_sum = 0.0

    for _ in range(n_games):
        _, info = env.reset(n_players=n_players)
        opponents = [opponent_factory(i) for i in range(n_players - 1)]
        for _ in range(2000):
            if env._turn_idx == 0:
                action = net.act(env, temperature=0.3)
            else:
                action = opponents[env._turn_idx - 1].act(env)
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                term = _term_type(info, truncated)
                counts[term] += 1
                red_r = info["rewards"].get("red", 0.0)
                progress_sum += red_r - 10.0 if term == "win" else red_r
                break
        else:
            counts["limit"] += 1

    net.train()
    total = max(n_games, 1)
    return {
        "win_rate": counts["win"] / total,
        "loss_rate": counts["loss"] / total,
        "cycle_rate": counts["cycle"] / total,
        "limit_rate": counts["limit"] / total,
        "draw_rate": (counts["cycle"] + counts["limit"]) / total,
        "progress": progress_sum / total,
    }


def _evaluate_promotion_suite(
    net: ChineseCheckersNet,
    pool: OpponentPool,
    heuristic: HeuristicAgent,
    device: str = "cpu",
) -> tuple[bool, list[str]]:
    """
    Gate pool promotion on all player counts.

    A checkpoint that is strong in 2p but causes 4-6p cycles should not be added
    to the pool, because each additional frozen seat multiplies draw risk.
    """
    latest = pool.latest()
    opponent_sets = [("heur", heuristic)]
    if latest is not None:
        opponent_sets.append(("pool", latest))

    promote = True
    lines: list[str] = []
    for label, opponent in opponent_sets:
        parts = []
        for n_players in EVAL_PLAYER_COUNTS:
            metrics = _evaluate_match(
                net,
                opponent_factory=lambda _seat, opp=opponent: opp,
                n_players=n_players,
                n_games=EVAL_GAMES_PER_COUNT,
                device=device,
            )
            min_win = PROMOTE_MIN_WIN_BY_COUNT[n_players]
            ok = (
                metrics["win_rate"] >= min_win
                and metrics["draw_rate"] <= PROMOTE_MAX_DRAW_RATE
            )
            promote = promote and ok
            parts.append(
                f"{n_players}p W={metrics['win_rate']:.2f} "
                f"D={metrics['draw_rate']:.2f} P={metrics['progress']:.2f}"
            )
        lines.append(f"[Eval:{label}] " + " | ".join(parts))
    return promote, lines


def _sample_game_actors(
    pool: OpponentPool,
    net: ChineseCheckersNet,
    n_players: int,
) -> tuple[list, str]:
    """
    Choose who controls each seat.

    Self-play games use the live net for every seat, yielding true parameter
    sharing data.  Pool games always keep red/live, but cap frozen network
    opponents and fill remaining seats with the heuristic.
    """
    if random.random() < SELFPLAY_GAME_RATIO:
        return [net for _ in range(n_players)], "self"

    actors = [net] + [pool.heuristic for _ in range(n_players - 1)]
    n_frozen = min(MAX_FROZEN_NET_OPPONENTS, n_players - 1, len(pool))
    if n_frozen > 0:
        for seat in random.sample(range(1, n_players), n_frozen):
            frozen = pool.sample_frozen()
            if frozen is not None:
                actors[seat] = frozen
    return actors, "pool"


def _bootstrap_value_for_colour(
    net: ChineseCheckersNet,
    env: ChineseCheckersEnv,
    colour: str,
    dev: torch.device,
) -> float:
    """Evaluate the current board from a specific player's perspective."""
    if colour not in env._active_colours:
        return 0.0
    old_turn = env._turn_idx
    try:
        env._turn_idx = env._active_colours.index(colour)
        mask = env._build_action_mask()
        if mask.sum() == 0:
            return 0.0
        obs = env._build_observation()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
            mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(dev)
            _, val = net(obs_t, mask_t)
        return float(val.item())
    finally:
        env._turn_idx = old_turn


def _new_transition_buffers() -> dict[str, dict[str, list]]:
    return {
        colour: {
            "obs": [],
            "actions": [],
            "logprobs": [],
            "values": [],
            "rewards": [],
            "dones": [],
            "masks": [],
        }
        for colour in COLOUR_ORDER
    }


def _credit_terminal_rewards(
    env: ChineseCheckersEnv,
    buffers_by_colour: dict[str, dict[str, list]],
    actors_for_env: list,
    net: ChineseCheckersNet,
    info: dict,
    acting_colour: str,
    acting_was_live: bool,
    truncated: bool,
) -> None:
    """Attach terminal rewards to each live player's latest transition."""
    rewards = info.get("rewards", {})
    is_cycle = (
        not truncated
        and all(r < 10.0 for r in rewards.values())
    )
    all_live = all(actor is net for actor in actors_for_env)

    for seat, colour in enumerate(env._active_colours):
        if actors_for_env[seat] is not net:
            continue

        buffers = buffers_by_colour[colour]
        if not buffers["rewards"]:
            continue

        if colour != acting_colour or not acting_was_live:
            buffers["rewards"][-1] += rewards.get(colour, 0.0) / REWARD_SCALE

        if is_cycle and CYCLE_TERMINAL_PENALTY > 0.0:
            # Penalise any live player whose own piece positions cycled.
            # Checking per-player counts avoids blaming players for cycles
            # triggered entirely by an opponent's repeated positions.
            player_pos_counts = env._player_pos_counts.get(colour, {})
            player_contributed = any(v >= 2 for v in player_pos_counts.values())
            if player_contributed or all_live:
                buffers["rewards"][-1] -= CYCLE_TERMINAL_PENALTY / REWARD_SCALE

        buffers["dones"][-1] = 1.0


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    stage1_ckpt: str = "checkpoints/stage1_supervised.pt",
    ckpt_dir: str = "checkpoints",
    max_steps: int = MAX_STEPS,
    device: str | None = None,
    resume: str | None = None,
    debug: bool = False,
) -> None:
    # -- Device --
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)
    print(f"Device: {dev}")

    # -- Load weights (resume from PPO checkpoint or start from Stage 1) --
    net = ChineseCheckersNet().to(dev)
    optimiser = torch.optim.Adam(net.parameters(), lr=LR)
    resume_step = 0

    if resume:
        resume_path = _ROOT / resume
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=dev, weights_only=False)
        if ckpt.get("action_encoding") != ACTION_ENCODING:
            raise ValueError(
                f"Resume checkpoint uses action_encoding={ckpt.get('action_encoding')!r}; "
                "regenerate Stage 1 and PPO checkpoints with the canonical-action env."
            )
        net.load_state_dict(ckpt["state_dict"])
        optimiser.load_state_dict(ckpt["optimiser"])
        resume_step = ckpt.get("step", 0)
        net.train()
        print(f"Resumed from {resume_path.name} at step {resume_step:,}")
    else:
        ckpt_path = _ROOT / stage1_ckpt
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Stage 1 checkpoint not found: {ckpt_path}\n"
                "Run supervised.py first."
            )
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        if ckpt.get("action_encoding") != ACTION_ENCODING:
            raise ValueError(
                f"Stage 1 checkpoint uses action_encoding={ckpt.get('action_encoding')!r}; "
                "rerun data_gen.py and supervised.py before PPO."
            )
        net.load_state_dict(ckpt["state_dict"])
        net.train()
        print(f"Loaded Stage 1 weights (epoch {ckpt.get('epoch', '?')})")

    # -- Environments --
    envs = [ChineseCheckersEnv(n_players=2) for _ in range(N_ENVS)]

    # -- Opponent pool --
    heuristic = HeuristicAgent(seed=42)
    pool = OpponentPool(heuristic_agent=heuristic, pool_size=5, ckpt_dir=ckpt_dir)
    # Seed the pool with Stage 1 so there is a stable frozen opponent from step 0.
    # Without this the pool is empty for the first 100K steps and 70% of games are
    # pure self-play against the same network that is currently degrading.
    if not resume:
        pool.add(net.state_dict(), step=0)

    # -- Tracking (global_step needed before env init for mix_ratio) --
    global_step = resume_step

    # -- Per-env state --
    obs_buf: list = [None] * N_ENVS       # current observation after last step
    info_buf: list = [None] * N_ENVS      # current info after last step
    actors: list = [[] for _ in range(N_ENVS)]  # List[List[Agent]]: one actor per seat
    game_modes = ["self"] * N_ENVS
    n_players_env: list[int] = [2] * N_ENVS  # current player count for each env

    _mix_keys    = list(PLAYER_MIX.keys())
    _mix_weights = list(PLAYER_MIX.values())

    for e in range(N_ENVS):
        n_p = random.choices(_mix_keys, weights=_mix_weights)[0]
        n_players_env[e] = n_p
        obs_buf[e], info_buf[e] = envs[e].reset(n_players=n_p)
        actors[e], game_modes[e] = _sample_game_actors(pool, net, n_p)

    # -- Per-env, per-colour rollout storage.  Each live-controlled player gets
    # its own value trajectory so GAE bootstraps from that player's perspective.
    per_env_data = [_new_transition_buffers() for _ in range(N_ENVS)]

    # -- Tracking --
    ckpt_save_dir = _ROOT / ckpt_dir
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)
    rollout_count = 0
    post_promo_countdown = 0   # rollouts remaining in post-promotion value warmup (0 = inactive)
    last_ckpt_step = resume_step
    last_eval_step = resume_step
    ep_wins: list[float] = []     # rolling window: 1=win, 0=loss for learning net
    ep_rewards: list[float] = []  # rolling window: per-episode total reward
    ep_term_types: list[str] = [] # rolling window: "win", "loss", "cycle", "limit"
    ep_progress: list[float] = [] # rolling window: distance progress reward (proxy for piece advancement)

    # Per-env episode reward accumulator
    ep_rew_accum = [0.0] * N_ENVS

    t_start = time.perf_counter()
    print(f"Starting PPO training. Max steps: {max_steps:,}")

    while global_step < max_steps:

        n_players_hist: dict[int, int] = {n: 0 for n in PLAYER_MIX}

        # ----------------------------------------------------------------
        # ROLLOUT PHASE
        # ----------------------------------------------------------------
        for e in range(N_ENVS):
            for buffers in per_env_data[e].values():
                for values in buffers.values():
                    values.clear()

        net.eval()

        for _step in range(ROLLOUT_STEPS):
            for e, env in enumerate(envs):
                obs = obs_buf[e]
                info = info_buf[e]
                mask = info["action_mask"]

                colour = env._active_colours[env._turn_idx]
                actor = actors[e][env._turn_idx]
                is_live = (actor is net)

                if is_live:
                    move_count = env._move_counts[colour]
                    temp = _get_temperature(move_count)

                    # Compute action + log_prob + value under learning net
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
                    mask_t = torch.from_numpy(mask).float().unsqueeze(0).to(dev)

                    with torch.no_grad():
                        if temp != 1.0:
                            # Sample with temperature; compute log_prob at temp=1.0
                            # so the PPO ratio is based on unscaled policy.
                            logits, val_t = net(obs_t, mask_t)
                            scaled_dist = torch.distributions.Categorical(
                                logits=logits / temp
                            )
                            action_t = scaled_dist.sample()
                            # Re-evaluate log_prob at temperature=1.0
                            _, log_prob_t, _, val_t = net.get_action_and_value(
                                obs_t, mask_t, action=action_t
                            )
                        else:
                            action_t, log_prob_t, _, val_t = net.get_action_and_value(
                                obs_t, mask_t
                            )

                    action = int(action_t.item())
                    log_prob_val = float(log_prob_t.item())
                    value_val = float(val_t.item())

                    next_obs, raw_r, done, trunc, next_info = env.step(action)
                    reward = raw_r / REWARD_SCALE
                    global_step += 1
                    if colour == "red":
                        ep_rew_accum[e] += reward

                    # Store transition
                    buffers = per_env_data[e][colour]
                    buffers["obs"].append(obs.copy())
                    buffers["actions"].append(action)
                    buffers["logprobs"].append(log_prob_val)
                    buffers["values"].append(value_val)
                    buffers["rewards"].append(reward)
                    buffers["dones"].append(float(done or trunc))
                    buffers["masks"].append(mask.copy())

                    obs_buf[e] = next_obs
                    info_buf[e] = next_info

                    if done or trunc:
                        if colour != "red":
                            ep_rew_accum[e] += (
                                next_info["rewards"].get("red", 0.0) / REWARD_SCALE
                            )
                        _credit_terminal_rewards(
                            env, per_env_data[e], actors[e], net,
                            next_info, colour, acting_was_live=True,
                            truncated=trunc,
                        )
                        _on_episode_end(
                            e, env, envs, obs_buf, info_buf,
                            actors, game_modes, pool, net,
                            ep_wins, ep_rewards, ep_rew_accum, ep_term_types,
                            ep_progress,
                            next_info, done, trunc,
                            global_step=global_step,
                            n_players_env=n_players_env,
                            n_players_hist=n_players_hist,
                        )

                else:
                    # Frozen/heuristic opponent acts; no transition is stored
                    # unless the game ends, in which case live players' latest
                    # transitions receive their terminal outcome.
                    action = actor.act(env)
                    next_obs, raw_r, done, trunc, next_info = env.step(action)
                    global_step += 1
                    if colour == "red":
                        ep_rew_accum[e] += raw_r / REWARD_SCALE
                    obs_buf[e] = next_obs
                    info_buf[e] = next_info
                    if done or trunc:
                        if colour != "red":
                            ep_rew_accum[e] += (
                                next_info["rewards"].get("red", 0.0) / REWARD_SCALE
                            )
                        _credit_terminal_rewards(
                            env, per_env_data[e], actors[e], net,
                            next_info, colour, acting_was_live=False,
                            truncated=trunc,
                        )
                        _on_episode_end(
                            e, env, envs, obs_buf, info_buf,
                            actors, game_modes, pool, net,
                            ep_wins, ep_rewards, ep_rew_accum, ep_term_types,
                            ep_progress,
                            next_info, done, trunc,
                            global_step=global_step,
                            n_players_env=n_players_env,
                            n_players_hist=n_players_hist,
                        )

        # ----------------------------------------------------------------
        # GAE COMPUTATION (per env)
        # ----------------------------------------------------------------
        net.eval()
        all_obs: list[np.ndarray] = []
        all_actions: list[int] = []
        all_logprobs: list[float] = []
        all_advantages: list[float] = []
        all_returns: list[float] = []
        all_masks: list[np.ndarray] = []

        for e, env in enumerate(envs):
            for colour, buffers in per_env_data[e].items():
                if len(buffers["obs"]) == 0:
                    continue

                rewards_e = np.array(buffers["rewards"], dtype=np.float32)
                values_e = np.array(buffers["values"], dtype=np.float32)
                dones_e = np.array(buffers["dones"], dtype=np.float32)

                # Bootstrap from this same player's perspective, not the next
                # seat to act.  This is the key difference from red-only PPO.
                if dones_e[-1]:
                    last_value = 0.0
                else:
                    last_value = _bootstrap_value_for_colour(net, env, colour, dev)

                adv, ret = _compute_gae(rewards_e, values_e, dones_e, last_value)

                # Per-episode advantage normalisation: equalises gradient magnitude
                # across episodes regardless of return scale. Without this, win-heavy
                # batches produce extreme negative advantages for cycle episodes (up to
                # −5.0 after batch normalisation, 35× larger than win-game advantages),
                # causing "avoid everything" gradients that cascade into more cycling.
                # In multi-player, also prevents N-player self-play episodes (which
                # contribute N colour trajectories) from dominating 2-player episodes.
                ep_start = 0
                for t in range(len(adv)):
                    if dones_e[t] == 1.0:
                        ep_adv = adv[ep_start : t + 1]
                        if len(ep_adv) > 1:
                            ep_std = float(ep_adv.std())
                            if ep_std > 1e-8:
                                adv[ep_start : t + 1] = (ep_adv - float(ep_adv.mean())) / ep_std
                        ep_start = t + 1
                if ep_start < len(adv):          # incomplete episode at rollout boundary
                    ep_adv = adv[ep_start:]
                    if len(ep_adv) > 1:
                        ep_std = float(ep_adv.std())
                        if ep_std > 1e-8:
                            adv[ep_start:] = (ep_adv - float(ep_adv.mean())) / ep_std

                all_obs.extend(buffers["obs"])
                all_actions.extend(buffers["actions"])
                all_logprobs.extend(buffers["logprobs"])
                all_advantages.extend(adv.tolist())
                all_returns.extend(ret.tolist())
                all_masks.extend(buffers["masks"])

        if len(all_obs) == 0:
            continue

        # Convert to tensors
        obs_batch = torch.from_numpy(np.array(all_obs)).float().to(dev)
        actions_batch = torch.tensor(all_actions, dtype=torch.long).to(dev)
        logprobs_batch = torch.tensor(all_logprobs, dtype=torch.float32).to(dev)
        adv_batch = torch.tensor(all_advantages, dtype=torch.float32).to(dev)
        ret_batch = torch.tensor(all_returns, dtype=torch.float32).to(dev)
        masks_batch = torch.from_numpy(np.array(all_masks)).to(dev)

        # Normalise advantages over the full batch; clip to ±5 to prevent
        # degenerate amplification when returns are compressed (std near zero).
        adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)
        adv_batch = adv_batch.clamp(-5.0, 5.0)

        # ----------------------------------------------------------------
        # PPO UPDATE
        # ----------------------------------------------------------------
        net.train()
        B = len(obs_batch)
        sum_p_loss = 0.0
        sum_v_loss = 0.0
        sum_ent = 0.0
        n_updates = 0
        n_ent_recovery = 0   # minibatches that fired entropy recovery mode this rollout
        kl_exceeded = False

        in_warmup = rollout_count <= WARMUP_ROLLOUTS
        if rollout_count == WARMUP_ROLLOUTS + 1:
            print(f"[Warmup] Value calibration complete at step {global_step:,} — policy updates enabled")

        # Ramp clip epsilon from CLIP_EPS_START → CLIP_EPS over the first POLICY_RAMPUP_ROLLOUTS
        # after warmup.  This prevents the concentrated Stage 1 policy from being shredded by
        # large-ε PPO updates before it has a chance to adapt gradually.
        rollouts_since_policy = max(0, rollout_count - WARMUP_ROLLOUTS)
        if rollouts_since_policy < POLICY_RAMPUP_ROLLOUTS:
            t = rollouts_since_policy / POLICY_RAMPUP_ROLLOUTS
            clip_eps = CLIP_EPS_START + (CLIP_EPS - CLIP_EPS_START) * t
        else:
            clip_eps = CLIP_EPS

        # Encoder freeze spans warmup AND rampup phases:
        #   warmup: value head calibrates; encoder frozen so its features don't drift under the policy head
        #   rampup: policy head adapts on fixed encoder representations; encoder still frozen so
        #     value-loss gradient can't shift encoder features and implicitly re-shape the policy
        #     distribution (this bypasses clip+KL and was the root cause of entropy explosion)
        # Policy head is unfrozen as soon as warmup ends so it can start learning new strategies.
        in_encoder_freeze = rollout_count <= WARMUP_ROLLOUTS + POLICY_RAMPUP_ROLLOUTS
        if rollout_count == WARMUP_ROLLOUTS + POLICY_RAMPUP_ROLLOUTS + 1:
            print(f"[Rampup] Encoder unfrozen at step {global_step:,} — full training begins")
        in_post_promo_warmup = post_promo_countdown > 0
        net.encoder.requires_grad_(not in_encoder_freeze and not in_post_promo_warmup)
        net.policy_head.requires_grad_(not in_warmup and not in_post_promo_warmup)

        for _ in range(N_EPOCHS):
            if kl_exceeded:
                break
            indices = torch.randperm(B, device=dev)
            for start in range(0, B, MINIBATCH):
                idx = indices[start : start + MINIBATCH]
                if len(idx) < 2:
                    continue

                _, new_lp, ent, new_val = net.get_action_and_value(
                    obs_batch[idx],
                    masks_batch[idx],
                    action=actions_batch[idx],
                )

                ratio = torch.exp(new_lp - logprobs_batch[idx])
                adv_mb = adv_batch[idx]

                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clamp returns to value head's range so Tanh can represent them.
                # Accumulated shaping rewards can push returns outside [-1, 1];
                # clamping keeps value targets valid without changing policy gradient.
                value_loss = nn.functional.mse_loss(
                    new_val, ret_batch[idx].clamp(-1.0, 1.0)
                )

                entropy_loss = ent.mean()

                # Entropy band: penalty above MAX_ENT ceiling + penalty below ENT_FLOOR.
                # Floor prevents the policy from becoming so concentrated (ent→0.077 at win=99%)
                # that any PPO update causes a catastrophic spike.
                ent_reg = (-ENT_COEF * entropy_loss
                           + ENT_PENALTY_COEF * torch.clamp(entropy_loss - MAX_ENT, min=0.0)
                           + ENT_FLOOR_COEF  * torch.clamp(ENT_FLOOR - entropy_loss, min=0.0))

                # Recovery mode: when entropy exceeds ceiling, skip the policy gradient
                # (which would amplify it further) and apply only entropy penalty + value.
                # Avoids the deadlock of break-before-update: penalty actively drives
                # entropy back below MAX_ENT rather than halting all learning.
                in_ent_recovery = (not in_warmup) and (entropy_loss.item() > MAX_ENT)
                if in_ent_recovery:
                    n_ent_recovery += 1

                # During warmup: only update value head so it can calibrate on real game dynamics.
                # Policy gradient is frozen — Stage 1 value head was trained on heuristic games and
                # mispredicts returns in Stage 1 self-play, producing large noisy advantages that
                # corrupt the policy in the very first rollout if left unchecked.
                if in_warmup or in_post_promo_warmup:
                    loss = VF_COEF * value_loss
                elif in_ent_recovery:
                    loss = (ENT_PENALTY_COEF * torch.clamp(entropy_loss - MAX_ENT, min=0.0)
                            + VF_COEF * value_loss)
                else:
                    loss = policy_loss + VF_COEF * value_loss + ent_reg

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                optimiser.step()

                sum_p_loss += policy_loss.item()
                sum_v_loss += value_loss.item()
                sum_ent += entropy_loss.item()
                n_updates += 1

                # KL early stopping: skip during entropy recovery (need all minibatches
                # to run their penalty updates to bring entropy back below MAX_ENT).
                if not in_ent_recovery:
                    with torch.no_grad():
                        approx_kl = (logprobs_batch[idx] - new_lp).mean().item()
                    if approx_kl > TARGET_KL:
                        kl_exceeded = True
                        break

        # Extra value-head calibration epochs: encoder and policy head frozen so these
        # updates only move the value head.  This keeps v_loss low even as the policy
        # changes each rollout, giving accurate advantages for the next update.
        net.encoder.requires_grad_(False)
        net.policy_head.requires_grad_(False)
        for _ in range(N_VALUE_EXTRA):
            v_idx = torch.randperm(B, device=dev)
            for start in range(0, B, MINIBATCH):
                idx = v_idx[start : start + MINIBATCH]
                if len(idx) < 2:
                    continue
                _, _, _, extra_val = net.get_action_and_value(
                    obs_batch[idx], masks_batch[idx], action=actions_batch[idx]
                )
                extra_v_loss = nn.functional.mse_loss(
                    extra_val, ret_batch[idx].clamp(-1.0, 1.0)
                )
                optimiser.zero_grad()
                (VF_COEF * extra_v_loss).backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                optimiser.step()
        net.encoder.requires_grad_(not in_encoder_freeze and not in_post_promo_warmup)
        net.policy_head.requires_grad_(not in_warmup and not in_post_promo_warmup)

        if in_post_promo_warmup:
            post_promo_countdown -= 1
            if post_promo_countdown == 0:
                print(f"[Promo-warmup] Value recalibration complete at step {global_step:,} — policy updates resumed")

        rollout_count += 1

        # ----------------------------------------------------------------
        # LOGGING
        # ----------------------------------------------------------------
        log_every = 1 if debug else 10
        if rollout_count % log_every == 0:
            recent_win = np.mean(ep_wins[-100:]) if ep_wins else 0.0
            recent_rew = np.mean(ep_rewards[-100:]) if ep_rewards else 0.0
            recent_prog = np.mean(ep_progress[-100:]) if ep_progress else 0.0
            elapsed = time.perf_counter() - t_start
            recent_types = ep_term_types[-100:]
            n_recent = max(len(recent_types), 1)
            pct_win   = recent_types.count("win")   / n_recent
            pct_loss  = recent_types.count("loss")  / n_recent
            pct_cycle = recent_types.count("cycle") / n_recent
            pct_limit = recent_types.count("limit") / n_recent
            p_loss_avg = sum_p_loss / max(n_updates, 1)
            v_loss_avg = sum_v_loss / max(n_updates, 1)
            ent_avg = sum_ent / max(n_updates, 1)
            if in_warmup:
                clip_tag = " | [warmup]"
            elif in_post_promo_warmup:
                clip_tag = f" | [promo-warmup={post_promo_countdown} left]"
            elif in_encoder_freeze:
                clip_tag = f" | clip={clip_eps:.3f}[enc-frz]"
            else:
                clip_tag = ""
            if n_ent_recovery > 0:
                clip_tag += f" | [rec={n_ent_recovery}mb]"
            hist_total = sum(n_players_hist.values())
            hist_str = (" | [" + " ".join(
                f"{n}p:{c}" for n, c in sorted(n_players_hist.items()) if c > 0
            ) + "]") if hist_total > 0 else ""
            print(
                f"step={global_step:>9,}"
                f" | win={recent_win:.3f}"
                f" | rew={recent_rew:.3f}"
                f" | prog={recent_prog:.3f}"
                f" | p_loss={p_loss_avg:.4f}"
                f" | v_loss={v_loss_avg:.4f}"
                f" | ent={ent_avg:.4f}"
                f" | term: W={pct_win:.0%} L={pct_loss:.0%} cyc={pct_cycle:.0%} lim={pct_limit:.0%}"
                f" | {elapsed / 3600:.2f}h"
                f"{clip_tag}"
                f"{hist_str}"
            )
            if debug and len(all_advantages) > 0:
                raw_adv = np.array(all_advantages)
                raw_ret = np.array(all_returns)
                val_arr = np.array([
                    v
                    for env_data in per_env_data
                    for buffers in env_data.values()
                    for v in buffers["values"]
                ])
                print(
                    f"  [debug] adv_raw  mean={raw_adv.mean():.3f} std={raw_adv.std():.3f}"
                    f" min={raw_adv.min():.3f} max={raw_adv.max():.3f}"
                )
                print(
                    f"  [debug] returns  mean={raw_ret.mean():.3f} std={raw_ret.std():.3f}"
                    f" min={raw_ret.min():.3f} max={raw_ret.max():.3f}"
                )
                print(
                    f"  [debug] val_pred mean={val_arr.mean():.3f} std={val_arr.std():.3f}"
                    f" min={val_arr.min():.3f} max={val_arr.max():.3f}"
                )

        # ----------------------------------------------------------------
        # CHECKPOINT + POOL PROMOTION (every CKPT_EVERY steps)
        # ----------------------------------------------------------------
        if global_step - last_ckpt_step >= CKPT_EVERY:
            ckpt_file = ckpt_save_dir / f"ppo_step_{global_step}.pt"
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "step": global_step,
                    "optimiser": optimiser.state_dict(),
                    "action_encoding": ACTION_ENCODING,
                },
                ckpt_file,
            )
            print(f"[Checkpoint] {ckpt_file.name}")
            last_ckpt_step = global_step

        if global_step - last_eval_step >= CKPT_EVERY:
            promote, eval_lines = _evaluate_promotion_suite(
                net, pool, heuristic, device=device
            )
            print(f"[Eval] step={global_step:,}  promote={promote}")
            for line in eval_lines:
                print(line)
            if promote:
                pool.add(net.state_dict(), step=global_step)
                post_promo_countdown = POST_PROMO_WARMUP
                print(f"[Promo-warmup] Starting {POST_PROMO_WARMUP}-rollout value recalibration")
            last_eval_step = global_step

    print(f"Training complete. Total steps: {global_step:,}")


# ---------------------------------------------------------------------------
# Episode-end helper
# ---------------------------------------------------------------------------

def _on_episode_end(
    e: int,
    env: ChineseCheckersEnv,
    envs: list,
    obs_buf: list,
    info_buf: list,
    actors: list,
    game_modes: list,
    pool: OpponentPool,
    net: ChineseCheckersNet,
    ep_wins: list,
    ep_rewards: list,
    ep_rew_accum: list,
    ep_term_types: list,
    ep_progress: list,
    info: dict,
    terminated: bool,
    truncated: bool,
    global_step: int = 0,
    n_players_env: list | None = None,
    n_players_hist: dict | None = None,
) -> None:
    """Reset env, sample new actors, record win/reward statistics."""
    rewards = info.get("rewards", {})
    red_r = rewards.get("red", 0.0)
    win = float(red_r >= 10.0)
    ep_wins.append(win)
    ep_rewards.append(ep_rew_accum[e])
    ep_rew_accum[e] = 0.0

    term_type = _term_type(info, truncated)
    ep_term_types.append(term_type)

    # Progress = distance component of terminal reward (strips win bonus for wins)
    if term_type == "win":
        ep_progress.append(red_r - 10.0)
    else:
        ep_progress.append(red_r)

    # Record player count of completed game, then reset with a freshly sampled count.
    if n_players_env is not None:
        if n_players_hist is not None:
            n_players_hist[n_players_env[e]] = n_players_hist.get(n_players_env[e], 0) + 1
        n_p = random.choices(list(PLAYER_MIX.keys()), weights=list(PLAYER_MIX.values()))[0]
        n_players_env[e] = n_p
        obs_buf[e], info_buf[e] = envs[e].reset(n_players=n_p)
        actors[e], game_modes[e] = _sample_game_actors(pool, net, n_p)
    else:
        obs_buf[e], info_buf[e] = envs[e].reset()
        actors[e], game_modes[e] = _sample_game_actors(pool, net, envs[e].n_players)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description="Stage 2 PPO self-play training")
    p.add_argument("--stage1-ckpt", type=str, default="checkpoints/stage1_supervised.pt")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from a PPO checkpoint, e.g. checkpoints/ppo_step_100352.pt")
    p.add_argument("--debug", action="store_true",
                   help="Log every rollout with value/return/advantage diagnostics")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    train(
        stage1_ckpt=args.stage1_ckpt,
        ckpt_dir=args.ckpt_dir,
        max_steps=args.max_steps,
        device=args.device,
        resume=args.resume,
        debug=args.debug,
    )
