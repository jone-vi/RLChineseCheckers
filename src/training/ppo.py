"""
Stage 2 PPO self-play training.

Loads Stage 1 weights and refines via PPO with:
- 8 parallel 2-player environments
- 128-step rollouts per env → 1024+ transitions per update
- GAE advantage estimation (γ=0.99, λ=0.95)
- Clipped PPO objective (ε=0.2), entropy bonus, value loss
- Opponent pool: 70% self-play / 30% vs frozen snapshots + heuristic
- Pool promotion when win rate >55% over 50 eval games (every 100K steps)
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
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.env.chinese_checkers_env import ChineseCheckersEnv
from src.models.network import ChineseCheckersNet
from src.training.heuristic import HeuristicAgent
from src.training.opponent_pool import OpponentPool

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
N_ENVS = 8
ROLLOUT_STEPS = 128          # steps per env per update → ~1024 transitions
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF = 0.0001            # was 0.003 — high entropy coef spread the policy faster than PPO could improve it
ENT_PENALTY_COEF = 0.05      # penalty for entropy exceeding MAX_ENT (raised from 0.01 — was too weak)
MAX_ENT = 2.0                # entropy ceiling — above this the policy is too diffuse to win reliably
VF_COEF = 0.5
LR = 1e-4                    # was 3e-4 — lower LR preserves Stage 1 knowledge during fine-tuning
N_EPOCHS = 1                 # 4 epochs × 16 minibatches = 64 steps per rollout was too many; value head
                             # can't track the policy when it changes this fast from a concentrated start
MINIBATCH = 64
TARGET_KL = 0.01             # stop epoch early if policy changes too much
POOL_MIX_RATIO = 1.0         # all games vs pool (Stage 1 + heuristic + promoted checkpoints); no self-play
MAX_STEPS = 5_000_000
REWARD_SCALE = 10.0          # divide raw rewards; terminal win→+1, loss→-1
CKPT_EVERY = 100_000
EVAL_GAMES = 50
PROMOTE_RATE = 0.55


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
    net.eval()
    wins = 0
    env = ChineseCheckersEnv(n_players=2)

    for _ in range(n_games):
        obs, info = env.reset()
        for _ in range(800):
            if env._turn_idx == 0:
                action = net.act(env, temperature=0.3)
            else:
                action = opponent.act(env)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if info["rewards"]["red"] > info["rewards"]["blue"]:
                    wins += 1
                break

    net.train()
    return wins / n_games


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    stage1_ckpt: str = "checkpoints/stage1_supervised.pt",
    ckpt_dir: str = "checkpoints",
    max_steps: int = MAX_STEPS,
    device: str = None,
    resume: str = None,
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
    obs_buf = [None] * N_ENVS      # current observation after last step
    info_buf = [None] * N_ENVS     # current info after last step
    opponents = [None] * N_ENVS    # opponent agent for each env
    is_selfplay = [True] * N_ENVS  # True when opponent IS the learning net

    for e in range(N_ENVS):
        obs_buf[e], info_buf[e] = envs[e].reset()
        opp = pool.sample_opponent(net, mix_ratio=POOL_MIX_RATIO)
        opponents[e] = opp
        is_selfplay[e] = (opp is net)

    # -- Per-env rollout storage (lists of numpy items, converted to arrays for GAE) --
    per_env_obs = [[] for _ in range(N_ENVS)]
    per_env_actions = [[] for _ in range(N_ENVS)]
    per_env_logprobs = [[] for _ in range(N_ENVS)]
    per_env_values = [[] for _ in range(N_ENVS)]
    per_env_rewards = [[] for _ in range(N_ENVS)]
    per_env_dones = [[] for _ in range(N_ENVS)]
    per_env_masks = [[] for _ in range(N_ENVS)]

    # -- Tracking --
    ckpt_save_dir = _ROOT / ckpt_dir
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)
    rollout_count = 0
    last_ckpt_step = resume_step
    last_eval_step = resume_step
    ep_wins: list[float] = []     # rolling window: 1=win, 0=loss for learning net
    ep_rewards: list[float] = []  # rolling window: per-episode total reward
    ep_term_types: list[str] = [] # rolling window: "win", "loss", "cycle", "limit"

    # Per-env episode reward accumulator
    ep_rew_accum = [0.0] * N_ENVS

    t_start = time.perf_counter()
    print(f"Starting PPO training. Max steps: {max_steps:,}")

    while global_step < max_steps:

        # ----------------------------------------------------------------
        # ROLLOUT PHASE
        # ----------------------------------------------------------------
        for e in range(N_ENVS):
            per_env_obs[e].clear()
            per_env_actions[e].clear()
            per_env_logprobs[e].clear()
            per_env_values[e].clear()
            per_env_rewards[e].clear()
            per_env_dones[e].clear()
            per_env_masks[e].clear()

        net.eval()

        for _step in range(ROLLOUT_STEPS):
            for e, env in enumerate(envs):
                obs = obs_buf[e]
                info = info_buf[e]
                mask = info["action_mask"]

                colour = env._active_colours[env._turn_idx]
                is_learning = (env._turn_idx == 0)

                if is_learning:
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
                    ep_rew_accum[e] += reward

                    # Store transition
                    per_env_obs[e].append(obs.copy())
                    per_env_actions[e].append(action)
                    per_env_logprobs[e].append(log_prob_val)
                    per_env_values[e].append(value_val)
                    per_env_rewards[e].append(reward)
                    per_env_dones[e].append(float(done or trunc))
                    per_env_masks[e].append(mask.copy())

                    obs_buf[e] = next_obs
                    info_buf[e] = next_info

                    if done or trunc:
                        _on_episode_end(
                            e, env, envs, obs_buf, info_buf,
                            opponents, is_selfplay, pool, net,
                            ep_wins, ep_rewards, ep_rew_accum, ep_term_types,
                            next_info, done, trunc,
                            global_step=global_step,
                        )

                else:
                    # Opponent acts — no buffer storage
                    action = opponents[e].act(env)
                    next_obs, raw_r, done, trunc, next_info = env.step(action)
                    global_step += 1
                    obs_buf[e] = next_obs
                    info_buf[e] = next_info
                    if done or trunc:
                        # Credit P0's last stored transition with the terminal outcome.
                        # P0 doesn't act again, so this is the only way it receives
                        # its loss/draw reward from the opponent ending the game.
                        if per_env_rewards[e]:
                            red_final = next_info["rewards"].get("red", 0.0) / REWARD_SCALE
                            per_env_rewards[e][-1] += red_final
                            per_env_dones[e][-1] = 1.0
                        _on_episode_end(
                            e, env, envs, obs_buf, info_buf,
                            opponents, is_selfplay, pool, net,
                            ep_wins, ep_rewards, ep_rew_accum, ep_term_types,
                            next_info, done, trunc,
                            global_step=global_step,
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

        for e in range(N_ENVS):
            if len(per_env_obs[e]) == 0:
                continue

            rewards_e = np.array(per_env_rewards[e], dtype=np.float32)
            values_e = np.array(per_env_values[e], dtype=np.float32)
            dones_e = np.array(per_env_dones[e], dtype=np.float32)

            # Bootstrap value for the state after the last collected step
            if per_env_dones[e][-1]:
                last_value = 0.0
            else:
                with torch.no_grad():
                    last_obs_t = torch.from_numpy(obs_buf[e]).float().unsqueeze(0).to(dev)
                    last_mask_t = torch.from_numpy(
                        info_buf[e]["action_mask"]
                    ).float().unsqueeze(0).to(dev)
                    _, last_val = net(last_obs_t, last_mask_t)
                    last_value = float(last_val.item())

            adv, ret = _compute_gae(rewards_e, values_e, dones_e, last_value)

            all_obs.extend(per_env_obs[e])
            all_actions.extend(per_env_actions[e])
            all_logprobs.extend(per_env_logprobs[e])
            all_advantages.extend(adv.tolist())
            all_returns.extend(ret.tolist())
            all_masks.extend(per_env_masks[e])

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
        kl_exceeded = False

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
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clamp returns to value head's range so Tanh can represent them.
                # Accumulated shaping rewards can push returns outside [-1, 1];
                # clamping keeps value targets valid without changing policy gradient.
                value_loss = nn.functional.mse_loss(
                    new_val, ret_batch[idx].clamp(-1.0, 1.0)
                )

                entropy_loss = ent.mean()

                # Entropy regularization: small bonus below MAX_ENT, strong penalty above.
                # Keeps the policy diverse enough to explore but not so random it loses signal.
                ent_reg = (-ENT_COEF * entropy_loss
                           + ENT_PENALTY_COEF * torch.clamp(entropy_loss - MAX_ENT, min=0.0))
                loss = policy_loss + VF_COEF * value_loss + ent_reg

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
                optimiser.step()

                sum_p_loss += policy_loss.item()
                sum_v_loss += value_loss.item()
                sum_ent += entropy_loss.item()
                n_updates += 1

                # KL early stopping: approximate KL = mean(log π_old - log π_new)
                with torch.no_grad():
                    approx_kl = (logprobs_batch[idx] - new_lp).mean().item()
                if approx_kl > TARGET_KL:
                    kl_exceeded = True
                    break

        rollout_count += 1

        # ----------------------------------------------------------------
        # LOGGING
        # ----------------------------------------------------------------
        log_every = 1 if debug else 10
        if rollout_count % log_every == 0:
            recent_win = np.mean(ep_wins[-100:]) if ep_wins else 0.0
            recent_rew = np.mean(ep_rewards[-100:]) if ep_rewards else 0.0
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
            print(
                f"step={global_step:>9,}"
                f" | win={recent_win:.3f}"
                f" | rew={recent_rew:.3f}"
                f" | p_loss={p_loss_avg:.4f}"
                f" | v_loss={v_loss_avg:.4f}"
                f" | ent={ent_avg:.4f}"
                f" | term: W={pct_win:.0%} L={pct_loss:.0%} cyc={pct_cycle:.0%} lim={pct_limit:.0%}"
                f" | {elapsed / 3600:.2f}h"
            )
            if debug and len(all_advantages) > 0:
                raw_adv = np.array(all_advantages)
                raw_ret = np.array(all_returns)
                val_arr = np.array([v for vs in per_env_values for v in vs])
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
                },
                ckpt_file,
            )
            print(f"[Checkpoint] {ckpt_file.name}")
            last_ckpt_step = global_step

        if global_step - last_eval_step >= CKPT_EVERY:
            eval_opp = pool._agents[-1] if len(pool) > 0 else heuristic
            win_rate = evaluate(net, eval_opp, n_games=EVAL_GAMES, device=device)
            print(f"[Eval] step={global_step:,}  win_rate={win_rate:.3f}")
            if win_rate >= PROMOTE_RATE:
                pool.add(net.state_dict(), step=global_step)
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
    opponents: list,
    is_selfplay: list,
    pool: OpponentPool,
    net: ChineseCheckersNet,
    ep_wins: list,
    ep_rewards: list,
    ep_rew_accum: list,
    ep_term_types: list,
    info: dict,
    terminated: bool,
    truncated: bool,
    global_step: int = 0,
) -> None:
    """Reset env, sample new opponent, record win/reward statistics."""
    rewards = info.get("rewards", {})
    red_r = rewards.get("red", 0.0)
    blue_r = rewards.get("blue", 0.0)
    win = float(red_r > blue_r)
    ep_wins.append(win)
    ep_rewards.append(ep_rew_accum[e])
    ep_rew_accum[e] = 0.0

    if truncated:
        term_type = "limit"
    elif red_r >= 10.0:
        term_type = "win"
    elif blue_r >= 10.0:
        term_type = "loss"
    else:
        term_type = "cycle"
    ep_term_types.append(term_type)

    obs_buf[e], info_buf[e] = envs[e].reset()
    opp = pool.sample_opponent(net, mix_ratio=POOL_MIX_RATIO)
    opponents[e] = opp
    is_selfplay[e] = (opp is net)


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
