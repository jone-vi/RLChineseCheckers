"""
Opponent pool for Stage 2 PPO self-play training.

Maintains up to pool_size frozen checkpoint snapshots plus a heuristic agent.
All agents implement the .act(env) -> int interface used by HeuristicAgent and
ChineseCheckersNet.act().

Usage:
    pool = OpponentPool(heuristic_agent, pool_size=5)
    pool.add(net.state_dict(), step=100_000)
    opponent = pool.sample_opponent(current_net, mix_ratio=0.3)
    action = opponent.act(env)
"""

import pathlib
import random
import sys

import torch

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.models.network import ChineseCheckersNet

ACTION_ENCODING = "canonical_destination_v2"


class _FrozenAgent:
    """
    Wraps a frozen (eval-mode) copy of ChineseCheckersNet on CPU.
    Implements the same .act(env) -> int contract as HeuristicAgent.
    """

    def __init__(self, state_dict: dict, step: int):
        self.net = ChineseCheckersNet()
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self._step = step

    def act(self, env) -> int:
        return self.net.act(env, temperature=0.3)

    def __repr__(self) -> str:
        return f"FrozenAgent(step={self._step:,})"


class OpponentPool:
    """
    Manages frozen checkpoint snapshots for opponent-pool mixing during PPO.

    Pool invariants:
    - Up to pool_size frozen nets kept (oldest evicted on overflow, FIFO)
    - Heuristic agent is always a candidate in the mix branch
    - sample_opponent() returns an agent with .act(env) -> int

    Args:
        heuristic_agent:  HeuristicAgent instance — fallback when pool is empty,
                          and always included as a candidate in the mix branch.
        pool_size:        Maximum number of frozen net checkpoints to retain.
        ckpt_dir:         Directory for saving pool checkpoint files.
    """

    def __init__(
        self,
        heuristic_agent,
        pool_size: int = 5,
        ckpt_dir: str = "checkpoints",
    ):
        self._heuristic = heuristic_agent
        self._pool_size = pool_size
        self._ckpt_dir = _ROOT / ckpt_dir
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._agents: list[_FrozenAgent] = []   # oldest at index 0
        self._rng = random.Random()

    # ------------------------------------------------------------------
    # Pool management
    # ------------------------------------------------------------------

    def add(self, state_dict: dict, step: int) -> None:
        """
        Save a checkpoint to disk and add a frozen agent to the pool.
        Evicts the oldest agent if the pool is at capacity.
        """
        ckpt_path = self._ckpt_dir / f"pool_{step}.pt"
        torch.save(
            {"state_dict": state_dict, "step": step, "action_encoding": ACTION_ENCODING},
            ckpt_path,
        )

        agent = _FrozenAgent(state_dict, step=step)
        self._agents.append(agent)

        if len(self._agents) > self._pool_size:
            evicted = self._agents.pop(0)
            print(f"[Pool] Evicted {evicted}")

        print(f"[Pool] Added step={step:,}  size={len(self._agents)}/{self._pool_size}")

    @property
    def heuristic(self):
        """Return the stable heuristic opponent."""
        return self._heuristic

    def latest(self):
        """Return the newest frozen network opponent, or None if the pool is empty."""
        return self._agents[-1] if self._agents else None

    def sample_frozen(self):
        """Sample only from frozen network opponents, excluding the heuristic."""
        if not self._agents:
            return None
        return self._rng.choice(self._agents)

    # ------------------------------------------------------------------
    # Opponent sampling
    # ------------------------------------------------------------------

    def sample_opponent(self, current_net, mix_ratio: float = 0.3):
        """
        Sample an opponent agent.

        With probability (1 - mix_ratio): return current_net (self-play).
        With probability mix_ratio: sample uniformly from frozen pool + heuristic.

        The heuristic is always included in the mix candidates, so this is safe
        even when the pool of frozen nets is empty.

        Args:
            current_net:  The live ChineseCheckersNet being trained.
            mix_ratio:    Fraction of games to play against pool/heuristic (default 0.3).

        Returns:
            Agent with .act(env) -> int interface.
        """
        if self._rng.random() >= mix_ratio:
            return current_net   # self-play (70%)

        candidates = list(self._agents) + [self._heuristic]
        return self._rng.choice(candidates)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._agents)

    def __repr__(self) -> str:
        return f"OpponentPool(size={len(self._agents)}/{self._pool_size})"
