"""
MLP actor-critic with shared encoder for Chinese Checkers.

Architecture:
    Shared encoder:  1089 → 512 (LayerNorm + ReLU) → 256 (LayerNorm + ReLU)
    Policy head:      256 → 256 (ReLU) → 1210  (logits, masked before softmax)
    Value head:       256 → 128 (ReLU) →    1  (Tanh, output in [-1, 1])

Total parameters: ~900 K
"""

import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

_ROOT = pathlib.Path(__file__).resolve().parents[2]


class ChineseCheckersNet(nn.Module):
    """
    Shared-encoder actor-critic for Chinese Checkers.

    The policy head outputs logits over all 1210 (pin_id, dest) pairs.
    Illegal actions are masked to -1e9 before sampling/argmax so they are
    never selected.  The value head estimates expected game outcome in [-1, 1]
    from the current player's perspective.
    """

    OBS_SIZE = 1089
    ACT_SIZE = 1210

    def __init__(self):
        super().__init__()

        # --- Shared encoder ---
        self.encoder = nn.Sequential(
            nn.Linear(self.OBS_SIZE, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
        )

        # --- Policy head ---
        self.policy_head = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.ACT_SIZE),
        )

        # --- Value head ---
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        """Orthogonal init for linear layers, zero bias."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Scale down the policy output layer for stable initial distribution
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)

    # ------------------------------------------------------------------
    # Core forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        obs:         torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            obs:         float32 [B, 1089]
            action_mask: int8 or float32 [B, 1210], binary (1=legal, 0=illegal).
                         Pass None to skip masking (e.g. during loss computation
                         when you have already stored masked logits).

        Returns:
            logits: float32 [B, 1210]  — illegal actions set to -1e9 if mask given
            value:  float32 [B, 1]
        """
        enc    = self.encoder(obs)
        logits = self.policy_head(enc)
        value  = self.value_head(enc)

        if action_mask is not None:
            # Cast mask to bool-compatible float and suppress illegal actions
            mask_f = action_mask.to(dtype=torch.float32)
            logits = logits.masked_fill(mask_f == 0, -1e9)

        return logits, value

    # ------------------------------------------------------------------
    # PPO training interface
    # ------------------------------------------------------------------

    def get_action_and_value(
        self,
        obs:         torch.Tensor,
        action_mask: torch.Tensor,
        action:      torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Standard CleanRL-style PPO interface.

        If action is None: sample a new action from the (masked) policy.
        If action is provided: evaluate log_prob for that existing action.

        Returns:
            action:   int64  [B]
            log_prob: float  [B]
            entropy:  float  [B]
            value:    float  [B]
        """
        logits, value = self.forward(obs, action_mask)
        dist          = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    # ------------------------------------------------------------------
    # Inference interface
    # ------------------------------------------------------------------

    def select_action(
        self,
        obs:         np.ndarray,
        action_mask: np.ndarray,
        temperature: float = 1.0,
        device:      str   = "cpu",
    ) -> int:
        """
        Single-step action selection for deployment / evaluation.

        Args:
            obs:         float32 array [1089]
            action_mask: int8 array [1210]
            temperature: 0.0 = argmax (greedy), >0 = sample

        Returns:
            Encoded action integer (0–1209).
        """
        obs_t  = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        mask_t = torch.from_numpy(action_mask).float().unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = self.forward(obs_t, mask_t)

        if temperature == 0.0:
            return int(logits.argmax(dim=-1).item())

        scaled = logits / temperature
        return int(torch.distributions.Categorical(logits=scaled).sample().item())

    def act(self, env, temperature: float = 0.3) -> int:
        """
        Shared act(env) -> int contract for opponent-pool compatibility.
        Uses the same interface as HeuristicAgent.act().
        """
        obs  = env._build_observation()
        mask = env._build_action_mask()
        return self.select_action(obs, mask, temperature=temperature)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    net = ChineseCheckersNet()
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    assert 800_000 <= n_params <= 1_200_000, f"Unexpected parameter count: {n_params:,}"

    B = 4
    obs  = torch.zeros(B, ChineseCheckersNet.OBS_SIZE)
    mask = torch.ones(B,  ChineseCheckersNet.ACT_SIZE, dtype=torch.int8)
    # Mask out the last 100 actions
    mask[:, -100:] = 0

    logits, value = net(obs, mask)
    assert logits.shape == (B, 1210), f"Bad logits shape: {logits.shape}"
    assert value.shape  == (B, 1),    f"Bad value shape:  {value.shape}"
    assert (logits[:, -100:] == -1e9).all(), "Mask not applied correctly"
    assert value.abs().max() <= 1.0,         "Value head outside [-1, 1]"

    # PPO interface
    action, log_prob, entropy, val = net.get_action_and_value(obs, mask)
    assert action.shape   == (B,)
    assert log_prob.shape == (B,)
    assert entropy.shape  == (B,)
    assert val.shape      == (B,)
    assert (action < 1210).all()
    assert (action >= 1110).logical_not().all() or True  # masked actions not sampled
    # Verify no masked action was sampled
    for i in range(B):
        assert action[i].item() < 1110, \
            f"Sampled a masked action {action[i]} for batch item {i}"

    # Supervised loss shapes (Stage 1)
    target_actions  = torch.randint(0, 1110, (B,))
    target_outcomes = torch.rand(B) * 2 - 1
    policy_loss = nn.CrossEntropyLoss()(logits, target_actions)
    value_loss  = nn.MSELoss()(val, target_outcomes)
    (policy_loss + value_loss).backward()
    print(f"policy_loss={policy_loss.item():.4f}  value_loss={value_loss.item():.4f}")

    # select_action (numpy interface)
    single_obs  = np.zeros(1089, dtype=np.float32)
    single_mask = np.ones(1210,  dtype=np.int8)
    a = net.select_action(single_obs, single_mask, temperature=1.0)
    assert 0 <= a < 1210, f"Action {a} out of range"

    print(f"Parameters: {n_params:,}  PASS")
