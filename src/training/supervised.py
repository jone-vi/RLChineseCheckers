"""
Stage 1 supervised pretraining.

Trains the MLP actor-critic on (observation, action, outcome) tuples generated
by heuristic self-play (see data_gen.py).

Loss: CrossEntropyLoss(policy logits, actions)
    + 0.5 * MSELoss(value, outcomes)
    + 1e-4 L2 weight decay (via Adam weight_decay)

Saves the best checkpoint (lowest total loss) to checkpoints/stage1_supervised.pt.

Usage:
    python -m src.training.supervised
    python -m src.training.supervised --data-path data/stage1_debug.h5 --epochs 3
"""

import argparse
import pathlib
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.models.network import ChineseCheckersNet


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Stage1Dataset(Dataset):
    """
    Loads the full HDF5 dataset into memory.

    HDF5 schema (written by data_gen.py):
        obs      float32 [N, 1089]  — board observation (acting player's POV)
        actions  int16   [N]        — encoded action (pin_id * 121 + canonical_dest_idx)
        outcomes float32 [N]        — game outcome for acting player in [-1, 1]

    IMPORTANT: actions are stored as int16 but CrossEntropyLoss requires int64.
    The cast happens here at load time so __getitem__ is fast.
    """

    def __init__(self, h5_path: str):
        with h5py.File(h5_path, "r") as f:
            self.obs = torch.from_numpy(f["obs"][:])                              # float32 [N,1089]
            self.actions = torch.from_numpy(f["actions"][:].astype(np.int64))    # int16 → int64
            self.outcomes = torch.from_numpy(f["outcomes"][:])                    # float32 [N]

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx], self.outcomes[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_path: str = "data/stage1.h5",
    ckpt_dir: str = "checkpoints",
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = None,
) -> str:
    """
    Run Stage 1 supervised pretraining. Returns path to saved checkpoint.
    """
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

    # -- Data --
    h5_path = _ROOT / data_path
    if not h5_path.exists():
        raise FileNotFoundError(
            f"HDF5 not found: {h5_path}\nRun data_gen.py first."
        )

    dataset = Stage1Dataset(str(h5_path))
    with h5py.File(h5_path, "r") as f:
        action_encoding = f.attrs.get("action_encoding", "legacy_absolute_destination")
    if action_encoding != "canonical_destination_v2":
        raise ValueError(
            f"Dataset action_encoding={action_encoding!r}; "
            "regenerate data with the current env for canonical-action training."
        )
    # num_workers=0 avoids multiprocessing issues on macOS/Windows
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(dev.type == "cuda"),
    )
    print(f"Dataset: {len(dataset):,} examples, {len(loader)} batches/epoch")

    # -- Model + optimiser --
    net = ChineseCheckersNet().to(dev)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_epochs)

    policy_crit = nn.CrossEntropyLoss()
    value_crit = nn.MSELoss()

    # -- Checkpoint path --
    ckpt_path = _ROOT / ckpt_dir / "stage1_supervised.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    best_total_loss = float("inf")

    # -- Training loop --
    for epoch in range(1, n_epochs + 1):
        net.train()
        sum_policy = 0.0
        sum_value = 0.0
        sum_total = 0.0
        t0 = time.perf_counter()

        for obs_b, actions_b, outcomes_b in loader:
            obs_b = obs_b.to(dev)           # [B, 1089] float32
            actions_b = actions_b.to(dev)   # [B]       int64
            outcomes_b = outcomes_b.to(dev) # [B]       float32

            # Pass None mask — all actions in dataset were legal at collection time
            logits, value = net(obs_b, None)   # [B,1210], [B,1]
            value = value.squeeze(-1)           # [B]

            p_loss = policy_crit(logits, actions_b)
            v_loss = value_crit(value, outcomes_b)
            total = p_loss + 0.5 * v_loss

            optimiser.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimiser.step()

            sum_policy += p_loss.item()
            sum_value += v_loss.item()
            sum_total += total.item()

        scheduler.step()

        n = len(loader)
        avg_p = sum_policy / n
        avg_v = sum_value / n
        avg_t = sum_total / n
        elapsed = time.perf_counter() - t0

        print(
            f"Epoch {epoch:3d}/{n_epochs}"
            f"  policy={avg_p:.4f}"
            f"  value={avg_v:.4f}"
            f"  total={avg_t:.4f}"
            f"  {elapsed:.1f}s"
        )

        if avg_t < best_total_loss:
            best_total_loss = avg_t
            torch.save(
                {
                    "state_dict": net.state_dict(),
                    "epoch": epoch,
                    "policy_loss": avg_p,
                    "value_loss": avg_v,
                    "total_loss": avg_t,
                    "n_examples": len(dataset),
                    "hyperparams": {
                        "lr": lr,
                        "batch_size": batch_size,
                        "weight_decay": weight_decay,
                        "n_epochs": n_epochs,
                    },
                    "action_encoding": "canonical_destination_v2",
                },
                ckpt_path,
            )

    print(f"\nSaved: {ckpt_path}  (best loss: {best_total_loss:.4f})")
    return str(ckpt_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description="Stage 1 supervised pretraining")
    p.add_argument("--data-path", type=str, default="data/stage1.h5")
    p.add_argument("--ckpt-dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    train(
        data_path=args.data_path,
        ckpt_dir=args.ckpt_dir,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )
