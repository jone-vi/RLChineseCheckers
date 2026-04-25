import contextlib
import importlib.util
import io
import pathlib
import sys
from typing import Dict, List, Optional, Set, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Import HexBoard and Pin from the space-named directory.
# checkers_pins.py does `from checkers_board import ...` at module level, so
# checkers_board MUST be registered in sys.modules before checkers_pins loads.
# ---------------------------------------------------------------------------
_GAME_DIR = pathlib.Path(__file__).resolve().parents[2] / "multi system single machine minimal"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _GAME_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod        # register before exec so cross-imports resolve
    spec.loader.exec_module(mod)
    return mod


_board_mod = _load("checkers_board", "checkers_board.py")   # MUST be loaded first
_pins_mod  = _load("checkers_pins",  "checkers_pins.py")

HexBoard = _board_mod.HexBoard
Pin      = _pins_mod.Pin

# ---------------------------------------------------------------------------
# Board constants — defined locally, not imported from game.py
# ---------------------------------------------------------------------------
COLOUR_ORDER = ['red', 'lawn green', 'yellow', 'blue', 'gray0', 'purple']
COLOUR_OPPOSITES = {
    'red':        'blue',
    'lawn green': 'gray0',
    'yellow':     'purple',
    'blue':       'red',
    'gray0':      'lawn green',
    'purple':     'yellow',
}


class ChineseCheckersEnv(gymnasium.Env):
    """
    Gymnasium environment for Chinese Checkers.

    Observation: 9-channel × 121-cell board, always from the current player's
    perspective (canonical channel rotation), flattened to shape (1089,).

    Action: Discrete(1210) = pin_id * 121 + destination_index.
    A binary action mask is returned in info["action_mask"].

    Multi-player: a single env instance manages all N players on one board.
    step() applies the current player's move and advances the turn.
    The returned obs is from the NEW current player's perspective.
    info["rewards"] carries per-player rewards so the training loop can
    attribute credit correctly.
    """

    metadata = {"render_modes": ["ansi"]}

    N_CELLS  = 121
    N_PINS   = 10
    N_CH     = 9
    OBS_SIZE = N_CH * N_CELLS    # 1089
    ACT_SIZE = N_PINS * N_CELLS  # 1210

    def __init__(self, n_players: int = 2, render_mode: Optional[str] = None):
        super().__init__()
        assert 2 <= n_players <= 6, "n_players must be between 2 and 6"

        self.n_players   = n_players
        self.render_mode = render_mode

        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.OBS_SIZE,), dtype=np.float32)
        self.action_space      = spaces.Discrete(self.ACT_SIZE)

        # Stable geometry — computed once from a throwaway board
        self._active_colours: List[str] = self._select_colours()
        self._goal_zone_indices: Dict[str, Set[int]] = {}
        self._goal_targets: Dict[str, Tuple[float, float]] = {}
        self._precompute_geometry()

        # Episode state — all re-initialised in reset()
        self._board:    Optional[HexBoard]          = None
        self._pins:     Dict[str, List[Pin]]        = {}
        self._turn_idx: int                         = 0
        self._move_counts:       Dict[str, int]     = {}
        self._still_playing:     List[str]          = []
        self._d_max:             Dict[str, float]   = {}
        self._prev_dist_sq:      Dict[str, float]   = {}
        self._pins_entered_goal: Dict[str, Set[int]]= {}
        self._state_hash_counts: Dict[int, int]     = {}
        self._player_pos_counts: Dict[str, Dict[int, int]] = {}

    # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._init_board()
        self._turn_idx    = 0
        self._move_counts = {c: 0 for c in self._active_colours}
        self._still_playing      = list(self._active_colours)
        self._pins_entered_goal  = {c: set() for c in self._active_colours}
        self._state_hash_counts  = {}
        self._player_pos_counts  = {c: {} for c in self._active_colours}

        # Compute D_max from starting positions to deepest goal cell
        for colour in self._active_colours:
            cq, cr = self._goal_targets[colour]
            d = sum(self._axial_dist_sq(p.axialindex, cq, cr) for p in self._pins[colour])
            self._d_max[colour]       = max(d, 1.0)
            self._prev_dist_sq[colour] = d

        h = self._board_state_hash()
        self._state_hash_counts[h] = 1

        obs  = self._build_observation()
        mask = self._build_action_mask()
        info = {
            "action_mask":    mask,
            "current_player": self._active_colours[self._turn_idx],
            "rewards":        {c: 0.0 for c in self._active_colours},
        }
        return obs, info

    def step(self, action: int):
        acting   = self._active_colours[self._turn_idx]
        pin_id   = action // self.N_CELLS
        dest_idx = action %  self.N_CELLS

        d_before = self._prev_dist_sq[acting]

        # Apply move — suppress the print() inside placePin
        with contextlib.redirect_stdout(io.StringIO()):
            ok = self._pins[acting][pin_id].placePin(dest_idx)
        assert ok, f"Illegal action {action} passed to step(); check action mask"

        self._move_counts[acting] += 1

        d_after = self._total_dist_sq(acting)
        self._prev_dist_sq[acting] = d_after

        h = self._board_state_hash()
        self._state_hash_counts[h] = self._state_hash_counts.get(h, 0) + 1

        # ---- Determine outcome -----------------------------------------
        rewards    = {c: 0.0 for c in self._active_colours}
        terminated = False
        truncated  = False

        if self._check_win(acting):
            rewards[acting] = 10.0
            # Non-winners keep default 0.0 — no loss penalty
            terminated = True

        elif self._state_hash_counts[h] >= 20:
            # Safety valve for extremely cyclic games; no explicit penalty
            terminated = True

        elif all(self._move_counts[c] >= 200 for c in self._active_colours):
            truncated = True

        else:
            p_hash = self._player_pos_hash(acting)
            self._player_pos_counts[acting][p_hash] = (
                self._player_pos_counts[acting].get(p_hash, 0) + 1
            )
            rewards[acting] += self._shaping_reward(
                acting, d_before, d_after, self._player_pos_counts[acting][p_hash]
            )
            self._advance_turn()
            # Skip over any players with no legal moves (e.g. all pins in goal
            # zone and no deeper empty cells).  Loop up to n_players-1 times so
            # we don't spin forever; if everyone is stuck, terminate as a draw.
            for _ in range(self.n_players - 1):
                if self._build_action_mask().sum() > 0:
                    break
                self._advance_turn()
            else:
                terminated = True

        if terminated or truncated:
            for c in self._active_colours:
                final_dist = self._prev_dist_sq[c]
                initial_dist = self._d_max[c]
                progress = (initial_dist - final_dist) / max(initial_dist, 1.0)
                rewards[c] += 1.0 * progress

        obs  = self._build_observation()
        mask = self._build_action_mask()

        info = {
            "action_mask":     mask,
            "current_player":  self._active_colours[self._turn_idx],
            "rewards":         rewards,
            "no_legal_moves":  False,
        }
        return obs, float(rewards[acting]), terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            all_pins = [p for ps in self._pins.values() for p in ps]
            self._board.print_ascii(pins=all_pins)

    def close(self):
        pass

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _select_colours(self) -> List[str]:
        """
        Pick n_players colours matching the server's assignment order:
        player 1→red, 2→blue, 3→lawn green, 4→gray0, 5→yellow, 6→purple.
        Supports any count 2–6, including odd counts (3, 5).
        """
        assignment = ['red', 'blue', 'lawn green', 'gray0', 'yellow', 'purple']
        return assignment[:self.n_players]

    def _precompute_geometry(self):
        """Compute goal zone indices and deepest goal cell once from a throwaway board."""
        tmp = HexBoard()
        for colour in COLOUR_ORDER:
            goal_colour = COLOUR_OPPOSITES[colour]
            idxs = tmp.axial_of_colour(goal_colour)   # list of 10 int indices
            self._goal_zone_indices[colour] = set(idxs)
            # Target = deepest cell in goal zone (max Chebyshev distance from origin).
            # This is the tip of the corner (e.g. (4,-8) for blue).  Using the tip
            # rather than the centroid gives a stronger incentive to push pieces all
            # the way into the corner rather than stopping near the entry cells.
            deepest_idx = max(idxs, key=lambda i: max(
                abs(tmp.cells[i].q), abs(tmp.cells[i].r),
                abs(-tmp.cells[i].q - tmp.cells[i].r)))
            dc = tmp.cells[deepest_idx]
            self._goal_targets[colour] = (float(dc.q), float(dc.r))

    def _init_board(self):
        """Create a fresh board and spawn all active players' pins at home positions."""
        self._board = HexBoard()
        self._pins  = {}
        for colour in self._active_colours:
            idxs = self._board.axial_of_colour(colour)[:10]   # (r,q)-sorted order
            self._pins[colour] = [
                Pin(self._board, idxs[i], id=i, color=colour)
                for i in range(10)
            ]

    def _build_observation(self) -> np.ndarray:
        """
        Build a (9, 121) feature tensor from the current player's perspective
        and return it flattened to (1089,).

        Channel layout:
            0   — own pins (binary)
            1–5 — opponent pins, clockwise from current player in COLOUR_ORDER
            6   — own goal zone (binary)
            7   — distance field: normalised squared axial distance to deepest goal cell
            8   — valid cells mask (all 1)
        """
        obs     = np.zeros((self.N_CH, self.N_CELLS), dtype=np.float32)
        current = self._active_colours[self._turn_idx]

        # CH 0: own pins
        for pin in self._pins[current]:
            obs[0, pin.axialindex] = 1.0

        # CH 1–5: opponents, clockwise from current player
        cur_pos = COLOUR_ORDER.index(current)
        ch = 1
        for offset in range(1, 6):
            candidate = COLOUR_ORDER[(cur_pos + offset) % 6]
            if candidate in self._active_colours:
                for pin in self._pins[candidate]:
                    obs[ch, pin.axialindex] = 1.0
            ch += 1

        # CH 6: own goal zone
        for idx in self._goal_zone_indices[current]:
            obs[6, idx] = 1.0

        # CH 7: distance field
        cq, cr = self._goal_targets[current]
        raw = np.array(
            [self._axial_dist_sq(i, cq, cr) for i in range(self.N_CELLS)],
            dtype=np.float32,
        )
        max_d = raw.max()
        obs[7] = raw / max_d if max_d > 0 else raw

        # CH 8: valid cells
        obs[8, :] = 1.0

        return obs.flatten()

    def _build_action_mask(self) -> np.ndarray:
        """Return a binary (1210,) int8 mask; 1 = legal action.

        Rule enforced here: once a pin is inside the goal zone it cannot leave.
        It may still move to other cells within the goal zone (to fill deeper
        spots), but every destination outside the goal zone is masked out.
        This matches the standard Chinese Checkers rule and prevents the RL
        agent from learning to evacuate pieces that have already arrived.
        """
        mask      = np.zeros(self.ACT_SIZE, dtype=np.int8)
        current   = self._active_colours[self._turn_idx]
        goal_zone = self._goal_zone_indices[current]
        for pin_id, pin in enumerate(self._pins[current]):
            in_goal = pin.axialindex in goal_zone
            for dest in pin.getPossibleMoves():
                if in_goal and dest not in goal_zone:
                    continue  # cannot leave the goal zone once entered
                mask[pin_id * self.N_CELLS + dest] = 1
        return mask

    def _player_pos_hash(self, colour: str) -> int:
        """Hash of this player's own piece positions only, for per-player cycle detection."""
        return hash(tuple(sorted(p.axialindex for p in self._pins[colour])))

    def _shaping_reward(self, colour: str, d_before: float,
                        d_after: float, pos_count: int) -> float:
        r = 0.0

        # 1. Distance reduction: positive when pins moved closer to goal
        r += -(d_after - d_before) / self._d_max[colour]

        # 2. Goal zone entry bonus: +0.2 first time each pin enters goal
        goal_zone = self._goal_zone_indices[colour]
        for pin_id, pin in enumerate(self._pins[colour]):
            if (pin.axialindex in goal_zone
                    and pin_id not in self._pins_entered_goal[colour]):
                r += 0.2
                self._pins_entered_goal[colour].add(pin_id)

        return r

    def _axial_dist_sq(self, idx: int, cq: float, cr: float) -> float:
        """Squared axial (Chebyshev-hex) distance from cell idx to float centroid."""
        cell = self._board.cells[idx]
        dq = cell.q - cq
        dr = cell.r - cr
        ds = -dq - dr
        return max(abs(dq), abs(dr), abs(ds)) ** 2

    def _total_dist_sq(self, colour: str) -> float:
        """Sum of squared axial distances from each pin of colour to its goal centroid."""
        cq, cr = self._goal_targets[colour]
        return sum(self._axial_dist_sq(p.axialindex, cq, cr) for p in self._pins[colour])

    def _check_win(self, colour: str) -> bool:
        """True if all 10 pins of colour are in the opposite corner zone."""
        opp = COLOUR_OPPOSITES[colour]
        return all(
            self._board.cells[p.axialindex].postype == opp
            for p in self._pins[colour]
        )

    def _board_state_hash(self) -> int:
        """Stable hash of the full board state for cycle detection."""
        return hash(tuple(
            (c, tuple(sorted(p.axialindex for p in self._pins[c])))
            for c in self._active_colours
        ))

    def _advance_turn(self):
        self._turn_idx = (self._turn_idx + 1) % len(self._active_colours)


# ---------------------------------------------------------------------------
# Smoke test — run directly to verify correctness
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    def run_smoke(n_players, n_episodes=3, max_steps=500):
        print(f"\n--- Smoke test: n_players={n_players} ---")
        env = ChineseCheckersEnv(n_players=n_players, render_mode="ansi")

        for ep in range(n_episodes):
            obs, info = env.reset()
            assert obs.shape == (1089,), f"Bad obs shape: {obs.shape}"
            assert obs.min() >= 0.0 and obs.max() <= 1.0, "Obs out of [0,1]"
            mask = info["action_mask"]
            assert mask.shape == (1210,), f"Bad mask shape: {mask.shape}"
            assert mask.sum() > 0, "No legal moves at reset"

            for step in range(max_steps):
                legal = np.where(mask)[0]
                action = int(random.choice(legal))
                obs, reward, terminated, truncated, info = env.step(action)

                assert obs.shape == (1089,)
                assert obs.min() >= 0.0 and obs.max() <= 1.0
                mask = info["action_mask"]

                if terminated or truncated:
                    print(f"  ep {ep}: finished at step {step+1}, "
                          f"terminated={terminated}, truncated={truncated}, "
                          f"reward={reward:.3f}")
                    break
            else:
                print(f"  ep {ep}: reached max_steps={max_steps} without termination")

        print(f"  render (ansi):")
        env.reset()
        env.render()
        print("PASS")

    run_smoke(n_players=2)
    run_smoke(n_players=4)
    run_smoke(n_players=6)
