def _cell_depth(cell_idx: int, cells) -> int:
    """Chebyshev distance from board origin.  Corner cells have depth > 4."""
    c = cells[cell_idx]
    return max(abs(c.q), abs(c.r), abs(-c.q - c.r))


def _dist_to_goal(cell_idx: int, goal_zone_indices, cells) -> int:
    """
    Distance metric for movement toward and *deep into* the goal zone.

    Outside goal zone:
        Chebyshev distance to the nearest unoccupied goal cell (approach).

    Inside goal zone:
        Chebyshev distance to the nearest unoccupied goal cell that is
        *deeper* (higher corner depth) than the current cell.
        Returns 0 when already at maximum available depth — the pin is
        optimally placed and should not be moved.

    Returns 0 for degenerate cases (all goal cells occupied, etc.).
    """
    in_goal = cell_idx in goal_zone_indices
    pc      = cells[cell_idx]

    if in_goal:
        current_depth = _cell_depth(cell_idx, cells)
        best = float('inf')
        for gidx in goal_zone_indices:
            if cells[gidx].occupied:
                continue
            if _cell_depth(gidx, cells) > current_depth:
                gc  = cells[gidx]
                dq  = pc.q - gc.q
                dr  = pc.r - gc.r
                d   = max(abs(dq), abs(dr), abs(-dq - dr))
                if d < best:
                    best = d
        return int(best) if best != float('inf') else 0
    else:
        best = float('inf')
        for gidx in goal_zone_indices:
            if cells[gidx].occupied:
                continue
            gc  = cells[gidx]
            dq  = pc.q - gc.q
            dr  = pc.r - gc.r
            d   = max(abs(dq), abs(dr), abs(-dq - dr))
            if d < best:
                best = d
        return int(best) if best != float('inf') else 0


def select_move(pins, cells, n_cells: int, goal_zone_indices, rng=None) -> int:
    """
    Core heuristic logic. Returns encoded action = pin_id * n_cells + dest_idx.

    Approach phase (pin outside goal zone):
        Select the move that most reduces the Chebyshev distance to the nearest
        unoccupied goal cell.  Tiebreak: prefer the trailing pin (furthest from
        its nearest empty goal cell) so no piece is left behind.

    Deep-fill phase (pin inside goal zone):
        A pin that has entered the goal zone is allowed to move WITHIN the goal
        zone to a deeper (higher corner-depth) empty cell.  It is never allowed
        to leave the goal zone.  Once it is at the deepest available position
        (no empty deeper cell exists), it is skipped.

    Cycle-break rule: when the greedy choice makes no progress (stuck at a
    Chebyshev local minimum), a random legal non-exit move is chosen instead.

    Args:
        pins:              list of 10 Pin objects for the current player
        cells:             board.cells list (BoardPosition objects, 0..120)
        n_cells:           121
        goal_zone_indices: set of cell indices for the player's goal corner
        rng:               optional random.Random for cycle-breaking;
                           if None greedy is always used (may cycle)

    Raises:
        RuntimeError: if truly no legal moves exist for any pin (caller must
                      check info["no_legal_moves"] before calling act()).
    """
    best_action       = None
    best_reduction    = -float('inf')
    best_current_dist = -float('inf')
    all_legal         = []

    for pin_id, pin in enumerate(pins):
        in_goal      = pin.axialindex in goal_zone_indices
        current_dist = _dist_to_goal(pin.axialindex, goal_zone_indices, cells)

        # Skip pins that are already at the deepest available goal position.
        if in_goal and current_dist == 0:
            continue

        for dest in pin.getPossibleMoves():
            # Goal zone is a one-way door — pins inside cannot leave.
            if in_goal and dest not in goal_zone_indices:
                continue

            new_dist  = _dist_to_goal(dest, goal_zone_indices, cells)
            reduction = current_dist - new_dist
            all_legal.append(pin_id * n_cells + dest)

            if (reduction > best_reduction or
                    (reduction == best_reduction and
                     current_dist > best_current_dist)):
                best_reduction    = reduction
                best_current_dist = current_dist
                best_action       = pin_id * n_cells + dest

    if best_action is None:
        # All remaining pins are at max depth inside the goal zone (no deeper
        # empty cells exist).  The env still permits within-goal-zone moves for
        # these pins; pick the first one as a legal no-op rather than crashing.
        for pin_id, pin in enumerate(pins):
            if pin.axialindex not in goal_zone_indices:
                continue
            for dest in pin.getPossibleMoves():
                if dest in goal_zone_indices:
                    return pin_id * n_cells + dest
        raise RuntimeError(
            "select_move: no legal moves — env should have auto-skipped this turn")

    # Greedy is stuck (no distance reduction possible) — randomise to escape the
    # local minimum caused by Chebyshev distance not reflecting the real path.
    if best_reduction <= 0 and rng is not None and all_legal:
        return rng.choice(all_legal)

    return best_action


class HeuristicAgent:
    """
    Greedy heuristic agent with cycle-breaking.

    Moves each pin toward the nearest unoccupied goal cell; never moves a pin
    that is already inside the goal zone.  When the greedy move would make no
    progress (all reachable cells are equidistant or further from the goal), a
    random legal move is chosen instead to escape the local minimum.

    The act(env) -> int interface is the shared contract for all agents
    (HeuristicAgent, RL network agent, MCTS agent), making opponent-pool
    substitution trivial.

    Args:
        seed: seed for the internal RNG used for cycle-breaking.
    """

    def __init__(self, seed: int = 0):
        import random
        self._rng = random.Random(seed)

    def act(self, env) -> int:
        colour = env._active_colours[env._turn_idx]
        return select_move(
            env._pins[colour],
            env._board.cells,
            env.N_CELLS,
            goal_zone_indices=env._goal_zone_indices[colour],
            rng=self._rng,
        )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import pathlib
    import time

    sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
    from src.env.chinese_checkers_env import ChineseCheckersEnv

    agent = HeuristicAgent()
    env   = ChineseCheckersEnv(n_players=2)
    N     = 20
    wins  = 0
    steps = 0
    t0    = time.perf_counter()

    for _ in range(N):
        obs, info = env.reset()
        for _ in range(1000):
            action = agent.act(env)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                if terminated and info["rewards"][info["current_player"]] != -2.0:
                    wins += 1
                break

    elapsed = time.perf_counter() - t0
    print(f"{N} games | wins={wins}/{N} | avg_len={steps/N:.1f} | {steps/elapsed:.0f} steps/s")
    assert wins == N, f"Expected {N} wins, got {wins} — heuristic may be cycling or drawing"
    print("PASS")
