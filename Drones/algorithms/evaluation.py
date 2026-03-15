from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from world.game_state import GameState


def evaluation_function(state: GameState) -> float:
    """
    Evaluation function for non-terminal states of the drone vs. hunters game.

    A good evaluation function can consider multiple factors, such as:
      (a) BFS distance from drone to nearest delivery point (closer is better).
          Uses actual path distance so walls and terrain are respected.
      (b) BFS distance from each hunter to the drone, traversing only normal
          terrain ('.' / ' ').  Hunters blocked by mountains, fog, or storms
          are treated as unreachable (distance = inf) and pose no threat.
      (c) BFS distance to a "safe" position (i.e., a position that is not in the path of any hunter).
      (d) Number of pending deliveries (fewer is better).
      (e) Current score (higher is better).
      (f) Delivery urgency: reward the drone for being close to a delivery it can
          reach strictly before any hunter, so it commits to nearby pickups
          rather than oscillating in place out of excessive hunter fear.
      (g) Adding a revisit penalty can help prevent the drone from getting stuck in cycles.

    Returns a value in [-1000, +1000].

    Tips:
    - Use state.get_drone_position() to get the drone's current (x, y) position.
    - Use state.get_hunter_positions() to get the list of hunter (x, y) positions.
    - Use state.get_pending_deliveries() to get the set of pending delivery (x, y) positions.
    - Use state.get_score() to get the current game score.
    - Use state.get_layout() to get the current layout.
    - Use state.is_win() and state.is_lose() to check terminal states.
    - Use bfs_distance(layout, start, goal, hunter_restricted) from algorithms.utils
      for cached BFS distances. hunter_restricted=True for hunter-only terrain.
    - Use dijkstra(layout, start, goal) from algorithms.utils for cached
      terrain-weighted shortest paths, returning (cost, path).
    - Consider edge cases: no pending deliveries, no hunters nearby.
    - A good evaluation function balances delivery progress with hunter avoidance.
    """

    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0

    layout = state.get_layout()
    drone_pos = state.get_drone_position()
    hunter_positions = state.get_hunter_positions()
    pending = state.get_pending_deliveries()
    score = state.get_score()

    value = 0.0

    # ── (e) Score base ──────────────────────────────────────────────────────
    value += score * 2.0

    # ── (d) Penalty por entregas pendientes ────────────────────────────────
    value -= len(pending) * 50.0

    # ── (a) Distancia al delivery más cercano ──────────────────────────────
    if pending:
        drone_to_delivery = [
            bfs_distance(layout, drone_pos, dp) for dp in pending
        ]
        min_delivery_dist = min(drone_to_delivery)
        value -= min_delivery_dist * 10.0
    else:
        min_delivery_dist = float("inf")

    # ── (b) Distancia de los hunters al drone ──────────────────────────────
    DANGER_RADIUS = 5
    hunter_threat = 0.0

    hunter_distances = [
        bfs_distance(layout, h, drone_pos, hunter_restricted=True)
        for h in hunter_positions
    ]

    for h_dist in hunter_distances:
        if h_dist == float("inf"):
            continue  # hunter bloqueado, no amenaza
        if h_dist == 0:
            return -1000.0  # colisión
        if h_dist <= DANGER_RADIUS:
            hunter_threat += 100.0 / h_dist  # más cerca → más peligro

    value -= hunter_threat

    # ── (f) Urgencia de entrega: premia ir a un delivery antes que el hunter ─
    if pending and hunter_positions:
        for dp, d2d in zip(pending, drone_to_delivery):
            min_hunter_to_dp = min(
                bfs_distance(layout, h, dp, hunter_restricted=True)
                for h in hunter_positions
            )
            if d2d < min_hunter_to_dp:
                # Drone llega antes que cualquier hunter → bonus
                value += 30.0 / (d2d + 1)

    # ── (c) Premio por estar lejos del hunter más cercano ──────────────────
    if hunter_distances:
        min_hunter_dist = min(
            d for d in hunter_distances if d != float("inf")
        ) if any(d != float("inf") for d in hunter_distances) else float("inf")

        if min_hunter_dist != float("inf"):
            value += min(min_hunter_dist * 5.0, 100.0)

    return max(-1000.0, min(1000.0, value))
