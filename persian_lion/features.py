"""59/88 geometry schema with anonymous, physically gated motion.

Every frame's team sets are canonically sorted before any reduction or matching.
Player slots are not identities. No ownership, status, human event, clock, label,
or absolute player-coordinate feature is used. All context stays within a half.
"""
from __future__ import annotations

import warnings
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment

from .data import Match, blocks


FEATURE_VERSION = "anonymous_assignment_geometry_1"
GLOBAL_FEATURES = 59
TEAM_FEATURES = 88
MOTION_RADIUS = 5
MAX_PLAYER_SPEED = 13.0
ASSIGNMENT_CHUNK_FRAMES = 2048


def nanmean(x, axis):
    count = np.isfinite(x).sum(axis=axis)
    total = np.nansum(x, axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=np.float32), where=count > 0)


def moving(x, radius):
    """Finite-value mean; empty windows remain missing."""
    finite = np.isfinite(x)
    den = uniform_filter1d(finite.astype(np.float32), 2 * radius + 1, axis=0, mode="constant")
    num = uniform_filter1d(np.where(finite, x, 0).astype(np.float32), 2 * radius + 1, axis=0, mode="constant")
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-6)


def shift(x, amount):
    out = np.full_like(x, np.nan)
    if amount == 0:
        return x.copy()
    if abs(amount) >= len(x):
        return out
    if amount > 0:
        out[amount:] = x[:-amount]
    else:
        out[:amount] = x[-amount:]
    return out


def velocity(x, fps, radius=3, max_speed=70.0):
    """Single-ball velocity; never applied to player slot trajectories."""
    v = (shift(x, -radius) - shift(x, radius)) * (fps / (2 * radius))
    v[np.linalg.norm(v, axis=-1) > max_speed] = np.nan
    return v


def context_indices(valid):
    n = len(valid)
    idx = np.arange(n)
    past = np.maximum.accumulate(np.where(valid, idx, -1))
    future = np.minimum.accumulate(np.where(valid, idx, n)[::-1])[::-1]
    return past, future


def carry(values, valid):
    past, future = context_indices(valid)
    previous = values[np.maximum(past, 0)].copy()
    following = values[np.minimum(future, len(valid) - 1)].copy()
    previous[past < 0] = np.nan
    following[future >= len(valid)] = np.nan
    return previous, following, past, future


def canonical_players(players, teams):
    """Sort each team's points by (x,y) per frame; finite points come first.

    Missing/partial points follow in deterministic coordinate order. Slots retain
    team membership only; the ordering is recomputed independently every frame.
    """
    players, teams = np.asarray(players), np.asarray(teams)
    if players.ndim != 3 or players.shape[1:] != (len(teams), 2):
        raise ValueError("Expected players[N,S,2] with teams[S]")
    if not np.isin(teams, (0, 1)).all() or set(teams.tolist()) != {0, 1}:
        raise ValueError("Explicit membership of both teams is required")
    if np.isinf(players).any():
        raise ValueError("Infinite coordinates are not missing observations")
    output = np.empty_like(players)
    for side in (0, 1):
        slots = np.flatnonzero(teams == side)
        points = players[:, slots]
        finite = np.isfinite(points).all(axis=2)
        x = np.where(np.isfinite(points[:, :, 0]), points[:, :, 0], np.inf)
        y = np.where(np.isfinite(points[:, :, 1]), points[:, :, 1], np.inf)
        order = np.lexsort((y, x, ~finite), axis=1)
        output[:, slots] = np.take_along_axis(points, order[:, :, None], axis=1)
    return output


def _canonical_velocity(players, teams, fps, radius, max_speed, chunk_frames):
    """Input is canonically sorted and contains one contiguous period only."""
    if fps <= 0 or not isinstance(radius, int) or radius <= 0 or max_speed <= 0 or chunk_frames <= 0:
        raise ValueError("Motion cadence, radius, speed and chunk size must be positive")
    n = len(players)
    output = np.full(players.shape, np.nan, dtype=np.float32)
    if n <= 2 * radius:
        return output
    max_distance = max_speed * radius / fps
    for side in (0, 1):
        slots = np.flatnonzero(teams == side)
        points = players[:, slots]
        count = len(slots)
        following = np.full((n, count), -1, np.int32)
        preceding = np.full((n, count), -1, np.int32)
        # A rejected edge costs more than every possible valid edge combined.
        # Thus assignment first maximizes valid cardinality, then minimizes total
        # Euclidean distance. Missing points can never become accepted matches.
        penalty = (count + 1) * (max_distance + 1)
        for a in range(0, n - radius, chunk_frames):
            b = min(n - radius, a + chunk_frames)
            distance = np.linalg.norm(points[a:b, :, None] - points[a+radius:b+radius, None, :], axis=3)
            admissible = np.isfinite(distance) & (distance <= max_distance)
            costs = np.where(admissible, distance, penalty)
            for local in np.flatnonzero(admissible.any(axis=(1, 2))):
                rows, columns = linear_sum_assignment(costs[local])
                keep = admissible[local, rows, columns]
                rows, columns = rows[keep], columns[keep]
                frame = a + int(local)
                following[frame, rows] = columns
                preceding[frame + radius, columns] = rows
        # Require independent evidence on both sides of the current set point.
        frames, current = np.nonzero((following >= 0) & (preceding >= 0))
        before = points[frames-radius, preceding[frames, current]]
        after = points[frames+radius, following[frames, current]]
        central = (after-before) * (fps / (2 * radius))
        valid = np.isfinite(central).all(axis=1) & (np.linalg.norm(central, axis=1) <= max_speed)
        output[frames[valid], slots[current[valid]]] = central[valid]
    return output


def anonymous_velocity(players, teams, fps=25, radius=MOTION_RADIUS,
                       max_speed=MAX_PLAYER_SPEED, chunk_frames=ASSIGNMENT_CHUNK_FRAMES):
    """Canonical-order central velocities from gated anonymous set assignments.

    Call separately for each half. Current points must have one-to-one matches
    at both ±radius frames, each within max_speed * radius / fps metres. This
    estimates local motion; it does not establish player identity or fill gaps.
    """
    canonical = canonical_players(players, teams)
    return _canonical_velocity(canonical, np.asarray(teams), fps, radius, max_speed, chunk_frames)


def _features_one(ball, players, teams, pitch, fps, use_motion=True):
    n = len(ball)
    players = canonical_players(players, teams)
    visible = np.isfinite(ball).all(axis=1)
    ball = ball.copy()
    plausible = (np.abs(ball[:, 0]) < pitch[0]) & (np.abs(ball[:, 1]) < pitch[1])
    ball[~plausible] = np.nan
    visible &= plausible
    bv = velocity(ball, fps)
    bs = np.linalg.norm(bv, axis=1)
    pv = (_canonical_velocity(players, teams, fps, MOTION_RADIUS, MAX_PLAYER_SPEED, ASSIGNMENT_CHUNK_FRAMES)
          if use_motion else np.full(players.shape, np.nan, np.float32))
    ps = np.linalg.norm(pv, axis=2)
    distances = np.linalg.norm(players - ball[:, None, :], axis=2)
    before, after, pi, fi = carry(ball, visible)
    age_before = np.minimum((np.arange(n) - pi) / fps, 60.)
    age_after = np.minimum((fi - np.arange(n)) / fps, 60.)
    age_before[pi < 0], age_after[fi >= n] = 60., 60.
    edge_x = pitch[0] / 2 - np.abs(ball[:, 0])
    edge_y = pitch[1] / 2 - np.abs(ball[:, 1])
    team_base = []
    for team in (0, 1):
        choose = teams == team
        p = players[:, choose]
        v = pv[:, choose]
        speeds = ps[:, choose]
        dist = distances[:, choose]
        order = np.sort(np.where(np.isfinite(dist), dist, np.inf), axis=1)
        nearest = np.pad(order, ((0, 0), (0, max(0, 3 - order.shape[1]))), constant_values=np.inf)[:, :3]
        nearest[~np.isfinite(nearest)] = np.nan
        centre = nanmean(p, 1)
        spread = np.sqrt(nanmean((p - centre[:, None]) ** 2, 1))
        count = np.isfinite(p).all(axis=2).sum(axis=1).astype(np.float32)
        centre_ball = np.linalg.norm(centre - ball, axis=1)
        speed_mean = nanmean(speeds, 1)
        speed_rms = np.sqrt(nanmean(speeds ** 2, 1))
        near_idx = np.argmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
        near_v = v[np.arange(n), near_idx]
        relative_speed = np.linalg.norm(near_v - bv, axis=1)
        relative_speed[~np.isfinite(nearest[:, 0])] = np.nan
        contact = np.exp(-((nearest[:, 0] / 2.) ** 2))
        close_count = (dist < 3.).sum(axis=1).astype(np.float32)
        close_count[~visible] = np.nan
        dist_before = np.linalg.norm(p - before[:, None], axis=2)
        dist_after = np.linalg.norm(p - after[:, None], axis=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            last_distance = np.nanmin(dist_before, axis=1)
            next_distance = np.nanmin(dist_after, axis=1)
        tb = np.column_stack([nearest, count, spread, centre_ball, speed_mean, speed_rms,
                              relative_speed, contact, close_count, last_distance, next_distance])
        team_base.append(tb)
    t = np.stack(team_base, axis=1).astype(np.float32)
    all_speed = nanmean(ps, 1)
    global_base = np.column_stack([visible, age_before, age_after, bs, edge_x, edge_y,
                                  np.minimum(edge_x, edge_y), all_speed,
                                  np.isfinite(players).all(axis=2).sum(axis=1)])
    d0, d1 = t[:, 0, 0], t[:, 1, 0]
    encounter = np.minimum(d0, d1) < 2.5
    indicator = np.column_stack([d0 < d1, d1 < d0]).astype(np.float32)
    prev, nxt, p_enc, f_enc = carry(indicator, encounter)
    enc_age_p = np.minimum((np.arange(n) - p_enc) / fps, 60.)
    enc_age_f = np.minimum((f_enc - np.arange(n)) / fps, 60.)
    enc_age_p[p_enc < 0], enc_age_f[f_enc >= n] = 60., 60.
    g_parts = [global_base, enc_age_p[:, None], enc_age_f[:, None]]
    t_parts = [t, prev[:, :, None], nxt[:, :, None]]
    for radius in (5, 20, 50, 125):
        g_parts.append(moving(global_base, radius))
        t_parts.append(moving(t, radius))
    for displacement in (-25, 25, -75, 75):
        t_parts.append(shift(t[:, :, [0, 6, 7, 10]], displacement))
        g_parts.append(shift(global_base[:, [3, 6, 7]], displacement))
    g = np.column_stack(g_parts).astype(np.float32)
    t = np.concatenate(t_parts, axis=2).astype(np.float32)
    return g, t


def extract(match: Match, use_motion=True):
    """Return global[N,59], team[N,2,88]; labels never enter features.

    ``use_motion=False`` removes all player velocity evidence using NaNs before
    aggregate and context calculations. All other feature definitions remain.
    """
    match.validate()
    out_g, out_t = [], []
    for a, b in blocks(match.period):
        g, t = _features_one(match.ball[a:b], match.players[a:b], match.teams, match.pitch, match.fps, use_motion)
        out_g.append(g)
        out_t.append(t)
    return np.concatenate(out_g), np.concatenate(out_t)


def live_matrix(g, t):
    return np.column_stack([g, (t[:, 0] + t[:, 1]) / 2,
                            np.abs(t[:, 0] - t[:, 1])]).astype(np.float32)


def team_matrix(g, t, side):
    return np.column_stack([g, t[:, side], t[:, 1-side], t[:, side] - t[:, 1-side]]).astype(np.float32)
