"""Dead-ball (stoppage) features for the football dead-ball classifier.

One row per frame, built from the anonymous player positions and the ball position when it is seen:

  * instant features (`instant_features`): geometry that a stoppage leaves in the players' positions:
    standing / VAR / injury (almost nobody moves), hydration or half-time (players walk to a touchline),
    penalty, corner, free-kick wall, throw-in, goal kick, kick-off.
  * ball features: where the ball is relative to the touchline and goal line, how fast it moves,
    whether it is observed. When the ball is hidden these are missing (NaN) and the model falls back on
    the player features; the model was trained with hidden-ball stretches for that reason.
  * time context (`_temporal_one`): rolling means over +-1, 5 and 10 s, and past / future / change
    over 1, 3 and 10 s. At most about 10 s of future frames are used and nothing crosses a half.

Coordinates are centred metres: x in [-L/2, L/2], y in [-W/2, W/2]. Features are symmetric under
left/right and home/away swaps, so they do not depend on the direction of play. No label, event,
clock or match identifier is ever an input.
"""
from __future__ import annotations

import numpy as np

from .data import blocks
from .features import anonymous_velocity, canonical_players, moving, shift

PEN_AREA_DEPTH, PEN_AREA_HALF_WIDTH = 16.5, 20.16
PEN_SPOT = 11.0
WALL_RADIUS = 3.5          # m: same-team players this close form a wall
KICK_DISTANCE = 9.15       # m (10 yards)
CHUNK = 1500


def _nanmean(x, axis=1):
    count = np.isfinite(x).sum(axis=axis)
    total = np.nansum(x, axis=axis)
    return np.divide(total, count, out=np.full(total.shape, np.nan, np.float32), where=count > 0)


def instant_features(players, teams, pitch=(105.0, 68.0)):
    """(T, F) features for every frame separately. players: T x S x 2 centred metres."""
    L, W = pitch
    hl, hw = L / 2, W / 2
    x, y = players[:, :, 0], players[:, :, 1]
    valid = np.isfinite(x) & np.isfinite(y)
    n = valid.sum(axis=1).astype(np.float32)
    safe = np.maximum(n, 1)

    # speed over +-0.5 s, and approach speed towards the nearest touchline
    k = 12
    dx = shift(x, -k) - shift(x, k)
    dy = shift(y, -k) - shift(y, k)
    speed = np.hypot(dx, dy) / (2 * k / 25.0)
    edge_before = np.abs(shift(y, k))
    edge_after = np.abs(shift(y, -k))
    approach = (edge_after - edge_before) / (2 * k / 25.0)       # >0 = moving towards the touchline
    approach = np.where(np.abs(y) > 0, approach, np.nan)

    cols = {}
    cols["n_players"] = n
    cols["mean_speed"] = _nanmean(speed)
    cols["frac_still"] = np.nansum(speed < 0.6, axis=1) / safe
    cols["frac_fast"] = np.nansum(speed > 3.0, axis=1) / safe
    cols["speed_std"] = np.nanstd(np.where(np.isfinite(speed), speed, np.nan), axis=1)
    cols["approach_mean"] = _nanmean(approach)
    cols["frac_approach"] = np.nansum(approach > 0.5, axis=1) / safe

    d_touch = hw - np.abs(y)          # distance to nearest touchline (y = +-W/2)
    d_goal = hl - np.abs(x)                         # distance to nearest goal line
    for r in (3.0, 8.0):
        cols["frac_touch%d" % r] = np.nansum(valid & (d_touch < r), axis=1) / safe
    cols["frac_goalzone"] = np.nansum(valid & (d_goal < 20.0), axis=1) / safe
    cols["frac_goalzone_far"] = np.nansum(valid & (d_goal > 20.0) & (np.abs(x) > 0), axis=1) / safe
    in_box = valid & (d_goal < PEN_AREA_DEPTH) & (np.abs(y) < PEN_AREA_HALF_WIDTH)
    cols["n_in_box_max"] = np.maximum((in_box & (x < 0)).sum(axis=1), (in_box & (x > 0)).sum(axis=1)).astype(np.float32)
    cols["frac_centre"] = np.nansum(valid & (np.hypot(x, y) < 10.0), axis=1) / safe
    cols["spread_x"] = np.nanstd(np.where(valid, x, np.nan), axis=1)
    cols["spread_y"] = np.nanstd(np.where(valid, y, np.nan), axis=1)

    # outside the lines: substitutes, medics, players leaving
    outside = valid & ((np.abs(y) > hw + 0.5) | (np.abs(x) > hl + 0.5))
    cols["n_outside"] = outside.sum(axis=1).astype(np.float32)
    cols["n_outside_mid"] = (outside & (np.abs(x) < 15.0) & (np.abs(y) > hw + 0.5)).sum(axis=1).astype(np.float32)

    # throw-in: nearest player to a touchline, and how many stand within 2 m of it
    near = np.where(valid, d_touch, np.inf)
    cols["touch_min"] = np.where(n > 0, near.min(axis=1), np.nan)
    cols["n_touch2"] = (valid & (d_touch < 2.0)).sum(axis=1).astype(np.float32)
    # the player nearest a touchline: is he standing still, and how many teammates/opponents are near him?
    t_i = near.argmin(axis=1)[:, None]
    cols["touch_player_speed"] = np.take_along_axis(speed, t_i, axis=1)[:, 0]
    tx, ty = np.take_along_axis(x, t_i, axis=1), np.take_along_axis(y, t_i, axis=1)
    cols["n_within15_of_touch"] = (valid & (np.hypot(x - tx, y - ty) < 15.0)).sum(axis=1).astype(np.float32)

    # goal kick: one player alone near a goal while the other team stays out of the 25 m zone
    gk_d, gk_count, gk_opp = [], [], []
    for sx in (-1.0, 1.0):
        d = np.where(valid, np.hypot(x - sx * hl, y), np.inf)
        i = d.argmin(axis=1)
        gk_d.append(d.min(axis=1))
        gk_count.append((d < 25.0).sum(axis=1))
        opp = (teams[None, :] != teams[i][:, None]) & valid
        gk_opp.append(np.where(opp, d, np.inf).min(axis=1))
    g = np.argmin(np.stack(gk_d), axis=0)
    pick = lambda arr: np.stack(arr)[g, np.arange(len(g))]
    cols["gk_d"] = np.where(n > 0, pick(gk_d), np.nan)
    cols["gk_n25"] = pick(gk_count).astype(np.float32)
    cols["gk_opp_min"] = np.where(np.isfinite(pick(gk_opp)), pick(gk_opp), np.nan)

    # penalty: one player at a spot, a ring of players 9.15-14 m away
    pen = []
    for sx in (-1.0, 1.0):
        d = np.hypot(x - sx * (hl - PEN_SPOT), y)
        d = np.where(valid, d, np.inf)
        d1 = d.min(axis=1)
        ring = ((d > 9.0) & (d < 14.0)).sum(axis=1)
        pen.append((d1, ring))
    best = np.argmin(np.stack([p[0] for p in pen]), axis=0)
    cols["pen_d1"] = np.where(n > 0, np.choose(best, [p[0] for p in pen]), np.nan)
    cols["pen_ring"] = np.choose(best, [p[1] for p in pen]).astype(np.float32)

    # corner: nearest player to a corner, and the box count at that end
    cd, cb = [], []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            d = np.where(valid, np.hypot(x - sx * hl, y - sy * hw), np.inf)
            cd.append(d.min(axis=1))
            cb.append((in_box & (np.sign(x) == sx)).sum(axis=1))
    cd, cb = np.stack(cd), np.stack(cb)
    b = np.argmin(cd, axis=0)
    cols["corner_d"] = np.where(n > 0, cd[b, np.arange(len(b))], np.nan)
    cols["corner_box"] = cb[b, np.arange(len(b))].astype(np.float32)

    # free kick: biggest same-team bunch (wall) and the nearest opponent to its centre
    wall_n = np.zeros(len(x), np.float32)
    kick_d = np.full(len(x), np.nan, np.float32)
    for a in range(0, len(x), CHUNK):
        sl = slice(a, min(a + CHUNK, len(x)))
        px, py, pv = x[sl], y[sl], valid[sl]
        dist = np.hypot(px[:, :, None] - px[:, None, :], py[:, :, None] - py[:, None, :])
        same = (teams[None, :, None] == teams[None, None, :]) & pv[:, :, None] & pv[:, None, :]
        count = ((dist < WALL_RADIUS) & same).sum(axis=2)          # per player: bunch size around it
        for side in (0, 1):
            c = np.where(teams[None, :] == side, count, 0)
            i = c.argmax(axis=1)
            size = c[np.arange(len(i)), i]
            cx = px[np.arange(len(i)), i]
            cy = py[np.arange(len(i)), i]
            opp = (teams[None, :] != side) & pv
            do = np.where(opp, np.hypot(px - cx[:, None], py - cy[:, None]), np.inf).min(axis=1)
            better = size > wall_n[sl]
            wall_n[sl] = np.where(better, size, wall_n[sl])
            kick_d[sl] = np.where(better, np.where(np.isfinite(do), do, np.nan), kick_d[sl])
    cols["wall_n"] = wall_n
    cols["kick_dist"] = kick_d
    cols["freekick_flag"] = ((wall_n >= 3) & (np.abs(kick_d - KICK_DISTANCE) < 1.5)).astype(np.float32)

    names = list(cols)
    return names, np.column_stack([cols[k] for k in names]).astype(np.float32)


ROLL = (25, 125, 250)           # +-1 s, +-5 s, +-10 s context windows
ROLL_ON = ("gk_n25", "touch_player_speed", "n_within15_of_touch", "mean_speed", "frac_still", "frac_touch3", "frac_touch8", "frac_goalzone", "n_in_box_max",
           "pen_ring", "corner_box", "wall_n", "n_touch2", "n_outside", "frac_approach", "freekick_flag",
           "frac_centre")
PRESENCE = (("corner_d", 2.5), ("pen_d1", 2.5), ("touch_min", 1.0))   # "someone stays there" for seconds


TEMPORAL_SCHEMA = "touchline_temporal_v2"   # name stored in models/deadball.joblib
MAX_BOUNDARY_AGE_S = 2.0   # a remembered ball position older than this is not used
MAX_BALL_SPEED_MPS = 60    # faster ball movement between frames is a tracking glitch
SIDELINE_BAND_M = 5        # a player this close to the touchline can be a throw-in taker
LOCAL_WINDOW_M = 15        # ... and counts as local if within this distance along the line
BALL_RADIUS_M = 0.11


def _directional_mean(values, width, future=False):
    """Mean of up to `width` strictly preceding/following INPUT observations."""
    x = np.asarray(values, np.float32)
    if future:
        return _directional_mean(x[::-1], width)[::-1]
    finite = np.isfinite(x)
    sums = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(np.where(finite, x, 0), axis=0, dtype=np.float64)])
    counts = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(finite, axis=0)])
    end = np.arange(len(x))
    start = np.maximum(0, end - width)
    den = counts[end] - counts[start]
    return np.divide(sums[end] - sums[start], den, out=np.full(x.shape, np.nan), where=den > 0).astype(np.float32)


def tracking_segments(match):
    """Half-local windows, additionally split around >=2 s complete blackouts."""
    reliable = np.isfinite(match.ball).all(axis=1) | np.isfinite(match.players).all(axis=2).any(axis=1)
    for a, b in blocks(match.period):
        cuts = {a, b}
        missing = ~reliable[a:b]
        edges = np.r_[0, np.flatnonzero(np.diff(missing.astype(np.int8))) + 1, b-a]
        for left, right in zip(edges[:-1], edges[1:]):
            if missing[left] and right-left >= 2*match.fps:
                cuts.update((a+int(left), a+int(right)))
        ordered = sorted(cuts)
        yield from zip(ordered[:-1], ordered[1:])


def _temporal_one(ball, players, teams, pitch, fps):
    players = canonical_players(players, teams)
    valid = np.isfinite(players).all(axis=2)
    n = valid.sum(axis=1)
    safe = np.maximum(n, 1)
    pv = anonymous_velocity(players, teams, fps=fps)
    speed = np.linalg.norm(pv, axis=2)
    approach = np.sign(players[:, :, 1]) * pv[:, :, 1]
    names, base = instant_features(players, teams, pitch)
    idx = {name: i for i, name in enumerate(names)}
    near = pitch[1]/2 - np.abs(players[:, :, 1])
    minimum = np.min(np.where(valid, near, np.inf), axis=1)
    at_min = valid & np.isclose(near, minimum[:, None], atol=1e-5)
    updates = dict(mean_speed=_nanmean(speed), frac_still=np.sum(speed < .6, axis=1)/safe,
                   frac_fast=np.sum(speed > 3, axis=1)/safe, speed_std=np.nanstd(speed, axis=1),
                   approach_mean=_nanmean(approach), frac_approach=np.sum(approach > .5, axis=1)/safe,
                   touch_player_speed=_nanmean(np.where(at_min, speed, np.nan)))
    for key, value in updates.items():
        base[:, idx[key]] = value
    # Resolve tied wall candidates without home/away or slot-order preference.
    for a in range(0, len(ball), CHUNK):
        sl = slice(a, min(a+CHUNK, len(ball)))
        distance = np.linalg.norm(players[sl, :, None]-players[sl, None, :], axis=3)
        same = teams[:, None] == teams[None, :]
        counts = ((distance < WALL_RADIUS) & same).sum(axis=2)
        maximum = counts.max(axis=1)
        candidates = (counts == maximum[:, None]) & valid[sl]
        opposing = np.min(np.where(~same[None] & np.isfinite(distance), distance, np.inf), axis=2)
        best = np.min(np.where(candidates, opposing, np.inf), axis=1)
        best[~np.isfinite(best)] = np.nan
        base[sl, idx["wall_n"]] = maximum
        base[sl, idx["kick_dist"]] = best
        base[sl, idx["freekick_flag"]] = ((maximum >= 3) & (np.abs(best-KICK_DISTANCE) < 1.5))
        near_counts = (distance < 15).sum(axis=2)
        base[sl, idx["n_within15_of_touch"]] = _nanmean(np.where(at_min[sl], near_counts, np.nan))
    columns = [base]
    output_names = list(names)
    for radius in ROLL:
        columns.append(moving(base[:, [idx[name] for name in ROLL_ON]], radius))
        output_names += [f"{name}_m{radius}" for name in ROLL_ON]
    for name, threshold in PRESENCE:
        flag = np.where(np.isfinite(base[:, idx[name]]), base[:, idx[name]] < threshold, np.nan)[:, None]
        for radius in (62, 125):
            columns.append(moving(flag, radius))
            output_names.append(f"{name}_stay{radius}")

    ball = ball.copy()
    ball[~np.isfinite(ball).all(axis=1)] = np.nan
    # positions far outside the pitch (more than one pitch length/width from the centre) are tracking errors
    ball[(np.abs(ball[:, 0]) > pitch[0]) | (np.abs(ball[:, 1]) > pitch[1])] = np.nan
    observed = np.isfinite(ball).all(axis=1)
    touch = pitch[1]/2 - np.abs(ball[:, 1])
    goal = pitch[0]/2 - np.abs(ball[:, 0])
    # Consecutive observations only. A centre crossing is uncertain evidence,
    # not a hard claim that the whole ball crossed or a referee stopped play.
    delta = ball - shift(ball, 1)
    bs = np.linalg.norm(delta, axis=1)*fps
    plausible = bs <= MAX_BALL_SPEED_MPS
    bs[~plausible] = np.nan
    outward = np.sign(ball[:, 1])*delta[:, 1]*fps
    outward[~plausible] = np.nan
    ball_side = np.sign(ball[:, 1])
    side_dist = pitch[1]/2 - players[:, :, 1]*ball_side[:, None]
    local = valid & (side_dist < SIDELINE_BAND_M) & (np.abs(players[:, :, 0]-ball[:, None, 0]) < LOCAL_WINDOW_M)
    dist = np.linalg.norm(players-ball[:, None], axis=2)
    ball_nearest = np.min(np.where(np.isfinite(dist), dist, np.inf), axis=1)
    ball_nearest[~np.isfinite(ball_nearest)] = np.nan
    extra = dict(ball_observed=observed.astype(float), track_visibility=np.minimum(n/22, 1),
                 valid_motion_fraction=np.isfinite(speed).sum(axis=1)/safe,
                 ball_touch_near_m=touch, ball_touch_far_m=pitch[1]/2+np.abs(ball[:, 1]),
                 ball_outside_signed_m=-touch, whole_ball_margin_m=-touch-BALL_RADIUS_M,
                 ball_goal_distance_m=goal, ball_goal_area_lateral_m=np.abs(ball[:, 1]),
                 ball_speed_mps=bs, ball_outward_mps=outward,
                 ball_inward_mps=-outward, local_sideline_count=local.sum(axis=1).astype(float),
                 local_sideline_speed_mps=_nanmean(np.where(local, speed, np.nan)),
                 nearest_ball_player_m=ball_nearest)
    extra["local_sideline_count"][~observed] = np.nan
    indices = np.arange(len(ball))
    previous = np.maximum.accumulate(np.where(observed, indices, -1))
    age = (indices-previous)/fps
    age[previous < 0] = np.nan
    remembered = touch[np.maximum(previous, 0)].copy()
    remembered[(previous < 0) | (age > MAX_BOUNDARY_AGE_S)] = np.nan
    extra["last_boundary_distance_m"] = remembered
    extra["last_ball_age_s"] = np.minimum(age, MAX_BOUNDARY_AGE_S + 1)
    extra["boundary_context_available"] = np.isfinite(remembered).astype(float)
    context_names = ["ball_observed", "track_visibility", "ball_touch_near_m", "ball_speed_mps",
                     "ball_outward_mps", "ball_goal_distance_m", "local_sideline_count",
                     "local_sideline_speed_mps", "nearest_ball_player_m"]
    context = np.column_stack([extra[k] for k in context_names] + [updates["mean_speed"], updates["frac_still"]])
    context_names += ["global_speed_mps", "global_still_fraction"]
    for seconds in (1, 3, 10):
        before = _directional_mean(context, seconds*fps)
        after = _directional_mean(context, seconds*fps, future=True)
        for j, name in enumerate(context_names):
            extra[f"{name}_past{seconds}s"] = before[:, j]
            extra[f"{name}_future{seconds}s"] = after[:, j]
            # Separate signed change can represent slowing before a stoppage or
            # accelerating at a fast restart, without extending every D run.
            extra[f"{name}_change{seconds}s"] = after[:, j]-before[:, j]
    columns.append(np.column_stack(list(extra.values())))
    output_names += list(extra)
    return output_names, np.concatenate(columns, axis=1).astype(np.float32)


def temporal_features(match):
    """Deployable tracking only, <=10.2 s future context, no memory across halves.

    All spatial features are invariant under team swapping and pitch reflection.
    Player motion uses the existing physically gated anonymous-set matching.
    Events, match ID, absolute clock, targets and ball height are never inputs.
    """
    result, names = [], None
    for a, b in tracking_segments(match):
        names, values = _temporal_one(match.ball[a:b], match.players[a:b], match.teams, match.pitch, match.fps)
        if match.period[a] == 0:
            values[:] = np.nan
        result.append(values)
    return names, np.concatenate(result)


def feature_schema(match, schema=TEMPORAL_SCHEMA):
    """(names, T x F matrix) for the named schema; the model file records which one it was trained on."""
    if schema != TEMPORAL_SCHEMA:
        raise ValueError(f"Unknown football feature schema: {schema}")
    return temporal_features(match)
