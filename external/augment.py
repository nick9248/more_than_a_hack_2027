"""Make clean external (IDSSE) tracking look like imperfect tracking. Used for training and for robustness tests.

Two fixed severity levels, written as round numbers; they are generic and are not fitted to any
particular tracker or match. Every generator is seeded and independent of the match labels.

  corrupt(match, seed, level)   players vanish in blocks and in bursts, stray tracks appear, Gaussian noise
  make_gaps(match, seed, level) the ball is hidden in random stretches (at least 5 s each)

level            players gone: present / gone (mean s)   burst of 4-8 players   noise (m): players / ball   ball hidden
"moderate"       120 s / 2 s                              every ~110 s, 2.4 s    0.2 / 0.10                  about 33%
"severe"         40 s / 4 s                               every ~25 s, 4.8 s     0.4 / 0.20                  about 57%
"""
from dataclasses import replace

import numpy as np

FPS = 25
LEVELS = {
    "moderate": dict(present=3000, gone=50, burst_every=2700, burst_len=60, noise=0.2, ball_noise=0.10,
                     ball_seen=750, ball_hidden=250),
    "severe": dict(present=1000, gone=100, burst_every=600, burst_len=120, noise=0.4, ball_noise=0.20,
                   ball_seen=400, ball_hidden=450),
}


def corrupt(match, seed, level):
    """Players: blocks gone, bursts of 4-8 players vanishing together, 4 stationary stray tracks, noise."""
    c = LEVELS[level]
    rng = np.random.default_rng(seed)
    p = match.players.copy()
    T, S, _ = p.shape
    for s in range(S):
        t, present = 0, True
        while t < T:
            n = int(rng.exponential(c["present"] if present else c["gone"])) + 5
            if not present:
                p[t:t + n, s] = np.nan
            t += n
            present = not present
    t = int(rng.exponential(c["burst_every"]))
    while t < T:
        n = int(rng.exponential(c["burst_len"])) + 25
        p[t:t + n, rng.choice(S, size=int(rng.integers(4, 9)), replace=False)] = np.nan
        t += n + int(rng.exponential(c["burst_every"]))
    spare = np.full((T, 4, 2), np.nan, np.float32)      # 4 extra slots for stationary stray tracks
    for s in range(4):
        t = int(rng.exponential(1500))
        while t < T:
            n = int(rng.exponential(250)) + 50
            spare[t:t + n, s] = (rng.uniform(-52, 52), rng.uniform(-34, 34))
            t += n + int(rng.exponential(8000))
    p = np.concatenate([p, spare], axis=1)
    p += rng.normal(0, c["noise"], p.shape).astype(np.float32)
    teams = np.concatenate([match.teams, np.array([0, 1, 0, 1], np.int8)])
    ball = match.ball + rng.normal(0, c["ball_noise"], match.ball.shape).astype(np.float32)
    return replace(match, players=p.astype(np.float32), teams=teams, ball=ball)


def make_gaps(match, seed, level):
    """Alternate seen / hidden ball stretches (exponential lengths; hidden at least 5 s)."""
    c = LEVELS[level]
    rng = np.random.default_rng(seed)
    ball = match.ball.copy()
    t, seen = 0, True
    while t < len(ball):
        length = int(rng.exponential(c["ball_seen"] if seen else c["ball_hidden"])) + (25 if seen else 125)
        if not seen:
            ball[t:t + length] = np.nan
        t += length
        seen = not seen
    return replace(match, ball=ball)


def degraded(match, seed, level):
    """Players corrupted and the ball hidden in stretches, at one severity level."""
    return make_gaps(corrupt(match, seed, level), seed + 1, level)
