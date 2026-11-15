"""Make clean external (IDSSE) tracking resemble imperfect tracking. Used for training and robustness tests.

The settings below were chosen to resemble AGGREGATE, unlabelled statistics of the provided tracking
(about 20.5 of 22 players visible per frame, 9% of frames with 17 or fewer, a few stray tracks, noisy
positions, the ball hidden for roughly 40% of the time). No frame-specific rule and no label of the
provided game is used. Every generator here is seeded and independent of the match labels, except
`realistic_gaps`, which is for evaluation only.

  corrupt         players vanish in blocks and in bursts, stray tracks appear, Gaussian noise
  make_gaps       the ball is hidden in random stretches (training and evaluation)
  realistic_gaps  evaluation only: also hides the ball in most long dead-ball stretches (uses labels)
"""
from dataclasses import replace

import numpy as np


def corrupt(match, seed):
    """Players: blocks gone (mean 3 s, present mean 60 s), bursts of 4-8 players, stray tracks, noise 0.3 m."""
    rng = np.random.default_rng(seed)
    p = match.players.copy()
    T, S, _ = p.shape
    for s in range(S):
        t, present = 0, True
        while t < T:
            n = int(rng.exponential(1500 if present else 75)) + 5
            if not present:
                p[t:t + n, s] = np.nan
            t += n
            present = not present
    t = int(rng.exponential(1500))          # bursts: several players vanish together for a few seconds
    while t < T:
        n = int(rng.exponential(90)) + 25
        p[t:t + n, rng.choice(S, size=int(rng.integers(4, 9)), replace=False)] = np.nan
        t += n + int(rng.exponential(900))
    spare = np.full((T, 4, 2), np.nan, np.float32)      # 4 extra slots for stationary stray tracks
    for s in range(4):
        t = int(rng.exponential(1500))
        while t < T:
            n = int(rng.exponential(250)) + 50
            spare[t:t + n, s] = (rng.uniform(-52, 52), rng.uniform(-34, 34))
            t += n + int(rng.exponential(8000))
    p = np.concatenate([p, spare], axis=1)
    p += rng.normal(0, 0.3, p.shape).astype(np.float32)
    teams = np.concatenate([match.teams, np.array([0, 1, 0, 1], np.int8)])
    ball = match.ball + rng.normal(0, 0.15, match.ball.shape).astype(np.float32)
    return replace(match, players=p.astype(np.float32), teams=teams, ball=ball)


def make_gaps(match, seed, mean_gap=375, mean_seen=500):
    """Alternate seen / hidden ball stretches (exponential lengths; hidden at least 5 s, about 43% of frames)."""
    rng = np.random.default_rng(seed)
    ball = match.ball.copy()
    t, seen = 0, True
    while t < len(ball):
        length = int(rng.exponential(mean_seen if seen else mean_gap)) + (25 if seen else 125)
        if not seen:
            ball[t:t + length] = np.nan
        t += length
        seen = not seen
    return replace(match, ball=ball)


def realistic_gaps(match, seed):
    """Evaluation only. Hides the ball in 80% of dead-ball stretches longer than 8 s, plus short gaps in play."""
    rng = np.random.default_rng(seed)
    ball = match.ball.copy()
    dead = match.labels == 2
    edges = np.r_[0, np.flatnonzero(np.diff(dead.astype(np.int8))) + 1, len(dead)]
    for a, b in zip(edges[:-1], edges[1:]):
        if dead[a] and b - a > 200 and rng.random() < 0.8:
            ball[a:b] = np.nan
    t = 0
    while t < len(ball):
        t += int(rng.exponential(1500)) + 25            # visible stretch
        n = int(rng.exponential(150)) + 25               # short gap in play (about 6 s)
        ball[t:t + n] = np.nan
        t += n
    return replace(match, ball=ball)
