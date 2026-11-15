"""Separate live/dead and candidate-team estimators, with exact team symmetry."""
import numpy as np
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
from .features import live_matrix, team_matrix


def probabilities(model, g, t, batch_size=12000):
    live, home = np.empty(len(g)), np.empty(len(g))
    with threadpool_limits(limits=2):
        for a in range(0, len(g), batch_size):
            b = min(len(g), a + batch_size)
            live[a:b] = model["live"].predict_proba(live_matrix(g[a:b], t[a:b]))[:, 1]
            q = []
            for side in (0, 1):
                p = model["owner"].predict_proba(team_matrix(g[a:b], t[a:b], side))[:, 1]
                q.append(logit(np.clip(p, 1e-6, 1-1e-6)))
            home[a:b] = expit((q[0] - q[1]) / 2)
    return live, home
