"""Predict transitions from anonymous tracking geometry; targets stay outside features."""
import numpy as np
from threadpoolctl import threadpool_limits
from .data import blocks
from .features import shift

FEATURE_VERSION = "boundary_geometry_1"


def features(g, t, period):
    results = []
    for a, b in blocks(period):
        local = t[a:b, :, :14]
        parts = [g[a:b], (local[:, 0] + local[:, 1]) / 2,
                 np.abs(local[:, 0] - local[:, 1])]
        for lag in (5, 12, 25, 50):
            before, after = shift(local, lag), shift(local, -lag)
            change = after - before
            parts.extend([(change[:, 0] + change[:, 1]) / 2,
                          np.abs(change[:, 0] - change[:, 1])])
            margin_before = before[:, 1, 0] - before[:, 0, 0]
            margin_after = after[:, 1, 0] - after[:, 0, 0]
            parts.append(np.column_stack([np.abs(margin_before), np.abs(margin_after),
                margin_before * margin_after, np.abs(margin_after - margin_before),
                np.sign(margin_before) * np.sign(margin_after)]))
            parts.append(shift(g[a:b, :9], -lag) - shift(g[a:b, :9], lag))
        results.append(np.column_stack(parts).astype(np.float32))
    result = np.concatenate(results)
    if np.isinf(result).any():
        raise ValueError("Infinite boundary feature")
    return result


def probabilities(model, x):
    assert model["feature_version"] == FEATURE_VERSION
    out = np.empty((len(x), 3), np.float32)
    with threadpool_limits(limits=2):
        for a in range(0, len(x), 12000):
            b = min(a+12000, len(x))
            for j, head in enumerate(model["heads"]):
                out[a:b, j] = head.predict_proba(x[a:b])[:, 1]
    return out
