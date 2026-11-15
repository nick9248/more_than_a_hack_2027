"""A provider-independent array schema; labels never enter feature extraction."""
from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np

H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])


@dataclass
class Match:
    match_id: str
    ball: np.ndarray                 # T x 2, centred metres; NaN means unobserved
    players: np.ndarray              # T x slots x 2; slot is NOT an identity
    teams: np.ndarray                # slots, 0=home, 1=away
    period: np.ndarray               # T, 0=outside play periods, 1/2=halves
    pitch: tuple[float, float] = (105.0, 68.0)
    fps: int = 25
    labels: np.ndarray | None = None # training targets only, never a feature
    source: str = "synthetic"

    def validate(self):
        n = len(self.ball)
        if self.ball.shape != (n, 2) or self.players.shape != (n, len(self.teams), 2):
            raise ValueError("Invalid tracking array shape")
        if self.period.shape != (n,) or self.fps != 25 or n == 0:
            raise ValueError("Expected nonempty 25 fps match with aligned periods")
        if not np.isin(self.teams, [H, A]).all() or set(self.teams) != {H, A}:
            raise ValueError("Expected both teams, with explicit membership")
        if not np.isfinite(self.pitch).all() or min(self.pitch) <= 0:
            raise ValueError("Invalid pitch dimensions")
        if np.isinf(self.ball).any() or np.isinf(self.players).any():
            raise ValueError("Infinite coordinates are not missing observations")
        if self.labels is not None and (self.labels.shape != (n,) or not np.isin(self.labels, [H,A,D,X]).all()):
            raise ValueError("Invalid labels")
        return self

    def without_labels(self):
        return replace(self, labels=None)


def blocks(period):
    """Contiguous period ranges. No smoothing or memory crosses a half boundary."""
    cuts = np.r_[0, np.flatnonzero(np.diff(period) != 0) + 1, len(period)]
    return [(int(a), int(b)) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
