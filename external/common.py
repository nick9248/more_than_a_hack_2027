"""Shared helpers for the external (IDSSE) experiments: converted .npz -> Match.

The converted matches live outside the repository. Point MTAH_EXTERNAL_DIR at a folder that holds
an `npz/` sub-folder with one <match_id>.npz per match (see idsse_convert.py); the default is
`external_data/` in the repository root, which Git ignores.
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from persian_lion.data import Match  # noqa: E402

EXTERNAL_DIR = Path(os.environ.get("MTAH_EXTERNAL_DIR", ROOT / "external_data"))
NPZ_DIR = EXTERNAL_DIR / "npz"
FIT = ("J03WMX", "J03WN1", "J03WOH", "J03WOY")     # fit the dead-ball model
VAL = ("J03WPY", "J03WQQ", "J03WR9")               # held out: evaluation and blend-weight choice


def load_match(match_id):
    z = np.load(NPZ_DIR / (match_id + ".npz"))
    centre = np.array([52.5, 34.0], np.float32)
    owner, status = z["owner"], z["status"]
    labels = np.where(status == 0, 2, np.where(owner >= 0, owner, 3)).astype(np.int8)  # 0 H, 1 A, 2 D, 3 X
    return Match(match_id, z["ball"] - centre, z["players"] - centre, z["team"].astype(np.int8),
                 z["period"].astype(np.int8), labels=labels, source="idsse").validate()
