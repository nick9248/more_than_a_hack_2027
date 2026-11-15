"""Label-free inference using only the selected version 1 artifacts."""
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import expit, logit

from . import boundary, football
from .decode import decode
from .features import extract
from .model import PossessionNet
from .release import ROOT, verify, verify_runtime
from .temporal import probabilities as temporal_probabilities
from .tree import probabilities as tree_probabilities


def load_release(root=ROOT, require_frozen=True):
    root = Path(root)
    verify(root) if require_frozen else verify_runtime(root)
    parameters = json.loads((root/"models/parameters.json").read_text())
    if parameters.get("schema_version") != 1 or parameters.get("runtime") != dict(device="cpu", threads=2, chunk_steps=2048):
        raise ValueError("Unexpected reference parameters")
    torch.set_num_threads(2)
    tree = joblib.load(root/"models/model.joblib")
    checkpoint = torch.load(root/"models/temporal.pt", map_location="cpu", weights_only=True)
    if tree.get("schema_version") != 1 or checkpoint.get("schema_version") != 1:
        raise ValueError("Unexpected model schema")
    neural = PossessionNet(**checkpoint["architecture"])
    neural.load_state_dict(checkpoint["state_dict"], strict=True)
    neural.eval()
    deadball = joblib.load(root/"models/deadball.joblib")
    return dict(tree=tree, neural=neural, deadball=deadball["model"],
                deadball_schema=deadball["feature_schema"], parameters=parameters)


def predict_match(loaded, match):
    """Labels are discarded before any feature extraction or inference."""
    match = match.without_labels()
    g, t = extract(match, use_motion=True)
    live, home = tree_probabilities(loaded["tree"], g, t)
    weight = loaded["parameters"]["football_blend"]
    if weight:
        # Football dead-ball model (player + ball features, fitted on external matches): log-odds blend.
        _, football_features = football.feature_schema(match, loaded["deadball_schema"])
        p_dead = loaded["deadball"].predict_proba(football_features)[:, 1]
        live = expit((1-weight)*logit(np.clip(live, 1e-5, 1-1e-5)) + weight*logit(np.clip(1-p_dead, 1e-5, 1-1e-5)))
    bp = boundary.probabilities(loaded["tree"]["boundary"], boundary.features(g, t, match.period))
    parameters = loaded["parameters"]
    mix = parameters["neural_owner_mix"]
    if not 0 <= mix <= 1:
        raise ValueError("Invalid selected owner mixture")
    if mix:
        _, neural_home, _ = temporal_probabilities(loaded["neural"], dict(g=g, t=t), match.period,
                                                   device="cpu", chunk_steps=parameters["runtime"]["chunk_steps"])
        home = (1-mix)*home + mix*neural_home
    return decode(live, home, np.isfinite(match.ball).all(axis=1), match.period, bp, **parameters["decoder"])
