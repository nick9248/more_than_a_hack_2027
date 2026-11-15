"""Train the dead-ball classifier on external (IDSSE) matches and test it on held-out ones.

    python external/train_deadball.py models/deadball.joblib

Target: the provider's BallStatus (dead = 1). Inputs: persian_lion.football features (players and ball).
Training: the four FIT matches, each used three times: as tracked, with the ball hidden in random
stretches, and with corrupted players plus hidden-ball stretches (see augment.py). Rows are sampled
every 5th frame. The result is evaluated on the three VAL matches, clean and corrupted with gaps.
"""
import sys
import time

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score

from common import FIT, VAL, load_match
from augment import corrupt, make_gaps
from persian_lion import football

STRIDE = 5
SEED = 20261113


def training_set(match_ids):
    xs, ys, names = [], [], None
    for mid in match_ids:
        base = load_match(mid)
        for match in (base, make_gaps(base, 101), make_gaps(corrupt(base, 100), 102)):
            names, features = football.feature_schema(match.without_labels())
            rows = np.flatnonzero(match.period > 0)[::STRIDE]      # only frames inside the two halves
            xs.append(features[rows])
            ys.append((match.labels[rows] == 2).astype(np.int8))
        print("features", mid, flush=True)
    return names, np.concatenate(xs), np.concatenate(ys)


def report(mid, condition, dead, p_dead):
    predicted = p_dead > 0.5
    print("{} {:18} AUC {:.4f}  dead-F1 {:.4f}  false-D on live frames {:.4f}".format(
        mid, condition, roc_auc_score(dead, p_dead), f1_score(dead, predicted), predicted[~dead].mean()), flush=True)


if __name__ == "__main__":
    start = time.time()
    names, x, y = training_set(FIT)
    print("training rows", x.shape, "dead share {:.3f}".format(y.mean()), flush=True)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                           min_samples_leaf=60, l2_regularization=4.0, random_state=SEED)
    model.fit(x, y)
    print("trained in", round(time.time() - start), "s", flush=True)
    for mid in VAL:
        base = load_match(mid)
        for condition, match in (("clean", base), ("corrupted+gaps", make_gaps(corrupt(base, 7), 8))):
            inside = match.period > 0
            _, features = football.feature_schema(match.without_labels())
            report(mid, condition, (match.labels == 2)[inside], model.predict_proba(features[inside])[:, 1])
    joblib.dump(dict(model=model, names=names, feature_schema=football.TEMPORAL_SCHEMA), sys.argv[1])
