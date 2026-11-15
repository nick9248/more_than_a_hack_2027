"""Choose the weight of the dead-ball model in the live/dead blend (football_blend in models/parameters.json).

    python external/sweep_blend_weight.py

For each held-out external match, two conditions: corrupted players + realistic ball gaps ("tracker-like")
and clean players + realistic ball gaps. For every weight the blended live probability goes through the
release decoder and the official scorer. The tree/network outputs are cached in MTAH_EXTERNAL_DIR/cache.
"""
import numpy as np
import joblib
from scipy.special import expit, logit

from common import EXTERNAL_DIR, ROOT, VAL, load_match
from augment import corrupt, realistic_gaps
import persian_lion.predict as P
from persian_lion import boundary, football
from persian_lion.decode import decode
from persian_lion.features import extract
from persian_lion.temporal import probabilities as temporal_probabilities
from persian_lion.tree import probabilities as tree_probabilities
import sys
sys.path.insert(0, str(ROOT / "evaluation"))
import possession_metrics as pm

WEIGHTS = (0.0, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0)
CACHE = EXTERNAL_DIR / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def tree_and_network_outputs(release, match):
    """Live probability, home probability and boundary evidence, exactly as predict_match computes them."""
    g, t = extract(match, use_motion=True)
    live, home = tree_probabilities(release["tree"], g, t)
    bp = boundary.probabilities(release["tree"]["boundary"], boundary.features(g, t, match.period))
    parameters = release["parameters"]
    _, neural_home, _ = temporal_probabilities(release["neural"], dict(g=g, t=t), match.period, device="cpu",
                                               chunk_steps=parameters["runtime"]["chunk_steps"])
    mix = parameters["neural_owner_mix"]
    return live, (1 - mix) * home + mix * neural_home, bp


def score(match, labels):
    truth = match.labels.copy()
    truth[int(np.flatnonzero(match.period == 2)[0])] = 3     # keep the two halves apart for the scorer
    r = pm.evaluate(truth, labels.astype(np.int8), bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"]


def logit_clipped(p):
    return logit(np.clip(p, 1e-5, 1 - 1e-5))


if __name__ == "__main__":
    release = P.load_release(ROOT)
    decoder = release["parameters"]["decoder"]
    artifact = joblib.load(ROOT / "models/deadball.joblib")
    results = {}
    for mid in VAL:
        base = load_match(mid)
        seed = sum(map(ord, mid))
        with_gaps = realistic_gaps(base, seed + 7)
        for condition, match in (("tracker-like", corrupt(with_gaps, seed)), ("clean players", with_gaps)):
            cache = CACHE / f"{mid}_{condition.split()[0]}.npz"
            if cache.exists():
                z = np.load(cache)
                live, home, bp = z["live"], z["home"], z["bp"]
            else:
                live, home, bp = tree_and_network_outputs(release, match.without_labels())
                np.savez(cache, live=live, home=home, bp=bp)
            p_dead = artifact["model"].predict_proba(
                football.feature_schema(match.without_labels(), artifact["feature_schema"])[1])[:, 1]
            seen = np.isfinite(match.ball).all(axis=1)
            for w in WEIGHTS:
                blended = expit((1 - w) * logit_clipped(live) + w * logit_clipped(1 - p_dead))
                results.setdefault((condition, w), []).append(
                    score(match, decode(blended, home, seen, match.period, bp, **decoder)))
            print("done", mid, condition, flush=True)
    for condition in ("tracker-like", "clean players"):
        print(condition, "+ realistic ball gaps; mean over held-out matches")
        for w in WEIGHTS:
            spell, macro, alive_dead = np.mean(results[(condition, w)], axis=0)
            print("  football weight {:.2f}: spellF1 {:.3f}  macroF1 {:.3f}  alive/dead F1 {:.3f}".format(w, spell, macro, alive_dead))
