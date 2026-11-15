"""Local inference CLI with fixed status messages and no data-bearing output."""
import argparse
import os
from pathlib import Path
import sys
import warnings

import numpy as np

from .predict import load_release, predict_match
from .reader import read_atd
from .release import ROOT, MANIFEST_NAME, verify, sha256, write_json
from .submission import write_csv, validate_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("predict", "validate", "verify"))
    local_data = os.environ.get("MTAH_DATA_DIR")
    parser.add_argument("--data-dir", type=Path, default=Path(local_data) if local_data else None,
                        help="Local authorized bundle; defaults to MTAH_DATA_DIR when set")
    parser.add_argument("--output", type=Path, default=ROOT/"evaluation/results/Persian Lions.csv")
    args = parser.parse_args(argv)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if args.command == "verify":
                verify(ROOT)
            elif args.command == "validate":
                validate_csv(args.output)
            else:
                if args.data_dir is None:
                    raise ValueError("Local input directory is required")
                loaded = load_release(ROOT)
                manifest_path = args.output.with_suffix(".manifest.json")
                if args.output.exists() or manifest_path.exists():
                    raise FileExistsError("Existing output is protected")
                tactical = args.data_dir/"tactical"
                meta, raw = tactical/"BAR-ATH_tactical_atd.xml", tactical/"BAR-ATH_tactical_atd.txt"
                match = read_atd(meta, raw)
                labels = predict_match(loaded, match)
                if np.any(labels[match.period > 0] == 3):
                    raise ValueError("Incomplete active-half prediction")
                write_csv(args.output, labels)
                validate_csv(args.output)
                write_json(manifest_path, dict(valid=True, csv_sha256=sha256(args.output),
                    release_sha256=sha256(ROOT/MANIFEST_NAME),
                    input_sha256={"atd_metadata": sha256(meta), "atd_tracking": sha256(raw)}))
    except Exception:
        print("FAILED: local input, output, runtime or release validation. No match contents displayed.", file=sys.stderr)
        return 1
    print("PASS: "+args.command+". No match contents displayed.")
    return 0
