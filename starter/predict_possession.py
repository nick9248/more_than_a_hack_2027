"""Run the Persian Lion model; MTAH_DATA_DIR can supply the local input folder.

    python starter/predict_possession.py --data-dir /local/licensed-match

The existing baseline and data loaders in this directory remain unchanged.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persian_lion.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(["predict", *sys.argv[1:]]))
