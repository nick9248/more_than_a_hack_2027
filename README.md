# Persian Lions: ball possession from tactical tracking (More than a Hack 2027, Phase 1)

This repository regenerates the team's possession file `Persian Lions.csv` (one label H / A / D / X for each of the 199,319 frames of the tactical video) from the provided tactical tracking. It contains the trained models, the inference code and the scripts that trained the dead-ball model. It contains no match data.

Read [Persian Lions.md](Persian%20Lions.md) first: it is the technical document (approach, assets used and dropped, training data, possession rules, limitations, use of AI tools, reproduction). [TESTING_REFERENCE.md](TESTING_REFERENCE.md) holds the detailed evidence.

## Layout

```text
persian_lion/                tracking reader, features, models, decoder, command line
models/                      trees, temporal network, dead-ball model, parameters, integrity manifest
starter/predict_possession.py   run the model (organisers' starter scripts are kept beside it)
evaluation/                  the organisers' scorer and validator (no labels included)
external/                    scripts that trained the dead-ball model and chose its blend weight
Persian Lions.md             technical document (Persian Lions.pdf is the same text as a PDF)
TESTING_REFERENCE.md         testing and provenance record
THIRD_PARTY_NOTICES.md       licences and attribution of the external training data
requirements*.txt, pyproject.toml   pinned dependencies
```

## Regenerate the submission file

Python 3.12 is required (the release checks it). Inference runs on the CPU with two threads and needs no GPU, account or download.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m persian_lion verify          # checks code, models and pinned packages
```

On Windows use `py -3.12 -m venv .venv` and `.venv\Scripts\python.exe`.

The provided match bundle stays outside this repository. The model needs only:

```text
<data-dir>/tactical/BAR-ATH_tactical_atd.xml
<data-dir>/tactical/BAR-ATH_tactical_atd.txt
```

Pass `--data-dir` (or set the `MTAH_DATA_DIR` environment variable). Then:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m persian_lion predict --data-dir <data-dir> --output 'Persian Lions.csv'
.venv/bin/python -m persian_lion validate --output 'Persian Lions.csv'
.venv/bin/python evaluation/validate_submission.py 'Persian Lions.csv'
```

In PowerShell set each variable separately (`$env:OMP_NUM_THREADS = '2'`) and drop the inline prefix. The command refuses to overwrite an existing output and prints only fixed status messages (never match contents). It also writes `Persian Lions.manifest.json` with the hashes of input, release and output; this is not a deliverable. Running it twice produces the same bytes.

## Retrain the dead-ball model (optional)

The shipped `models/deadball.joblib` was trained only on external matches (other games, CC BY 4.0). To rebuild it:

1. Download the seven IDSSE matches J03WMX, J03WN1, J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9 (revision and licence in `THIRD_PARTY_NOTICES.md`) and convert each with `external/idsse_convert.py <xml_dir> <match_id> <out_dir>`; put the `.npz` files in `external_data/npz/` (or set `MTAH_EXTERNAL_DIR`).
2. `python external/train_deadball.py models/deadball.joblib` trains on four matches and prints held-out scores on the other three.
3. `python external/sweep_blend_weight.py` reproduces the choice of the blend weight (0.9).
4. `python external/make_manifest.py` rebuilds `models/release.json` after any accepted change.

## Notes

- Do not use the competition match, its public five minutes or any of its labels to tune anything.
- Keep outputs and the match bundle out of this repository: `.gitignore` is a guard, not permission to publish them.
- The runtime was verified on Python 3.12; other platforms were not certified.
