"""Verify the inference handoff's source, selected weights and pinned runtime."""
from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "models/release.json"
PACKAGE_REVISION = "football-temporal-1"
REQUIRED_FILES = (
    "models/model.joblib", "models/temporal.pt", "models/parameters.json", "models/deadball.joblib",
    ".gitattributes",
    "requirements-lock.txt", "pyproject.toml", "README.md", "Persian Lions.md",
    "TESTING_REFERENCE.md", ".gitignore", "LICENSE", "THIRD_PARTY_NOTICES.md",
    "games.py", "evaluation/possession_metrics.py", "evaluation/validate_submission.py",
    "requirements.txt", "requirements-starter.txt", "data/README.md",
    "evaluation/run_evaluation.py", "evaluation/results/.gitkeep", "evaluation/gt/README.md",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def verify_runtime(root=ROOT):
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Reference inference requires Python 3.12")
    runtime = {}
    for raw in (Path(root)/"requirements-lock.txt").read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            package, expected = line.split("==")
            if version(package) != expected:
                raise ValueError("Reference package version mismatch")
            runtime[package] = expected
    return runtime


def release_files(root=ROOT):
    root = Path(root)
    paths = [root/name for name in REQUIRED_FILES]
    paths += sorted((root/"persian_lion").glob("*.py"))
    paths += sorted((root/"starter").glob("*.py"))
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("An inference release component is missing or linked")
    return sorted(set(paths))


def verify(root=ROOT):
    root = Path(root).resolve()
    runtime = verify_runtime(root)
    manifest = json.loads((root/MANIFEST_NAME).read_text())
    if (manifest.get("schema_version") != 1 or manifest.get("release_version") != "1.0.0"
            or manifest.get("package_revision") != PACKAGE_REVISION):
        raise ValueError("Unexpected release manifest version")
    if manifest["runtime"] != runtime:
        raise ValueError("Release runtime changed")
    if not set(REQUIRED_FILES).issubset(manifest["files"]):
        raise ValueError("Release manifest omits required components")
    for name, digest in manifest["files"].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or sha256(path) != digest:
            raise ValueError("Current release component changed")
    if {path.relative_to(root).as_posix() for path in release_files(root)} != set(manifest["files"]):
        raise ValueError("Release inventory changed")
    return manifest
