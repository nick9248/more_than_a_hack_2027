"""Rebuild models/release.json after an accepted, validated change (see Persian Lions.md)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from persian_lion.release import MANIFEST_NAME, PACKAGE_REVISION, ROOT, release_files, sha256, verify_runtime

old = json.loads((ROOT/MANIFEST_NAME).read_text())
manifest = dict(old)
manifest["files"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in release_files(ROOT)}
manifest["package_revision"] = PACKAGE_REVISION
manifest["runtime"] = verify_runtime(ROOT)
manifest["policy"] = (old["policy"].split(" Version 1.1")[0] + " Version 1.1 adds a football dead-ball model (player and ball features, fitted on external "
                      "IDSSE matches, temporal 205-feature schema, trained with hidden-ball stretches) blended in log-odds (football weight 0.9, chosen on external held-out matches) "
                      "with the tree live probability.")
# newline="\n": the manifest must be byte-identical on Windows and Linux
with open(ROOT/MANIFEST_NAME, "w", newline="\n") as handle:
    handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print("manifest written:", len(manifest["files"]), "files")
