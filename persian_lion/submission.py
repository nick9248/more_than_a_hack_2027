"""Exact submission CSV formatting; no prediction contents are logged."""
import csv
from pathlib import Path
import numpy as np
from .reader import N_FRAMES
from .data import LETTERS

def validate_csv(path, expected=N_FRAMES):
    """Stricter than the official validator: no extra columns, spaces or case coercion."""
    with Path(path).open(newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["frame", "label"]:
            raise ValueError("Expected exactly frame,label header")
        count = 0
        for count, row in enumerate(reader, 1):
            if len(row) != 2 or row[0] != str(count-1) or row[1] not in {"H", "A", "D", "X"}:
                raise ValueError("Invalid submission format or frame order")
        if count != expected:
            raise ValueError("Submission row count mismatch")

def write_csv(path, labels):
    path = Path(path)
    if not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError("Invalid prediction codes")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        stream.write("frame,label\n")
        for frame, label in enumerate(LETTERS[labels]):
            stream.write(f"{frame},{label}\n")
