"""Local-only input parsing. Exceptions deliberately contain no restricted rows."""
from __future__ import annotations
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .data import Match
from .epts import coordinates, frame_number, resolve, split_fields

N_FRAMES = 199319
FPS = 25
# Published evaluation coverage metadata. No labels or inferred boundaries are encoded here.
PERIODS = ((11185, 90652), (110808, 190198))


def read_atd(meta_path: Path, raw_path: Path, n_frames=N_FRAMES, pitch=(105.3, 68.0),
             intervals=PERIODS):
    root = ET.parse(meta_path).getroot()
    slot_team, layouts = resolve(root, n_frames)
    ball = np.full((n_frames, 2), np.nan, np.float32)
    players = np.full((n_frames, len(slot_team), 2), np.nan, np.float32)
    count = 0
    layout_index = 0
    with raw_path.open() as stream:
        for count, line in enumerate(stream, 1):
            if count > n_frames:
                raise ValueError("ATD has too many frames")
            parts = split_fields(line.strip(), ":", 3)
            if frame_number(parts[0]) != count:
                raise ValueError("ATD frames must be contiguous, ordered and one-based")
            while count > layouts[layout_index].last:
                layout_index += 1
            layout = layouts[layout_index]
            if not layout.first <= count <= layout.last:
                raise ValueError("ATD frame is outside its declared format interval")
            groups = split_fields(parts[1], ";", len(layout.players))
            for group, (slot, width, x, y) in zip(groups, layout.players):
                players[count-1, slot] = coordinates(group, (width, x, y))
            ball[count-1] = coordinates(parts[2], layout.ball)
    if count != n_frames:
        raise ValueError("ATD frame count does not match tactical reference")
    scale = np.array(pitch, np.float32)
    ball = (ball - .5) * scale
    players = (players - .5) * scale
    period = np.zeros(n_frames, np.int8)
    for number, (a, b) in enumerate(intervals, 1):
        if not 0 <= a < b <= n_frames:
            raise ValueError("Invalid public period metadata")
        period[a:b] = number
    return Match("BAR-ATH", ball, players, slot_team, period, pitch=pitch, source="challenge").validate()
