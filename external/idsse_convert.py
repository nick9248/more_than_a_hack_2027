"""Convert one IDSSE (Sportec/DFL, CC BY 4.0) match to a compact .npz for training.

    python external/idsse_convert.py <idsse_dir> <match_id> <out_dir>

External data about OTHER matches only (allowed as training and validation data: it is not from the provided game).
Referees are dropped on purpose: the competition tracking has no referee slot.

Output arrays, one row per 25 Hz frame, both halves stacked (period says which):
    players  (T, 22+, 2)  x, y in metres on a 0..105 x 0..68 pitch, NaN when not on the pitch
    team     (22+,)       0 = home, 1 = guest
    ball     (T, 2)       x, y in metres
    status   (T,)         1 = ball alive, 0 = dead (provider BallStatus)
    owner    (T,)         0 = home, 1 = guest (provider BallPossession), -1 unknown
    period   (T,)         1 or 2
"""
import os
import sys
from datetime import datetime
import xml.etree.ElementTree as ET

import numpy as np

PITCH_L, PITCH_W = 105.0, 68.0


def read_match(idsse_dir, match_id):
    info = [f for f in os.listdir(idsse_dir) if f.startswith("DFL_02_01") and match_id in f][0]
    pos = [f for f in os.listdir(idsse_dir) if f.startswith("DFL_04_03") and match_id in f][0]
    root = ET.parse(os.path.join(idsse_dir, info)).getroot()
    role_of_team = {t.get("TeamId"): t.get("Role") for t in root.iter("Team")}

    tracks = {}   # (half, person) -> dict(team, frames{N:(x,y)})
    ball = {}     # half -> {N: (x, y, status, owner)}
    first_time = {}   # half -> (N, epoch seconds) of the first ball frame
    for _, el in ET.iterparse(os.path.join(idsse_dir, pos), events=("end",)):
        if el.tag != "FrameSet":
            continue
        half = 1 if el.get("GameSection") == "firstHalf" else 2
        team_id = el.get("TeamId")
        if team_id == "referee":
            pass
        elif team_id == "BALL":
            ball[half] = {int(f.get("N")): (float(f.get("X")), float(f.get("Y")),
                                            int(f.get("BallStatus")), int(f.get("BallPossession")))
                          for f in el.iter("Frame")}
            f0 = next(iter(el.iter("Frame")))
            first_time[half] = (int(f0.get("N")), datetime.fromisoformat(f0.get("T")).timestamp())
        else:
            tracks[(half, el.get("PersonId"))] = (
                0 if role_of_team[team_id] == "home" else 1,
                {int(f.get("N")): (float(f.get("X")), float(f.get("Y"))) for f in el.iter("Frame")})
        el.clear()

    persons = sorted({p for _, p in tracks})
    team = np.array([next(t for (h, p), (t, _) in tracks.items() if p == person) for person in persons])
    order = np.argsort(team, kind="stable")
    persons = [persons[i] for i in order]
    team = team[order]

    players, balls, status, owner, period = [], [], [], [], []
    for half in (1, 2):
        frames = sorted(ball[half])
        idx = {n: i for i, n in enumerate(frames)}
        block = np.full((len(frames), len(persons), 2), np.nan, dtype=np.float32)
        for j, person in enumerate(persons):
            if (half, person) in tracks:
                for n, (x, y) in tracks[(half, person)][1].items():
                    if n in idx:
                        block[idx[n], j] = (x, y)
        players.append(block)
        balls.append(np.array([ball[half][n][:2] for n in frames], dtype=np.float32))
        status.append(np.array([ball[half][n][2] for n in frames], dtype=np.int8))
        # BallPossession: 1 = home, 2 = guest in the DFL spec
        owner.append(np.array([{1: 0, 2: 1}.get(ball[half][n][3], -1) for n in frames], dtype=np.int8))
        period.append(np.full(len(frames), half, dtype=np.int8))

    players = np.concatenate(players)
    balls = np.concatenate(balls)
    players[:, :, 0] += PITCH_L / 2   # centred -> 0..105
    players[:, :, 1] += PITCH_W / 2
    balls[:, 0] += PITCH_L / 2
    balls[:, 1] += PITCH_W / 2
    return dict(players=players, team=team, ball=balls, status=np.concatenate(status),
                owner=np.concatenate(owner), period=np.concatenate(period),
                t0=np.array([first_time[1][1], first_time[2][1]]))


if __name__ == "__main__":
    idsse_dir, match_id, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    data = read_match(idsse_dir, match_id)
    np.savez_compressed(os.path.join(out_dir, match_id + ".npz"), **data)
    st = data["status"]
    print(match_id, "frames", len(st), "dead share {:.1%}".format(np.mean(st == 0)),
          "slots", data["players"].shape[1])
