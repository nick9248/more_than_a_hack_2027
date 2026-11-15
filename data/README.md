# Match data goes outside this repository

The match assets are licensed for the competition only and must not be committed or
redistributed. This repository is public, so the data is kept in a separate folder and found
through the `MTAH_DATA_DIR` environment variable (read by `games.py`).

Expected layout inside that folder (names exactly as the bundle arrives, videos renamed):

```
<MTAH_DATA_DIR>/
├─ tactical/
│  ├─ BAR-ATH_tactical.mp4
│  ├─ BAR-ATH_tactical_atd.xml
│  ├─ BAR-ATH_tactical_atd.txt
│  ├─ BAR-ATH_tactical_homography.csv
│  └─ BAR-ATH_tactical_smart_tagging.xml
└─ broadcast/
   ├─ BAR-ATH_broadcast.mp4
   ├─ BAR-ATH_broadcast_atd.xml
   ├─ BAR-ATH_broadcast_atd.txt
   └─ BAR-ATH_broadcast_homography.csv
```

Set it for the current PowerShell session:

```powershell
$env:MTAH_DATA_DIR = "C:\path\to\mtah2027-data"
```

If the variable is not set, `games.py` falls back to this `data/` folder. Everything in it except
this README is gitignored. See the
[official DATA.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/DATA.md)
for what each file contains.
