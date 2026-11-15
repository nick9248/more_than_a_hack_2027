# Persian Lions: testing and provenance reference

Evidence consolidated on 7 November 2026 for the single version 1 inference release. This document preserves the findings from development and testing; the handoff does not include downloaded datasets, obsolete weights or the historical experiment/test harnesses.

**The model has a material missing-ball weakness.** On a previously unused Barcelona–Real Madrid match, agreement with available automatic owner references was **79.63% overall, 86.03% when the ball was observed and 46.42% when it was missing**. The model produces complete outputs and has reproducible inference, but these facts do not establish competition accuracy or a winning rank.

## Data, permissions and separation

All retained estimators and normalization statistics were fitted on four external IDSSE matches. Competition assets were not inputs to the training, selection or evaluations reported here. Provider possession/event annotations were used only as training targets or evaluation references where explicitly stated, never as inference features.

| Source and permission | Role and exact split |
|---|---|
| **IDSSE/Sportec DFL**, Bassek, Rein, Weber and Memmert (2025), **CC BY 4.0**. [Dataset paper](https://doi.org/10.1038/s41597-025-04505-y), [original data](https://doi.org/10.6084/m9.figshare.28196177), [PySport mirror](https://huggingface.co/datasets/pysport/idsse-data). | Fit: **J03WMX, J03WN1, J03WOH, J03WOY**. Numerical validation: **J03WPY, J03WQQ**. Repeated diagnostic: **J03WR9**. |
| **SkillCorner modern open data**, copyright © 2020 SkillCorner, **MIT**. [Source and licence](https://github.com/SkillCorner/opendata/tree/4340d274572876239c154c90bc507a9b3250a656). | Validation: **1874553, 1886347, 1899585, 1925299**. Diagnostics: **2007448, 2007721, 2010085, 2011166, 2013725, 2015213, 2016236, 2017461**. No retained component fitted on these matches. Credit SkillCorner/PySport. |
| **Metrica Sports public sample data**, [README community-analysis permission](https://github.com/metrica-sports/sample-data/blob/e706dd506b360d69d9d123d5b8026e7294b13996/README.md), responsible use and acknowledgement. No named standard data licence asserted. | Games **1 and 2**: reused partial-event diagnostics. Game **3**: acquired and evaluated after the final model choice. These are public sample games, not the competition match. |
| **SkillCorner original nine-match release**, **MIT**, credit SkillCorner/Friends of Tracking. [Pinned source](https://github.com/SkillCorner/opendata/tree/1b99d401ddc82e366c5c20d6860878668fd5858f). | **2841**, Barcelona–Real Madrid, Camp Nou, **18 December 2019**. Acquired after model freeze; selected before inference as an unused La Liga match with the same home club. One declared condition, no fitting or tuning. |

Source revisions: IDSSE mirror `a715a38dfbaf5f58e431727c2b78d174101a703c`; modern SkillCorner `4340d274572876239c154c90bc507a9b3250a656`; Metrica `e706dd506b360d69d9d123d5b8026e7294b13996`; legacy SkillCorner `1b99d401ddc82e366c5c20d6860878668fd5858f`.

The dense validation matches were reused during development. Earlier diagnostic outcomes informed architecture and safeguards. The four provider validation games had also trained discarded weak-supervision experiments, although no retained component or eligible final neural candidate was fitted on them. Consequently, these validation and repeated diagnostic results are **development-exposed**, not blind estimates. Only the first evaluations of Metrica game 3 and legacy match 2841 were fresh after the final model choice; they too are now exposed diagnostics.

## Reference semantics and alignment

IDSSE supplies 25 Hz positions and Sportec/DFL possession/live-status fields. Targets came from the released `BallStatus` and `BallPossession` fields, not labels newly authored for this project. `BallStatus=0` maps to D; live status with owner 1/2 maps to H/A; unknown targets remain X. Coordinates were converted to centred metric pitch coordinates. Ball height and annotated events were excluded from features. Source possession conventions can differ from the competition's last-controlling-team rule.

Modern SkillCorner was evaluated both with detected positions only and with supplied extrapolated positions. Ten-Hz positions were interpolated to 25 Hz through a common 50 Hz lattice, separately within each half and only between valid adjacent samples. Automatic phase intervals used `[start,end)`; phase gaps formed the declared dead-ball proxy. These labels are provider-generated references, not independent human possession truth.

Metrica checks cover selected pass intervals, exact pass/shot action onsets and stoppage interiors trimmed by one second at both ends. Pass references require named recipients and exclude competing control. Other intervals remain ungraded. These checks cannot provide dense full-match accuracy or spellF1.

For legacy match 2841, owner references came from the possession object's roster team, falling back to its explicit group; both had to agree when present. Missing ownership was excluded, **not converted to D**. References were compared only at exact common 5 Hz native timestamps. Ten-to-25 Hz interpolation never crossed halves, missing endpoints or a change in anonymous track identity. Inference received positions, team membership and timing, with owner fields removed. The adapter used 36 roster slots plus five anonymous slots per team, 46 slots in total.

## Fitting and model selection

The four dense fitting matches each supplied clean, moderate and severe views: **12 fitting views**, not 12 independent matches. Corruption was label-independent, reset at halves, and seeded by `20261105 + CRC32(match_id + "/" + scenario)` over UTF-8 bytes.

The original 25 Hz fitting inputs totaled **567,278 frames** before the three views. Match dates below come from the public source metadata. Counts include both active half timelines; source-unknown targets were excluded from supervised losses.

| Fitting match | Source match date, UTC | Native frames |
|---|---|---:|
| J03WMX | 27 May 2023 | 145,967 |
| J03WN1 | 27 May 2023 | 141,561 |
| J03WOH | 26 August 2022 | 137,214 |
| J03WOY | 10 September 2022 | 142,536 |

| View | Ball missing fraction | Mean missing gap | Ball noise SD | Player missing fraction |
|---|---:|---:|---:|---:|
| Clean | 0 | 0 s | 0 m | 0 |
| Moderate | 0.35 | 1.2 s | 0.15 m | 0.10 |
| Severe | 0.60 | 3.0 s | 0.35 m | 0.25 |

Missing/observed intervals alternated with exponential durations. Player missing gaps averaged one second; player noise SD was half the ball value. Original missingness was preserved.

The physical heads used histogram gradient boosting: 280 iterations, 31 leaves, minimum leaf size 60, learning rate 0.08, L2 4, seed 20261105, stride 10 over known native-frame labels; 170,190 samples. Three boundary heads used 180 iterations, 15 leaves, minimum leaf 80, learning rate 0.06, L2 6, seed 20261106, no early stopping, stride 5 and a ±12-frame boundary target; 340,251 samples.

The 138,565-parameter relative temporal network used 59 global and 88 per-team features, hidden width 64, kernel 3, dilations 1/2/4/8/16/32 and dropout 0.2. Training used 3,000 steps, seed 20261106, learning rate 0.0007, weight decay 0.001, batch 8, 256-step windows and loss on the central 128 steps, gradient clipping 5, boundary-loss weight 0.2 and same-label smoothness weight 0.05. Checkpoints at steps 1,000/2,000/3,000 were eligible; validation selected step 3,000. Training ran locally on Apple MPS. It did not use a downloaded pretrained video model.

Median/IQR normalization used only 340,374 half-local 5 Hz training rows, pooling teams, with minimum scale 0.001 and clipping at 8. Finite-value flags represent missingness. Normalization is fixed during inference. Five half-local temporal phase streams recover 25 Hz output.

Final selection considered **102 configurations**, of which 46 passed safeguards, over exactly **14 cases**: two dense matches × three views and four provider matches × two conditions. Each candidate had to meet the same fixed physical-tree reference's mean spellF1 separately in dense, detected-only and extrapolated groups, with tolerance `1e-12`. Eligible candidates were ranked by dense spellF1, dense macroF1, then provider spellF1. This protects group means, not each match or an unseen domain.

The retained inference uses a 50/50 physical/neural owner mixture. Other fixed settings include missing-ball owner-logit factor 0.35, team-switch cost 1, dead-entry/exit cost 60, boundary strength 1.5 and completed interior short-spell penalty `6 × max(0, 1 − duration/39)`. The 79-state decoder resets at halves and D. Anonymous motion uses ±5-frame gated assignment with a 13 m/s speed limit. These settings were not replaced by either rejected calibration experiment.

## Numerical validation

The primary metric is the official spellF1, averaging H/A spell matching across temporal-IoU thresholds 0.3/0.4/0.5/0.6/0.7; macroF1 is the first tie-break. Scores below are fractions. Different reference sources are not interchangeable estimates of accuracy.

| Selection group | Cases | Fixed tree spellF1 | Selected spellF1 | Selected macroF1 |
|---|---:|---:|---:|---:|
| Dense, two matches × three views | 6 | 0.619764 | **0.649450** | **0.923827** |
| Provider, detected only | 4 | 0.546487 | 0.560934 | 0.837942 |
| Provider, supplied extrapolation | 4 | 0.575526 | 0.578100 | 0.854754 |

Dense spellF1 by match/view: J03WPY clean **0.710280**, moderate **0.632955**, severe **0.553254**; J03WQQ clean **0.733955**, moderate **0.676111**, severe **0.590145**.

## Diagnostics after selection

On the repeatedly evaluated dense match J03WR9, all 146,810 frames were graded:

| Condition | Earlier development reference spellF1 | Selected spellF1 | Selected macroF1 | Frame accuracy |
|---|---:|---:|---:|---:|
| Clean | 0.717735 | 0.751622 | 0.955178 | 96.30% |
| Moderate | 0.666261 | 0.708157 | 0.934679 | 94.35% |
| Severe | 0.546580 | 0.617035 | 0.907785 | 91.88% |

The earlier development reference is a historical model, distinct from the fixed pure-tree selection reference.

Across the eight modern SkillCorner diagnostic matches, detected-only mean spellF1 increased from **0.504536 to 0.514603**, with 8/8 match wins and paired-match bootstrap 95% interval for the difference **[+0.005348, +0.016241]**. With supplied extrapolation, mean spellF1 fell from **0.570077 to 0.559547**: two wins, six losses, interval **[−0.020932, +0.000953]**. An interval crossing zero does not establish equivalence. Provider macroF1 and dead recall also regressed in both conditions; selected mean macroF1 was 0.826167 detected-only and 0.846605 extrapolated.

Public Metrica partial-event checks:

| Game | Pass-frame team agreement | Exact action-onset agreement | Stoppage-interior recall |
|---|---:|---:|---:|
| 1, reused | 96.65% | 91.86% | 93.55% |
| 2, reused | 96.53% | 93.42% | 91.15% |
| **3, fresh after final choice** | **96.92%** | **92.73%** | **88.35%** |

Game 3 covered 42,338 pass frames from 1,121 passes, 1,141 action onsets, and 27,664 interior frames from 74 stoppages. Its 143,761-frame inference took 14.74 seconds with two CPU threads. Game 1/2 onset agreement and game 2 pass agreement remained below the earlier reference. No model changes followed game 3.

### Fresh legacy Barcelona–Real Madrid result

| Scored subset | Correct / known owner references | Owner agreement |
|---|---:|---:|
| All known references | 15,888 / 19,953 | **79.63%** |
| Ball observed | 14,391 / 16,728 | **86.03%** |
| Ball missing | 1,497 / 3,225 | **46.42%** |
| First half | 8,155 / 9,900 | 82.37% |
| Second half | 7,733 / 10,053 | 76.92% |

Only **19,953 of 28,752 common timestamps (69.40%)** had an owner reference; 8,799 were excluded. Ball missingness was **40.02%** of native half frames but only **16.16%** of scored owner-reference frames. Thus the reported overall agreement disproportionately represents visible-ball periods.

Among the 3,225 missing-ball scored frames, 922 predicted the other team and 806 predicted D. Because references are automatic owner labels without independent live/dead truth, these D disagreements cannot all be certified as false stoppages. Nevertheless, the missing-ball result is a substantial transfer weakness. There is no dense H/A/D accuracy, spellF1 or verified stoppage score for this match.

Inference produced 143,754 labels in 14.62 seconds with two CPU threads. Source identities and frozen release hashes matched before and after; the model was unchanged. This was a diagnostic, not an opportunity to tune on a same-club match.

### Separate 41% missing-ball stress

This used **J03WR9 again**, not a new match. Seed 20261107 selected half-local 50-frame blocks, hiding exactly 60,193 of 146,810 ball observations (41.0006%), without reading labels. Adjacent blocks merged into 708 missing runs, mean 3.40 seconds and maximum 16 seconds. No player removal or new noise was added.

| Condition | spellF1 | macroF1 | Frame accuracy |
|---|---:|---:|---:|
| Original clean input | 0.751622 | 0.955178 | 96.30% |
| New missing-ball mask | **0.631811** | 0.926471 | 93.69% |
| Same mask plus independent per-frame, within-team slot permutation | **Identical labels** | Identical | Identical |

Slot permutation seed was 20261108; zero frames changed. The missing-ball spellF1 loss was **0.119811**. The experiment demonstrates permutation invariance and degradation under this artificial corruption; it does not establish robustness to real broadcast occlusion. The weaker legacy-match result illustrates that distinction.

## Rejected approaches and limits of the evidence

| Attempt | Outcome; selected model remained unchanged |
|---|---|
| Extra finite internal-gap training views | One fit, same four dense games; interpolated corrupted gaps up to 75 frames without using hidden true ball values. Rejected: dense spellF1 **0.637225**, macroF1 **0.920455**, detected spellF1 **0.525783**, extrapolated spellF1 **0.546392**; every acceptance guard failed. |
| Sparse ELASTIC owner-control fine-tuning | Annotation alignment and 228 exact 5 Hz training points were audited. Superseded **before fitting**; no ELASTIC supervision in retained weights. |
| First learned sequence calibration | Disqualified for a mathematical evidence-monotonicity failure at the joint-probability floor. Automated fitting/evaluation had completed, but numeric validation results were **never inspected or used**. |
| Corrected positive-scaling sequence calibration | One 10-parameter convex NLL fit on the original four dense games × three views. Rejected: dense spellF1 **0.584081**, macroF1 **0.918624**, despite provider spellF1 gains to **0.573917/0.596256**. A lower training likelihood loss did not improve the required dense spell metric. |

Both eligible improvement trials required all three validation-group spell means and dense macroF1 to be no worse than the retained model, plus a strict dense lexicographic gain, with `1e-12` numerical tolerance. Neither passed. No subsequent selection experiment was performed. No rejected weak checkpoint, video detector, NVIDIA/Kaggle model, PathCRF checkpoint or SoccerNet checkpoint is part of the selected release.

Remaining limitations include real missing-ball periods, supplied extrapolated tracks, uncertain turnovers and source label conventions. The decoder is not a proof that every football deflection or contested ball obeys the competition's control semantics. The small, repeatedly used dense validation set limits generalization evidence; the fresh tests have partial or automatic references. Complete H/A/D output coverage is not accuracy.

## Implementation and reproduction evidence

Before handoff cleanup, the historical suite recorded **392 passing tests, zero failures**, including synthetic contracts, corruption/selection guards, parsing and decoder mathematics. These are historical checks; the test harnesses are not retained in the inference-only handoff. Subsequent isolated stress and legacy adapter checks passed before their one declared evaluations.

Additional checks established:

- Exact within-team per-frame slot-permutation invariance and exact team-swap equivariance on the 146,211-frame external J03WPY run.
- Identical external labels and CSV bytes in separate processes, after standalone-package migration, and from a freshly extracted copy on the same Mac using the existing pinned environment.
- Public EPTS channel mapping across 143,761 game-3 frames, 11 layouts and 10,350,792 coordinate components: no mapping/value or missing-mask differences.
- Official validator acceptance of a **synthetic** 199,319-row file; synthetic CLI checks covered exact frame order, release verification, fixed status output and overwrite refusal. This establishes the output contract, not target accuracy.
- Archive extraction and hash verification with exact external CSV reproduction. The archive check reused an existing pinned environment; it was not a new dependency-install test.

The reference platform was Python **3.12.12**, macOS **27.0.1 arm64**, with two CPU inference threads. Use the retained dependency lock and current release manifest for the final package. Historical source hashes were not silently rebased after reader or packaging changes. Version 1 consolidated unchanged fitted estimators/tensors; it was not a new training run.

The final reference J03WPY run took **14.18 seconds**. Stored-weight inference is the reproducibility target; bitwise MPS retraining and arbitrary-platform floating-point identity are not promised. Historical diagnostics cannot be rerun from this inference-only handoff without reacquiring the pinned data and reconstructing their evaluation harnesses.

During handoff cleanup, a further label-free J03WPY run reproduced the same 146,211-row CSV hash in **15.36 seconds with two CPU threads**. All three model artifact hashes and the abstract syntax of 42 retained inference classes/functions were unchanged. This verifies the inference migration; it does not rerun the historical performance evaluation.

A separate temporary copy containing only the 31 handoff files then reproduced the same external CSV bytes and generated a complete 199,319-row **synthetic** submission through the documented CLI. The synthetic recording exercised modern EPTS channels, reordered slots and missing ball observations. The official validator passed; the private output manifest matched the CSV/release hashes, output overwrite was refused and a changed parameter file caused verification to fail. The existing pinned environment was reused; no fresh dependency installation or cross-platform equivalence was asserted. Temporary copies, generated inputs/outputs and the downloaded external datasets were removed after verification. The manifest was updated for the reduced source/document inventory, without changing any model weight or parameter.

The subsequent alignment with the team's `main` branch restored the original non-data `starter/`, `evaluation/` and `data/README.md` scaffold, producing a 44-file review tree. Thirteen original files match `main` exactly; the README lists the four modified original files and the model additions. A clean copy ran the new `starter/predict_possession.py` entry point with `MTAH_DATA_DIR` and its default `evaluation/results/` output path. Its complete synthetic CSV was byte-identical to the prior layout (SHA-256 `4d71ebc9811cede8ace970e6e057307a01688d2f9bb1ca7be3c3afbbef948844`) and passed the official validator. Explicit input flags override the environment; existing output is protected. Integrity verification also rejected a changed restored source file. The conventional `requirements.txt` entry point passed an offline installation dry run against the existing pinned environment, and original scorer/validator/baseline help commands remained functional. No training, parameter changes, target-data processing or new external-data download accompanied this layout change. All temporary synthetic inputs and outputs were removed.

At this reference's cutoff, competition assets were absent and a target CSV from the retained model had not been generated. The required local CSV contract is `frame,label`, exactly 199,319 data rows, frames 0–199318, H/A/D/X only. H is Barcelona, A Athletic Club and D stopped play; X outside official graded halves is timeline handling, not learned possession. Target inference and permitted format checks must remain local; no prediction correctness review, manual repair or parameter selection on the target is authorized by these external results.

## Provenance fingerprints

These SHA-256 values identify the evidence, not files that must remain in the repository. The current inference release manifest is authoritative for the packaged source. None of the values below contains match observations.

```text
SELECTED ARTIFACTS
models/model.joblib
  e7ba2bd895a74eee0a82e792e017671102806aadfeeee45e286a8d8553d5f994
models/temporal.pt
  8cee3a9552cc6c6ab37338f185b1708ab2e6e28497441778bbdec1b9c03ce179
models/parameters.json
  f78dc6cf61b501c87d9cd6fbe91646d0867efe5d425ecf2cd61dc230c27aa300

PRE-CONSOLIDATION LINEAGE (unchanged estimator/state_dict, metadata reserialized)
physical heads
  b34bdfe35d49b9fb6e1bb12c0dd3956cde8e165275503564dc063cc125f749a4
selected step-3000 temporal checkpoint
  19f3a5f32b8490dc082ad253e1a51ae3c8e5830d9c4362de7f6042ed64789684

HISTORICAL EVALUATED RELEASE MANIFEST (before handoff source cleanup)
  73a272361ebffb89ae7df791507371555e632cc11c21ad8ac3f99ee9e9484bc8

PROCESSED IDSSE INPUTS
J03WMX 4036634ae1cf90fbc17aa857277d0d432e5143c92babe3692221434b7e1550e5
J03WN1 5e74aa23d67685b6f419c11115c16415cb54395d2108589b06ed1a2f3062ba8b
J03WOH 56046befb2d70bbce651b8d3df46c77792f122016f471b755ab5cba8695c5989
J03WOY cff44a4ec36879d224105c3a7561738674605deae2eca5dbd655575681d32d95
J03WPY 66f08e21b8f4af0468a0ff20abbbbe28ca37f8e41cbb086db4cacf1aace068c3
J03WQQ 00296c962f26a9564b10f7658c40146982385c6ba10f4f9b5d40027c0bfbe404
J03WR9 9e08f9f905e22a3c52d97e161cf955962b7733fb1602e9fd924f8b69f04caf55

ORIGINAL ACQUISITION MANIFESTS
IDSSE 8734df97554fa788baf46a8cacd6e897614785246a42f2221829c23e1e36388b
modern SkillCorner 63fdf95e83f48de32f0594f82aa5840cc8131255d187c8979e7b5da7a78c3795
Metrica games 1/2 7dc08fe0717f5901f23627b8961c3cb3e852e46cab06fbbefd8e6d52a02f6150
Metrica game 3 5b3345573813c6f9fbf211f36ef3873f25394af1650b82d40fda6351bdaa9977

LEGACY MATCH 2841
structured_data.json
  6f9fca12bbcc4ad0f5be2e2a5b95856a429cf6023e658b0ab1fded99b89b20e4
match_data.json
  9fb0185520f721dcfc9804f2a06915dad0989d11cf1380be0e65079c15790707
evaluated adapter source
  5192ef1e360d28617126884c3e7378ecd1f82f650406f0fb0684e2c9a92786bd
result record
  5ff964fc068e97e449824d1759502bd07076a69bdae57056cebafdfc490edb75

41% STRESS
protocol 12e6d2e4b592b7035ae1c433c1259ee897b7af8a217ab50ddc72d948682cfec2
added mask 7775138250ec5b7c4b19ffdc55ef460622347fe2aef2658949c5563060a9ca15
result 475e2d80320c769b90b851bf968f251aaa7e8d3adf7749ecbdea62d9c306cb8b

EXTERNAL REPRODUCTION: J03WPY, 146211 frames
label array 079e64fe7984ba99673fbac3e776f306a58fac356add866f8289038a55aa8188
CSV bytes 4a2cf6cce9e055d1a3bd93aef314a9a07cf0a773c1ff10dc08b42ce25799746d

OFFICIAL EVALUATION UTILITIES USED
scorer 341bad48904b489d6e875397b27687f5d6a4d22a397f3dc8af3a00d48f16874d
validator 07ac29499acbd337e1dc1de1832d63f6e1fa1d981e22f6ad08212529e2bbee14
```

The official metric and output requirements are defined in the read-only [competition repository](https://github.com/metrica-sports/more-than-a-hack-2027). The participant clarification requiring external-only training/tuning takes precedence over older public wording permitting target self-supervision.

## Football dead-ball model (added to version 1)

`models/deadball.joblib` is a histogram gradient-boosting classifier of live versus dead (D), 300 iterations, learning rate 0.06, 31 leaves, seed 20261113. Its 205 inputs are described in `persian_lion/football.py` (schema `touchline_temporal_v2`): player geometry, the ball position when seen, and about 10 s of context. Target: the provider's `BallStatus`. Fit matches: J03WMX, J03WN1, J03WOH, J03WOY (every 5th frame inside the halves; each match three times: as tracked, with the ball hidden in random stretches, and with corrupted players plus hidden-ball stretches; 340,374 rows). Held out: J03WPY, J03WQQ, J03WR9. Source licence, revision and attribution are the IDSSE entries above. No competition data is used for fitting. The corruption settings in `external/augment.py` resemble aggregate, unlabelled statistics of the provided tracking (disclosed in the technical document).

| Held-out evaluation | Earlier dead-ball model (88 player-only features) | This model (205 features) |
|---|---|---|
| Dead-ball F1, corrupted players + hidden ball, three matches | 0.902 to 0.926 | 0.956 to 0.971 |
| Dead-ball F1, clean | 0.911 to 0.935 | 0.998 to 0.999 |
| False D on live frames, corrupted + hidden ball | 4.6% to 5.0% | 2.3% to 2.5% |
| Mean spellF1 through the decoder, corrupted players + realistic ball gaps (best weight) | 0.628 (weight 0.65) | 0.650 (weight 0.9) |
| Mean spellF1, clean players + realistic ball gaps | 0.674 (weight 0.5) | 0.683 (weight 0.9) |

The blend weight 0.9 was picked with `external/sweep_blend_weight.py` on the held-out matches (weights 0.9 and 1.0 differ by 0.001 spellF1). Clean-ball results near 1.0 mostly reflect the ball being visible: a first model trained without hidden-ball stretches scored 0.999 yet was worse than the earlier model once the ball was hidden, which is why the hidden-ball stretches are part of training. Caveats: the corruption generators are the same family in training and evaluation, the three held-out matches are a small sample, all are development-exposed, and the provider's `BallStatus` is not the competition's definition of stopped play.
