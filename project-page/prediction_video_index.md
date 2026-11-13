# Prediction-video index — best / worst clip per approach

Source: `results/videos/` (one `.mp4` per clip, two stills per clip — `_worst` + `_middle` — and `results.json` per folder).
Paths below are relative to the repository root. Error = per-clip mean in mm/node (`results.json` -> `clips.<id>.reproduced`), which matches the table values.
Machine-readable copy: `project-page/prediction_video_index.json`.

Videos exist for 8 scored approaches + 2 superseded earlier runs. Newton and DLO-Lab have no overlay videos yet.

## Summary — best and worst clip

| Approach | Folder | Mean (mm) | Best clip | Best err | Worst clip | Worst err |
|---|---|---|---|---|---|---|
| mujoco | `mujoco_9.7` | 9.696 | 20261021_164755 | 5.914 | 20261026_130441 | 22.347 |
| adaptigraph | `adaptigraph_10.3` | 10.262 | 20261021_164755 | 6.187 | 20261026_130441 | 16.923 |
| genesis | `genesis_12.7` | 12.683 | 20261026_125511 | 7.026 | 20261021_180139 | 36.44 |
| caporali | `caporali_15.8` | 15.779 | 20261021_164755 | 10.221 | 20261021_180139 | 33.967 |
| yang | `yang_16.0` | 16.003 | 20261021_170550 | 7.535 | 20261021_180139 | 54.226 |
| ropedreamer | `ropedreamer_18.5` | 18.488 | 20261026_130128 | 10.117 | 20261021_160124 | 46.615 |
| phystwin (superseded) | `phystwin_22.0` | 22.034 | 20261026_130128 | 10.797 | 20261021_175638 | 53.777 |

## Per approach


### mujoco_9.7
- Approach: MuJoCo general fit (181 parameters), 36-clip refit
- Metric: 16 nodes, mean over cables, every frame (mujoco_general_fit_cmaes._eval_one)
- Mean over 36 clips: **9.696 mm/node**

**Best clip — 20261021_164755 (5.914 mm)**
- video: `results/videos/mujoco_9.7/mujoco_20261021_164755.mp4`
- middle still: `results/videos/mujoco_9.7/mujoco_20261021_164755_frame082_middle.png`
- worst-frame still: `results/videos/mujoco_9.7/mujoco_20261021_164755_frame065_worst.png`

**Worst clip — 20261026_130441 (22.347 mm)**
- video: `results/videos/mujoco_9.7/mujoco_20261026_130441.mp4`
- middle still: `results/videos/mujoco_9.7/mujoco_20261026_130441_frame082_middle.png`
- worst-frame still: `results/videos/mujoco_9.7/mujoco_20261026_130441_frame104_worst.png`

<details><summary>All 36 clips (best → worst)</summary>

| # | Clip | Error (mm) |
|---|---|---|
| 1 | 20261021_164755 | 5.914 |
| 2 | 20261021_163624 | 6.062 |
| 3 | 20261021_171842 | 6.536 |
| 4 | 20261026_130128 | 6.558 |
| 5 | 20261026_125511 | 6.642 |
| 6 | 20261021_173628 | 6.743 |
| 7 | 20261021_173858 | 6.745 |
| 8 | 20261021_164524 | 6.852 |
| 9 | 20261021_165753 | 6.865 |
| 10 | 20261021_162340 | 6.891 |
| 11 | 20261021_174701 | 6.935 |
| 12 | 20261021_172103 | 6.938 |
| 13 | 20261021_164010 | 7.16 |
| 14 | 20261021_172928 | 7.275 |
| 15 | 20261021_180646 | 7.444 |
| 16 | 20261021_170550 | 7.501 |
| 17 | 20261021_172318 | 7.546 |
| 18 | 20261021_165018 | 7.652 |
| 19 | 20261021_173234 | 7.815 |
| 20 | 20261021_172716 | 8.059 |
| 21 | 20261021_174117 | 8.192 |
| 22 | 20261021_165246 | 8.376 |
| 23 | 20261028_100701 | 8.451 |
| 24 | 20261021_162632 | 9.017 |
| 25 | 20261028_101001 | 9.845 |
| 26 | 20261028_101510 | 10.213 |
| 27 | 20261021_180422 | 10.561 |
| 28 | 20261021_163244 | 11.054 |
| 29 | 20261028_121810 | 12.489 |
| 30 | 20261021_160124 | 12.897 |
| 31 | 20261028_105803 | 14.49 |
| 32 | 20261021_171620 | 14.953 |
| 33 | 20261021_180139 | 16.574 |
| 34 | 20261028_114103 | 17.945 |
| 35 | 20261021_175638 | 21.52 |
| 36 | 20261026_130441 | 22.347 |

</details>

### adaptigraph_10.3
- Approach: AdaptiGraph, rebuilt (K=3 BPTT, rest + end-rotation inputs, noise), trained on simulation data from the 36 clips only; 4 phi (one per cable) fitted on all clips and frames
- Metric: 16 nodes (ends pinned), mean over cables, frames 5..T-1 (after warm-up)
- Mean over 36 clips: **10.262 mm/node**

**Best clip — 20261021_164755 (6.187 mm)**
- video: `results/videos/adaptigraph_10.3/adaptigraph_20261021_164755.mp4`
- middle still: `results/videos/adaptigraph_10.3/adaptigraph_20261021_164755_frame082_middle.png`
- worst-frame still: `results/videos/adaptigraph_10.3/adaptigraph_20261021_164755_frame065_worst.png`

**Worst clip — 20261026_130441 (16.923 mm)**
- video: `results/videos/adaptigraph_10.3/adaptigraph_20261026_130441.mp4`
- middle still: `results/videos/adaptigraph_10.3/adaptigraph_20261026_130441_frame082_middle.png`
- worst-frame still: `results/videos/adaptigraph_10.3/adaptigraph_20261026_130441_frame110_worst.png`

<details><summary>All 36 clips (best → worst)</summary>

| # | Clip | Error (mm) |
|---|---|---|
| 1 | 20261021_164755 | 6.187 |
| 2 | 20261021_172318 | 7.205 |
| 3 | 20261026_125511 | 7.734 |
| 4 | 20261021_171620 | 7.944 |
| 5 | 20261021_180646 | 8.1 |
| 6 | 20261021_172103 | 8.157 |
| 7 | 20261021_165246 | 8.191 |
| 8 | 20261021_164010 | 8.27 |
| 9 | 20261021_174117 | 8.359 |
| 10 | 20261021_165753 | 8.389 |
| 11 | 20261021_162340 | 8.389 |
| 12 | 20261021_165018 | 8.398 |
| 13 | 20261021_173858 | 8.406 |
| 14 | 20261021_173628 | 8.457 |
| 15 | 20261021_163624 | 8.788 |
| 16 | 20261021_171842 | 9.068 |
| 17 | 20261028_114103 | 9.102 |
| 18 | 20261021_164524 | 9.108 |
| 19 | 20261028_101001 | 9.117 |
| 20 | 20261026_130128 | 9.246 |
| 21 | 20261021_174701 | 9.424 |
| 22 | 20261021_172928 | 9.694 |
| 23 | 20261028_100701 | 10.161 |
| 24 | 20261021_173234 | 10.542 |
| 25 | 20261021_172716 | 10.558 |
| 26 | 20261021_162632 | 10.754 |
| 27 | 20261021_180422 | 11.026 |
| 28 | 20261028_101510 | 11.774 |
| 29 | 20261021_163244 | 13.481 |
| 30 | 20261021_170550 | 13.692 |
| 31 | 20261021_175638 | 13.938 |
| 32 | 20261028_105803 | 14.116 |
| 33 | 20261021_180139 | 14.927 |
| 34 | 20261028_121810 | 15.453 |
| 35 | 20261021_160124 | 16.35 |
| 36 | 20261026_130441 | 16.923 |

</details>

### genesis_12.7
- Approach: Genesis general fit, 36-clip refit (stopped after generation 33)
- Metric: 16 nodes, mean over cables, every frame (genesis_general_fit_cmaes._eval_one)
- Mean over 36 clips: **12.683 mm/node**

**Best clip — 20261026_125511 (7.026 mm)**
- video: `results/videos/genesis_12.7/genesis_20261026_125511.mp4`
- middle still: `results/videos/genesis_12.7/genesis_20261026_125511_frame084_middle.png`
- worst-frame still: `results/videos/genesis_12.7/genesis_20261026_125511_frame070_worst.png`

**Worst clip — 20261021_180139 (36.44 mm)**
- video: `results/videos/genesis_12.7/genesis_20261021_180139.mp4`
- middle still: `results/videos/genesis_12.7/genesis_20261021_180139_frame078_middle.png`
- worst-frame still: `results/videos/genesis_12.7/genesis_20261021_180139_frame148_worst.png`

<details><summary>All 36 clips (best → worst)</summary>

| # | Clip | Error (mm) |
|---|---|---|
| 1 | 20261026_125511 | 7.026 |
| 2 | 20261026_130128 | 7.129 |
| 3 | 20261021_164010 | 7.812 |
| 4 | 20261021_164755 | 8.184 |
| 5 | 20261021_162340 | 8.238 |
| 6 | 20261021_172103 | 8.273 |
| 7 | 20261021_165246 | 8.521 |
| 8 | 20261021_174701 | 8.566 |
| 9 | 20261021_165753 | 8.584 |
| 10 | 20261021_174117 | 8.908 |
| 11 | 20261021_164524 | 8.982 |
| 12 | 20261021_163624 | 9.006 |
| 13 | 20261021_172318 | 9.104 |
| 14 | 20261028_101001 | 9.189 |
| 15 | 20261021_172928 | 9.228 |
| 16 | 20261021_171842 | 9.47 |
| 17 | 20261021_165018 | 9.737 |
| 18 | 20261021_173858 | 9.769 |
| 19 | 20261021_173628 | 10.376 |
| 20 | 20261021_180646 | 10.616 |
| 21 | 20261021_173234 | 11.045 |
| 22 | 20261021_180422 | 11.096 |
| 23 | 20261028_101510 | 11.129 |
| 24 | 20261021_172716 | 11.464 |
| 25 | 20261021_162632 | 12.183 |
| 26 | 20261028_100701 | 13.007 |
| 27 | 20261021_170550 | 13.243 |
| 28 | 20261021_163244 | 15.167 |
| 29 | 20261028_121810 | 16.868 |
| 30 | 20261028_114103 | 16.98 |
| 31 | 20261028_105803 | 17.513 |
| 32 | 20261021_171620 | 18.917 |
| 33 | 20261021_160124 | 22.6 |
| 34 | 20261021_175638 | 25.459 |
| 35 | 20261026_130441 | 26.749 |
| 36 | 20261021_180139 | 36.44 |

</details>


### caporali_15.8
- Approach: Caporali et al., FiLM MLP trained on data from the 36 clips, online (m, k_b, k_d)
- Metric: 16 nodes, mean over cables, every frame incl. frame 0
- Mean over 36 clips: **15.779 mm/node**

**Best clip — 20261021_164755 (10.221 mm)**
- video: `results/videos/caporali_15.8/caporali_20261021_164755.mp4`
- middle still: `results/videos/caporali_15.8/caporali_20261021_164755_frame082_middle.png`
- worst-frame still: `results/videos/caporali_15.8/caporali_20261021_164755_frame148_worst.png`

**Worst clip — 20261021_180139 (33.967 mm)**
- video: `results/videos/caporali_15.8/caporali_20261021_180139.mp4`
- middle still: `results/videos/caporali_15.8/caporali_20261021_180139_frame078_middle.png`
- worst-frame still: `results/videos/caporali_15.8/caporali_20261021_180139_frame155_worst.png`

<details><summary>All 36 clips (best → worst)</summary>

| # | Clip | Error (mm) |
|---|---|---|
| 1 | 20261021_164755 | 10.221 |
| 2 | 20261021_172318 | 10.329 |
| 3 | 20261021_171842 | 10.848 |
| 4 | 20261026_130128 | 10.874 |
| 5 | 20261021_172103 | 11.474 |
| 6 | 20261021_165018 | 12.224 |
| 7 | 20261021_162340 | 12.232 |
| 8 | 20261021_164010 | 12.272 |
| 9 | 20261021_163624 | 12.443 |
| 10 | 20261021_164524 | 12.513 |
| 11 | 20261021_165246 | 12.652 |
| 12 | 20261021_180646 | 12.795 |
| 13 | 20261021_173234 | 12.876 |
| 14 | 20261021_173858 | 12.976 |
| 15 | 20261021_173628 | 13.154 |
| 16 | 20261021_171620 | 13.174 |
| 17 | 20261021_174117 | 14.082 |
| 18 | 20261021_180422 | 14.374 |
| 19 | 20261021_174701 | 14.695 |
| 20 | 20261021_163244 | 14.716 |
| 21 | 20261021_172928 | 15.038 |
| 22 | 20261021_165753 | 15.279 |
| 23 | 20261026_125511 | 15.387 |
| 24 | 20261021_170550 | 16.619 |
| 25 | 20261028_101001 | 16.972 |
| 26 | 20261028_105803 | 17.008 |
| 27 | 20261028_100701 | 17.872 |
| 28 | 20261028_101510 | 17.897 |
| 29 | 20261021_162632 | 18.283 |
| 30 | 20261021_160124 | 18.308 |
| 31 | 20261021_172716 | 18.847 |
| 32 | 20261026_130441 | 20.465 |
| 33 | 20261028_114103 | 24.576 |
| 34 | 20261021_175638 | 25.232 |
| 35 | 20261028_121810 | 25.363 |
| 36 | 20261021_180139 | 33.967 |

</details>

### yang_16.0
- Approach: Yang et al., IN + bi-LSTM trained on Newton rollouts from the DE fit (stored network replayed, no re-training)
- Metric: 16 nodes (ends pinned), mean over cables, frames 5..T-1 (after warm-up)
- Mean over 36 clips: **16.003 mm/node**

**Best clip — 20261021_170550 (7.535 mm)**
- video: `results/videos/yang_16.0/yang_20261021_170550.mp4`
- middle still: `results/videos/yang_16.0/yang_20261021_170550_frame079_middle.png`
- worst-frame still: `results/videos/yang_16.0/yang_20261021_170550_frame144_worst.png`

**Worst clip — 20261021_180139 (54.226 mm)**
- video: `results/videos/yang_16.0/yang_20261021_180139.mp4`
- middle still: `results/videos/yang_16.0/yang_20261021_180139_frame078_middle.png`
- worst-frame still: `results/videos/yang_16.0/yang_20261021_180139_frame148_worst.png`

<details><summary>All 36 clips (best → worst)</summary>

| # | Clip | Error (mm) |
|---|---|---|
| 1 | 20261021_170550 | 7.535 |
| 2 | 20261021_164524 | 7.613 |
| 3 | 20261021_164755 | 7.977 |
| 4 | 20261021_165018 | 8.247 |
| 5 | 20261021_172318 | 8.741 |
| 6 | 20261021_173628 | 8.817 |
| 7 | 20261021_172103 | 9.215 |
| 8 | 20261028_101510 | 9.404 |
| 9 | 20261021_165246 | 9.408 |
| 10 | 20261021_164010 | 9.988 |
| 11 | 20261021_163624 | 10.0 |
| 12 | 20261021_172716 | 10.123 |
| 13 | 20261021_174701 | 10.312 |
| 14 | 20261021_171842 | 10.847 |
| 15 | 20261028_101001 | 10.872 |
| 16 | 20261026_125511 | 11.041 |
| 17 | 20261021_173858 | 11.264 |
| 18 | 20261026_130128 | 11.691 |
| 19 | 20261028_100701 | 12.207 |
| 20 | 20261021_172928 | 12.524 |
| 21 | 20261028_105803 | 12.931 |
| 22 | 20261021_162632 | 13.061 |
| 23 | 20261021_163244 | 13.155 |
| 24 | 20261021_180646 | 13.529 |
| 25 | 20261021_174117 | 13.852 |
| 26 | 20261021_173234 | 13.916 |
| 27 | 20261021_165753 | 14.243 |
| 28 | 20261021_162340 | 15.777 |
| 29 | 20261021_180422 | 20.496 |
| 30 | 20261021_160124 | 22.223 |
| 31 | 20261026_130441 | 24.902 |
| 32 | 20261028_114103 | 32.625 |
| 33 | 20261021_171620 | 32.839 |
| 34 | 20261021_175638 | 33.456 |
| 35 | 20261028_121810 | 47.052 |
| 36 | 20261021_180139 | 54.226 |

</details>

### ropedreamer_18.5
- Approach: RopeDreamer, RSSM over a quaternionic chain, trained on MuJoCo rollouts of the 36 clips only (stored network replayed, no re-training)
- Metric: 16 nodes (ends pinned), mean over cables, frames 5..T-1 (after warm-up)
- Mean over 36 clips: **18.488 mm/node**

**Best clip — 20261026_130128 (10.117 mm)**
- video: `results/videos/ropedreamer_18.5/ropedreamer_20261026_130128.mp4`
- middle still: `results/videos/ropedreamer_18.5/ropedreamer_20261026_130128_frame089_middle.png`
- worst-frame still: `results/videos/ropedreamer_18.5/ropedreamer_20261026_130128_frame177_worst.png`

**Worst clip — 20261021_160124 (46.615 mm)**
- video: `results/videos/ropedreamer_18.5/ropedreamer_20261021_160124.mp4`
- middle still: `results/videos/ropedreamer_18.5/ropedreamer_20261021_160124_frame083_middle.png`
- worst-frame still: `results/videos/ropedreamer_18.5/ropedreamer_20261021_160124_frame123_worst.png`

<details><summary>All 36 clips (best → worst)</summary>

| # | Clip | Error (mm) |
|---|---|---|
| 1 | 20261026_130128 | 10.117 |
| 2 | 20261021_164524 | 11.563 |
| 3 | 20261021_172318 | 11.827 |
| 4 | 20261021_163624 | 12.024 |
| 5 | 20261021_180646 | 12.346 |
| 6 | 20261028_101510 | 12.549 |
| 7 | 20261021_173858 | 12.868 |
| 8 | 20261021_174701 | 13.137 |
| 9 | 20261026_125511 | 13.714 |
| 10 | 20261021_173628 | 13.869 |
| 11 | 20261021_172103 | 14.071 |
| 12 | 20261021_165246 | 14.17 |
| 13 | 20261021_164755 | 14.574 |
| 14 | 20261021_171842 | 15.029 |
| 15 | 20261021_165753 | 15.121 |
| 16 | 20261021_172716 | 15.138 |
| 17 | 20261021_180422 | 15.145 |
| 18 | 20261028_101001 | 15.203 |
| 19 | 20261021_172928 | 15.22 |
| 20 | 20261021_173234 | 15.489 |
| 21 | 20261021_164010 | 15.903 |
| 22 | 20261021_162632 | 15.914 |
| 23 | 20261021_162340 | 16.163 |
| 24 | 20261021_165018 | 17.64 |
| 25 | 20261021_174117 | 18.001 |
| 26 | 20261028_100701 | 19.647 |
| 27 | 20261028_105803 | 20.363 |
| 28 | 20261021_163244 | 22.779 |
| 29 | 20261021_171620 | 22.782 |
| 30 | 20261028_121810 | 22.883 |
| 31 | 20261021_175638 | 24.292 |
| 32 | 20261021_170550 | 25.612 |
| 33 | 20261026_130441 | 28.825 |
| 34 | 20261028_114103 | 33.731 |
| 35 | 20261021_180139 | 41.252 |
| 36 | 20261021_160124 | 46.615 |

</details>

## Superseded runs


### phystwin_22.0
> Superseded earlier run — kept for reference only.
- Approach: PhysTwin, per-clip dense twins refitted on the 36-clip selection
- Metric: 14 inner nodes (ends kinematic), mean over cables, every frame
- Mean over 36 clips: **22.034 mm/node**

**Best clip — 20261026_130128 (10.797 mm)**
- video: `results/videos/phystwin_22.0/phystwin_20261026_130128.mp4`
- middle still: `results/videos/phystwin_22.0/phystwin_20261026_130128_frame089_middle.png`
- worst-frame still: `results/videos/phystwin_22.0/phystwin_20261026_130128_frame046_worst.png`

**Worst clip — 20261021_175638 (53.777 mm)**
- video: `results/videos/phystwin_22.0/phystwin_20261021_175638.mp4`
- middle still: `results/videos/phystwin_22.0/phystwin_20261021_175638_frame077_middle.png`
- worst-frame still: `results/videos/phystwin_22.0/phystwin_20261021_175638_frame154_worst.png`

<details><summary>All 36 clips (best → worst)</summary>

| # | Clip | Error (mm) |
|---|---|---|
| 1 | 20261026_130128 | 10.797 |
| 2 | 20261021_164010 | 11.1 |
| 3 | 20261021_162340 | 11.932 |
| 4 | 20261021_164755 | 11.959 |
| 5 | 20261021_164524 | 13.538 |
| 6 | 20261021_173628 | 14.571 |
| 7 | 20261021_165753 | 14.588 |
| 8 | 20261021_170550 | 14.631 |
| 9 | 20261021_163624 | 14.768 |
| 10 | 20261021_172103 | 15.445 |
| 11 | 20261021_172928 | 15.467 |
| 12 | 20261021_172318 | 15.832 |
| 13 | 20261028_101510 | 15.85 |
| 14 | 20261028_101001 | 16.394 |
| 15 | 20261021_180422 | 17.016 |
| 16 | 20261021_162632 | 17.03 |
| 17 | 20261021_165018 | 17.165 |
| 18 | 20261021_172716 | 17.495 |
| 19 | 20261021_163244 | 17.506 |
| 20 | 20261028_100701 | 17.517 |
| 21 | 20261021_173858 | 17.525 |
| 22 | 20261021_180646 | 18.163 |
| 23 | 20261021_173234 | 20.664 |
| 24 | 20261021_174117 | 21.651 |
| 25 | 20261028_121810 | 22.94 |
| 26 | 20261021_174701 | 22.96 |
| 27 | 20261021_171842 | 22.995 |
| 28 | 20261021_165246 | 23.145 |
| 29 | 20261028_105803 | 23.359 |
| 30 | 20261026_125511 | 24.326 |
| 31 | 20261021_171620 | 36.502 |
| 32 | 20261021_160124 | 43.042 |
| 33 | 20261026_130441 | 43.697 |
| 34 | 20261028_114103 | 46.532 |
| 35 | 20261021_180139 | 51.35 |
| 36 | 20261021_175638 | 53.777 |

</details>
