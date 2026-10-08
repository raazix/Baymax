# Demo images

Upload these on the **Camera station** tab with **Analyse image file**. Pick the matching model first.

**Important:** these are *hand-picked best cases* from the held-out test split, chosen because the models handle them clearly. They are not a representative sample and say nothing about production accuracy. Casting scores and NEU detections below come from the live API on the active checkpoints. Neither dataset contains brake discs.

## 1. Steel-surface defects — model: "Steel-surface defects (YOLO11n)"

Folder `1_steel_defects_YOLO`. Each image has the named defect class correctly detected.

| File | Defect class | Top confidence | Boxes drawn |
|---|---|---|---|
| crazing_188.jpg | crazing | 68% | 3 |
| crazing_292.jpg | crazing | 61% | 2 |
| inclusion_86.jpg | inclusion | 84% | 6 |
| inclusion_261.jpg | inclusion | 80% | 3 |
| patches_248.jpg | patches | 92% | 5 |
| patches_129.jpg | patches | 91% | 1 |
| pitted_surface_291.jpg | pitted surface | 91% | 3 |
| pitted_surface_293.jpg | pitted surface | 91% | 1 |
| rolled-in_scale_214.jpg | rolled-in scale | 87% | 5 |
| rolled-in_scale_277.jpg | rolled-in scale | 84% | 5 |
| scratches_111.jpg | scratches | 88% | 4 |
| scratches_55.jpg | scratches | 86% | 1 |

Boxes beyond the true class may be extra detections of other classes; show the strongest one.

## 2. Casting anomaly — model: "Casting anomaly (PatchCore)"

Folder `2_casting_anomaly_PatchCore`. Decision threshold is 2.322; score above it is flagged **Anomalous**. Score is a feature distance, not a probability.

| File | Truth | Anomaly score |
|---|---|---|
| defect_score3.81_cast_def_0_8615.jpeg | defect | 3.809 |
| defect_score3.74_cast_def_0_3480.jpeg | defect | 3.738 |
| defect_score3.69_cast_def_0_1077.jpeg | defect | 3.691 |
| defect_score3.69_cast_def_0_6417.jpeg | defect | 3.686 |
| normal_score2.02_cast_ok_0_1604.jpeg | normal | 2.022 |
| normal_score2.03_cast_ok_0_1879.jpeg | normal | 2.027 |
| normal_score2.03_cast_ok_0_1695.jpeg | normal | 2.033 |
| normal_score2.04_cast_ok_0_4358.jpeg | normal | 2.036 |

Filenames start with the true label and score so you can tell which is which. Rename them if you do not want the audience to see the answer.

Most casting images are not this clear-cut: on the full test set the calibrated threshold still misses some defects and flags some normal parts. The earlier uncalibrated threshold flagged 43.6% of normal parts.
