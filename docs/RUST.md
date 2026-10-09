# Corrosion (rust) classifier

Source: Roboflow Universe "Rust Detection" v1 (`Rust Detection.v1i.multiclass.zip`), CC BY 4.0, image-level
multi-label tags. Code: `scripts/prepare_rust.py`, `scripts/train_rust.py`, `scripts/calibrate_rust_domain.py`,
runtime `vision/rust_classifier.py`, tests `tests/test_rust.py`.

## Why the supplied splits were not used

- The 9,473 "train" images are Roboflow 2x2 **mosaics**: each stitches tiles from several photos and carries the union
  of their tags. The supplied test split also contains mosaics.
- Every validation and test source photo also appears in train (226 shared sources); validation and test share 116.
- The archive holds only 826 distinct source photos, each expanded about 11.5 times.

`prepare_rust.py` removes mosaics by detecting full-width/full-height seams (5,412 excluded; visual spot checks of both
classes), keeps single photos, and re-splits **by source photo** (seeded, stratified 60/20/20). A test asserts that no
source and no identical image crosses splits.

| Split | Images | Source photos | Corrosion | No corrosion |
|---|---|---|---|---|
| Train | 2,741 | 463 | 2,329 | 412 |
| Validation | 971 | 156 | 842 | 129 |
| Test | 939 | 154 | 811 | 128 |

Targets: **corrosion present** (any rust/corrosion tag; "car" is an object, not a defect) and **severe corrosion**
(worst grade is severe). Mild and moderate are too rare per photo (23 and 30) to learn separately.

## Model and results

ResNet18 from the project's pinned ImageNet weights, two sigmoid heads, imbalance-weighted loss, 12 epochs, epoch
selected on validation (epoch 3), thresholds by Youden J on validation (corrosion 0.862, severe 0.516), then the
untouched test split scored once (`models/rust/evaluation.json`).

| Test | Corrosion AUROC | Recall | Specificity | Precision | Severe-grade AUROC |
|---|---|---|---|---|---|
| Per image (939) | 0.957 | 82.2% | 96.1% | 99.3% | 0.741 |
| Per source photo (154) | 0.968 | 82.3% | 100% (only 7 negatives) | 100% | 0.839 |

## Use in the pipeline

The classifier runs on every quality-passed inspection. A detected corrosion becomes a `corrosion` or
`severe_corrosion` finding; its class-activation map gives a coarse region for size and position. Severity rules v3
rate it like any other class (corrosion medium, severe corrosion high, never critical). The probability only gates the
finding; it never sets severity.

### Out-of-domain behaviour (measured)

| Images | General threshold | Handling |
|---|---|---|
| Casting test (78 normal, 150 defective), greyscale | 0 flagged | skipped: greyscale (colour gate) |
| NEU steel test (270), greyscale | 45 flagged, mostly "patches" | skipped: greyscale (colour gate) |
| MPDD metal plates, colour | 24 of 26 good plates flagged | plate-specific threshold |

Plate threshold = maximum corrosion score over the 54 MPDD metal_plate **train/good** images (0.985), the same
normal-only rule used for PatchCore. Untouched MPDD test: rust vs non-rust AUROC 0.979; rust recall 31/37; good false
alarms 1/26; scratched plates flagged 1/34 (`models/rust/domain_thresholds.json`). A 95th-percentile rule was also
scored (32/37, same false alarms) and not chosen. The severe head is not trusted on plates, so plates get presence only.

Limits: noisy public tags, general rust photography rather than automotive brake discs, heuristic mosaic removal, and
only 128 corrosion-free test images.
