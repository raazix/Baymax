# Calibration status (2026-10-09)

Calibration parameters are stored in `models/calibration/calibration.json`. Runtime use is gated by the exact model artifact SHA-256 where a fitted parameter belongs to one model. Replacing model weights disables that calibration until it is re-evaluated.

| Signal | Runtime behavior | Held-out evidence and scope |
|---|---|---|
| NEU YOLO confidence | Isotonic-calibrated confidence is shown beside raw detector confidence when the loaded weights match. | NEU validation/test split, pooled classes (per-class samples too small): ECE 0.1179 to 0.0642, Brier 0.1770 to 0.1656, n=678. Steel-surface proxy only; not validated on brake discs. |
| General corrosion model | Raw probability remains the decision input; artifact-matched calibrated probability is displayed as additional context. | General rust photo validation/test: presence ECE 0.0597 to 0.0299 and Brier 0.0659 to 0.0570 (n=939); severe-grade ECE 0.1518 to 0.0733 and Brier 0.2304 to 0.2112 (n=811). Does not establish brake-disc accuracy. Domain-specific plate thresholds are not calibrated by this mapping. |
| PLSR forecast interval | Residual interval is widened by 1.15 when the forecast artifact matches. Forecast remains a synthetic predicted defect fraction, not a validated event probability. | Synthetic chronologically later lots: 95% interval coverage 0.889/0.939 before and 0.967/0.972 after, on two halves of 180 lots. |
| XGBoost RCA | No probability correction is applied. | Synthetic held-out lots: raw ECE 0.0136, Brier 0.0272, accuracy 0.9633; temperature scaling worsened ECE to 0.0204 and Brier to 0.0290. Synthetic evidence only. |
| PatchCore casting bank | No calibrated probability is exposed at runtime. Raw anomaly score and threshold remain model-specific. | Casting validation/test Platt result (test ECE 0.0247, Brier 0.0646, n=469) reflects an 83% defective dataset prevalence and a different artifact. It is not applicable to custom brake-disc or MPDD banks. |
| Custom brake-disc PatchCore | Not calibrated; threshold is not presented as a validated brake-disc decision boundary. | The current 40-image folder check at the held-out-normal threshold found 0/22 `correct`, 1/6 `defects`, and 0/12 `rusted` flagged (1/18 labeled defect/rust images). This is inadequate defect sensitivity. General rust classifier on the same small folder flagged 9/12 rusted and 4/6 other defects. This folder is exploratory evidence, not an independent validation set. |

## Engineering limits

Severity rules are deterministic software rules, not model confidence. They need engineer-approved defect-size and location limits for each part before severity can be called calibrated. The current user folder does not provide enough verified measurements or labels to set those limits. Preserve unknown/out-of-domain results as unclassified, and label anomaly maps as anomaly evidence rather than a defect mask or brake-disc classification.

The calibration command is `python scripts/calibrate_confidences.py --only <corrosion|yolo|patchcore|rca|forecast>` (use the project virtual environment). The existing `--only yolo` and `--only forecast` runs completed on 2026-10-09. Re-run calibration and review the dataset split before changing runtime thresholds or severity rules.
