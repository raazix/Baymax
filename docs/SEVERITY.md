# Severity rules v3 and YOLO spatial merge

Code: `vision/severity.py` (`classify_severity`), applied in `core/upload_inspection.py` after measurement and merge.
Tests: `tests/test_severity_merge.py`.

## Severity matrix

Severity is deterministic. **Model confidence and anomaly score are never inputs.**

| Step | Rule |
|---|---|
| Base by class | crack-like (crazing, crack) **high** · inclusion, pitting **medium** · patches, rolled-in scale, scratches **low** · unclassified anomaly **medium** |
| Size | with a scale: longest side ≥ 5 mm +1, ≥ 15 mm +2 · without a scale: longest side ≥ 15% of the part diameter (or frame) +1, ≥ 35% +2 |
| Location | rim / friction edge or bore / hub zone: +1 for crack-like or unclassified findings |
| Multiplicity | 3 or more findings on the part: +1 |
| Caps | only crack-like defects can reach **critical**, and only with physical evidence (a measured size in mm or a located rim/hub zone; image pixels alone cap at **high**); an unclassified anomaly is capped at **high** until an engineer classifies it |

Each result records the factors that applied, the rule version and `validated_for_production: false`. The thresholds
are demo engineering assumptions for a brake-disc-like round part, not validated limits.

## YOLO inside the pipeline

Every casting/component inspection now also runs YOLO11n. A YOLO box is merged into a PatchCore anomaly as a class
hint only if at least 50% of the box lies inside the flagged anomaly region and the detector's own confidence is at
least 0.35. A merged crack-like class can raise severity (and can reach critical); unmerged YOLO boxes are shown as
"unconfirmed detections" and never count as defects.

### Why the merge is strict (measured, 2026-10-09)

YOLO11n was trained on NEU steel-surface patches. On the casting proxy test images:

- Full-image YOLO produced large, low-confidence "inclusion" boxes covering half the part, not the defect; the merge
  correctly declined them.
- Running YOLO on crops of PatchCore-flagged regions returned a class for 21 of 26 defect crops, **but also for 19 of
  20 random crops of normal parts**. On this domain its class names carry no information.

So on castings the merge normally attaches nothing, and severity relies on size and location. The merge becomes useful
when YOLO is trained on the same component domain (for example labelled brake-disc defects).
