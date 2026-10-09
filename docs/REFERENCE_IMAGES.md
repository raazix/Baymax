# Brake-disc reference images

The dashboard shows **Brake disc · reference image, operator-declared part type** for exact uploads matching the 40 images registered from `Downloads/bottle/dents`. This is declared reference identity, not learned brake-disc classification. The label is independent of defect predictions, model confidence, quality rejection, severity, and heatmaps.

Other images and all camera captures show **Unclassified part**. A copy of a registered image matches regardless of its filename or location; changed image bytes (including screenshots, edits, and re-encoding) do not match. Browsers do not expose the original local folder path to the application.

The registry contains SHA-256 hashes only, in `models/reference_parts/brake_disc.json`. Matched inspections record `context.part_identity`, with `model_prediction: false`, and an audit event `reference_part_identity_matched`. Missing or malformed registries safely fall back to unclassified. Prior inspections are unchanged.

Use the usual **Upload image** control with a JPEG or PNG from the reference folder. Models still produce the actual findings and anomaly maps; an image that fails the quality gate requires recapture and does not receive a model heatmap. WebP/AVIF files are outside the existing JPEG/PNG upload contract.

To refresh the registry after adding images:

```powershell
.\.venv\Scripts\python.exe scripts/register_disc_references.py --folder "<path-to-your-dents-folder>"
```

The 2,000-row UTF-8 synthetic sensor CSV is at `data/synthetic_history/synthetic_sensor_history_2000.csv`. It has `timestamp,machine_id,sensor,value,unit,lot_id`, timezone-aware ISO-8601 timestamps, and has been validated with the backend CSV importer. Upload it in the dashboard's historical sensors panel with a synthetic-data source label.
