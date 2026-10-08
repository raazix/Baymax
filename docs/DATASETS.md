# Dataset plan for Track 3

These datasets are proxies. None of the sources below provides calibrated brake-disc images paired with casting telemetry and verified root causes.

| Dataset | Use in LineGuard | Limit |
|---|---|---|
| [KolektorSDD2](https://www.vicos.si/resources/kolektorsdd2/) | First surface-defect localisation/segmentation baseline; 356 defective and 2,979 normal images | Industrial surfaces, not brake discs; do not invent the four LineGuard class labels. CC BY-NC-SA 4.0. |
| [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | PatchCore normal-only training and pixel-level anomaly evaluation | Different objects and textures; normal brake-disc data is needed for rotor deployment. CC BY-NC-SA 4.0. |
| [NEU-DET](https://faculty.neu.edu.cn/songkc/en/zdylm/263265/list/) | Small detection benchmark: 1,800 steel-surface images, six classes including scratches | Bounding boxes cannot train or validate pixel-accurate segmentation/metrology. Preserve original classes; confirm download terms. |
| [Casting product inspection](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) | Casting classification proxy for a quick normal/defective baseline | Inspect publisher annotations and license before use. Image labels alone cannot validate localisation, defect dimensions or crack classes. |
| [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom) | Separate process-to-quality modelling benchmark: 1,567 records, 591 features, missing values | Semiconductor data; anonymous variables are not casting temperature or pressure. No brake-disc or verified root-cause labels. CC BY 4.0. |

## Recommended first experiment

Use KolektorSDD2 for localisation and MVTec AD for anomaly detection. If time only allows one trained model, prioritise localisation. Keep brake-disc replay separate from proxy-model evaluation. Do not quietly render proxy detections as measured rotor defects.

Train YOLO segmentation only on masks/polygons converted to its segmentation format; boxes alone are insufficient. A binary defect mask is one `surface_defect` class, not evidence of a crack/porosity taxonomy. The included training command expects a correctly labelled dataset YAML.

Keep official held-out test splits; split augmentation descendants with their source. For self-collected images, split by physical part/lot, not frame. Calibrate images individually and validate dimensions against known measurements.

For forecasting, target future lot defective fraction. Shift targets forward, hold out later lots, and fit imputation/scaling/PCA only on training data. PLSR/PCR regression outputs are not calibrated probabilities by default. Report MAE and forecast calibration before calling them probabilities. XGBoost needs actual root-cause labels to learn RCA; without labels, feature attribution explains quality prediction rather than proving cause.

## Honest hackathon data pairing

Generate reproducible synthetic temperature, pressure, speed, vibration, machine, lot and timestamps to demonstrate the full workflow. Explicitly mark both synthetic relationships and evaluation metrics. Do not join unrelated public image and sensor rows and describe the result as an observed causal dataset.

## What to collect next

Normal and defective brake-disc photographs across lighting, physical parts and batches; per-image scale marker; defect masks reviewed by a domain expert; actual timestamped process logs; and engineer-confirmed incident/root-cause/action records. The demo severity thresholds require engineering validation.
