# LineGuard: implementation and comparison with existing solutions

## 1. What it is

LineGuard is a closed-loop quality-inspection system for automotive parts (Singularity 2026, Track 3). It goes beyond "is this part defective?" to "why, what happens next, and did the fix work?":

> **image → quality gate → defect model → severity → root cause → risk forecast → recommended action → engineer approval → verification → audit**

## 2. Architecture

```
Next.js + TypeScript dashboard (127.0.0.1:3000)
  Inspection console · Evidence & audit · Camera station
  Train PatchCore · Process what-if lab · Presenter mode
                  │  /api proxy
FastAPI backend (127.0.0.1:8000)
  vision/     quality gate, PatchCore, YOLO adapter, severity rules
  analytics/  XGBoost + TreeSHAP, PLSR / PCR, Monte Carlo, briefing
  actions/    SOP engine (recommendations only, no PLC)
  core/       upload pipeline, replay, schemas
  database/   SQLAlchemy repository, hash-chained audit
                  │
SQLite (PostgreSQL configured)     models/ (local weights, hash-pinned)
```

All models run locally on an RTX 4050 (6 GB); nothing goes to the cloud. Every model artifact is pinned by its SHA-256, so the system refuses to run if weights change silently.

## 3. The pipeline, stage by stage

### ① Ingestion (`vision/image_ingestion.py`)
- Accepts JPEG or PNG up to 16 megapixels and checks the dimensions before decoding.
- Fingerprints every image with SHA-256. That hash is the evidence identity and the root of the part ID.

### ② Quality gate (`vision/quality_gate.py`)
- Measures sharpness, contrast, brightness and clipping on a 224 px copy, so results don't depend on resolution.
- Thresholds were calibrated on each dataset's training images only.
- For images unlike the dataset, it waives the dataset-specific lighting and size limits, uses generic ones, and records the waiver. Sharpness is never waived.
- Bad images are rejected before reaching any model, so a model is never asked to judge a blurry photo.

### ③ Defect models

| Model | Task | Training | Result |
|---|---|---|---|
| **PatchCore** (ResNet18 features, 1,024-patch memory bank) | Unsupervised anomaly detection on castings | 363 *normal* images only | Test AUROC 0.956, 0% false alarms, 77.5% recall |
| **YOLO11n** | Steel-surface defect detection, 6 classes | NEU-DET, 1,260 images, then fine-tuned | Test mAP50 about 72% |

- PatchCore compares each image patch with its memory of normal patches. The largest distance is the score, and the 28×28 distance grid becomes the heatmap.
- Non-square images are sliced into overlapping square crops and stitched back together.
- **Custom training:** upload about 20 or more normal images of any casting and a new memory bank trains in 2 to 4 seconds. Its threshold comes from held-out normal images.

### ④ Severity (`vision/severity.py`)
- Fixed engineering rules by defect class. ML confidence never decides severity.
- Without real geometry, uploads can't be rated "critical".

### ⑤ Root cause (`analytics/xgboost_rca.py`)
- XGBoost classifies the process telemetry into five hypotheses: nominal, thermal drift, pressure instability, tooling vibration, speed drift.
- **Exact TreeSHAP** shows which reading pushed the prediction, for example "temperature +3.24".
- Results are labelled as a ranked hypothesis, not a proven cause.

### ⑥ Forecast and risk (`analytics/plsr_forecast.py`, `monte_carlo.py`)
- PLSR predicts the next lot's defect fraction, with PCR as a benchmark.
- Monte Carlo resamples 10,000 held-out forecast errors (seeded, so reproducible) to give a 95% range and the chance of exceeding tolerance.

### ⑦ Action, approval and verification (`actions/sop_engine.py`)
- Builds a recommended action from severity and the hypothesis: quarantine the lot if critical, otherwise hold the part plus a cause-specific check.
- An engineer must approve, reject or escalate. There is no machine control.
- Verification requires 20 or more later, quality-passed, distinct parts from the same machine. A clean sample of 20 still allows a defect rate of up to 13.9% (one-sided 95% bound), and the system says so.

### ⑧ Evidence and audit (`database/repository.py`)
- Every inspection, model run, decision and verification is stored.
- The audit events are hash-chained, so any inconsistent edit is detectable.
- A grounded briefing endpoint is ready for an optional LLM summary later.

**Dashboard:** heatmap and box overlays, the what-if lab, the custom training tab, and Presenter mode.

**Quality:** 73 automated tests, plus end-to-end API and browser tests.

## 4. Data, and what's real versus simulated

| Part | Data | Status |
|---|---|---|
| Vision | Casting impellers (1,300 images), NEU steel surfaces (1,800 images) | Real images, proxy domain (not brake discs) |
| Process analytics | 500 synthetic lots × 12 parts | Synthetic. In production, replace with real sensor history |
| Process readings on upload | Operator-chosen preset | Simulated and labelled as such |

## 5. Comparison with existing solutions

| | **Traditional rule-based vision** (classic machine-vision tools) | **Commercial deep-learning inspection** (e.g. Cognex, Keyence, Landing AI) | **Open-source libraries** (e.g. Anomalib, Ultralytics) | **LineGuard** |
|---|---|---|---|---|
| Defect detection | Hand-tuned thresholds, brittle to new defects | Strong, mature, production-validated | Strong models, no application around them | Real PatchCore and YOLO; proxy-validated only |
| Needs defect samples? | No, but needs manual rules | Usually labelled defects; some unsupervised modes | Depends on the model | **No.** PatchCore trains on normal parts in seconds |
| Root cause | No | Usually separate MES/analytics tools | No | **Built in:** XGBoost plus TreeSHAP explanation |
| Risk forecast | No | Typically separate SPC/MES software | No | **Built in:** PLSR plus Monte Carlo risk range |
| Closed loop (act → verify) | No | Partial; usually through MES integration | No | **Built in:** approval, 20-part verification, audit |
| Explainability | Rules are transparent | Mostly heatmaps | Heatmaps | Heatmap, SHAP, uncertainty ranges and stated assumptions |
| Evidence integrity | Varies | Varies by product | No | SHA-256 per image plus a hash-chained audit trail |
| Hardware and cost | Specialised cameras | Proprietary hardware and licences | Free | Free; webcam plus consumer GPU |
| Production maturity | High | **High**, certified, deployed widely | Library only | **Prototype**: no brake-disc validation, no real process data |

**Where LineGuard is stronger:** it joins detection, cause, forecast, action and verification in one evidence-linked loop. Existing tools usually split these across vision software, separate analytics and MES software, and manual quality procedures. It onboards new parts from good samples only, and states its uncertainty openly.

**Where existing solutions are clearly stronger:** proven accuracy on real production parts, certified hardware, high-speed line integration (PLC triggers, lighting control), support, and years of deployment. LineGuard is a prototype that shows the integrated approach.

> The vendor columns describe each category in general terms. No specific product was benchmarked, so treat them as typical capabilities rather than measured head-to-head results.

## 6. Limitations and next steps

1. Collect and label real brake-disc images, and add segmentation for true mm measurement.
2. Connect real machine sensors (temperature, pressure, vibration) in place of the synthetic process data.
3. Calibrate the camera and lighting at the inspection station.
4. Add authenticated engineers, PostgreSQL deployment and signed audit logs.
5. Optionally add an LLM briefing on top of the existing grounded-evidence endpoint.

## One-line pitch

"Existing systems tell you a part is bad. LineGuard tells you why, how likely the next lot is to be bad, what to do about it, whether the fix worked, and proves all of it with an audit trail."
