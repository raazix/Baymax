# LineGuard: tech stack, models and results

Every number below comes from the project's evaluation files. "Test" means the held-out test split, never used for training or for choosing thresholds.

## 1. Tech stack

| Layer | Tools | Used for |
|---|---|---|
| **Frontend** | Next.js 16 (Turbopack), React 19, TypeScript 5.9 | Dashboard, Presenter mode, all tabs |
| | HTML Canvas 2D | PatchCore heatmap and YOLO box overlays |
| | SVG | Monte Carlo histogram, SHAP and probability bars |
| | Three.js 0.186 | 3D rotor viewer (synthetic replay records) |
| **Backend API** | FastAPI, Uvicorn, Pydantic | REST API, strict request validation |
| **Database** | SQLAlchemy, SQLite (PostgreSQL configured) | Inspections, frames, model runs, decisions, hash-chained audit |
| **Image processing** | OpenCV, NumPy | Decoding, quality gate metrics, cropping |
| **Deep learning** | PyTorch 2.6 + CUDA 12.6, torchvision | PatchCore feature extraction on the GPU |
| | Ultralytics 8.3 | YOLO11n training and inference |
| **Classical ML** | XGBoost 3.4 | Root-cause classifier and exact TreeSHAP |
| | scikit-learn | PLSR and PCR forecasting, metrics |
| **Simulation** | NumPy (seeded RNG) | Monte Carlo residual bootstrap |
| **Integrity** | SHA-256 (hashlib) | Image fingerprints, model pinning, audit chain |
| **Testing** | Python unittest (73 tests), puppeteer-core with headless Chrome | Unit, API and browser end-to-end tests |
| **Hardware** | RTX 4050 laptop GPU (6 GB) | All training and inference, fully local |

## 2. Each model, its task and its results

### A. Quality gate (OpenCV, resolution-normalised metrics)

**Task:** reject blurry, blank, dark or blown-out images before any model sees them.

| Held-out training images | Casting gate | Steel gate |
|---|---|---|
| Good images accepted | **100%** | **99.6%** |
| Blurred (σ=3) rejected | 100% | 98.4% |
| Blurred (σ=5) rejected | 100% | 94.4% |
| Underexposed rejected | 100% | 100% |
| Overexposed rejected | 100% | 94.0% |
| Blank rejected | 100% | 100% |

### B. PatchCore (ResNet18 features, 1,024-patch memory bank)

**Task:** casting anomaly detection plus heatmap. Trained on 363 **normal** images only.

The test set is 469 images (78 normal, 391 defective).

| Measure | **Active model** (low false-alarm threshold) | Original threshold (best F1 on validation) |
|---|---|---|
| AUROC | **95.6%** | 95.6% |
| Precision | **100%** | 91.7% |
| Recall (defects caught) | **77.5%** | 96.4% |
| False alarms on good parts | **0.0%** | 43.6% |
| F1 | 87.3% | 94.0% |
| Accuracy | 81.2% | 89.8% |
| Inference (API median) | about 15 ms per image | |

The original threshold caught more defects but flagged 44% of good parts. The threshold was moved to a ≤5% false-alarm target on **validation** (3.8% achieved there) before the test set was scored.

**Custom few-shot PatchCore** (train on your own casting): same test set, normal images only, training takes 2 to 4 seconds.

| Normal images used | AUROC | False alarms | Recall |
|---|---|---|---|
| 60 | 91.9% | 2.6% | 38.9% |
| 25 | 88.3% | 0.0% | 22.5% |

### C. YOLO11n (Ultralytics)

**Task:** steel-surface defect detection, 6 classes (NEU-DET, 1,260 training images, rebalanced fine-tune).

| Overall (270 test images) | Result |
|---|---|
| Precision | **67.7%** |
| Recall | **71.0%** |
| mAP50 | **72.4%** |
| mAP50-95 | 42.4% |
| Model inference | about **3.4 ms** per image |

| Class | Precision | Recall | mAP50 |
|---|---|---|---|
| Scratches | 63.5% | **95.8%** | **92.8%** |
| Pitted surface | **92.2%** | 79.4% | 86.3% |
| Patches | 75.4% | 83.8% | 85.8% |
| Inclusion | 64.1% | 79.0% | 74.4% |
| Rolled-in scale | 60.7% | 59.8% | 60.3% |
| Crazing | 50.1% | **28.0%** | **34.6%** (weakest class) |

The original 30-epoch baseline scored 73.4% mAP50 on the same test set. The fine-tune improved validation but didn't generalise to test, so state that honestly if asked.

### D. XGBoost with exact TreeSHAP

**Task:** root-cause hypothesis from process telemetry (temperature, pressure, vibration, speed), with a per-feature explanation.

Evaluated on 1,200 parts from synthetic lots 400 to 499, a chronological hold-out.

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Nominal process | 95.6% | 90.8% | 93.2% |
| Thermal drift | 95.5% | 97.7% | 96.6% |
| Pressure instability | 95.1% | 97.5% | 96.3% |
| Tooling vibration | 100% | 98.8% | 99.4% |
| Speed drift | 98.5% | 100% | 99.2% |
| **Overall accuracy** | | | **97.0%** (macro F1 96.9%) |

TreeSHAP is exact: the contributions add up to the model's output, with near-zero residual.

### E. PLSR forecast (PCR as benchmark)

**Task:** predict the next lot's defect fraction.

| Model (chronological test, 360 rows) | Mean absolute error |
|---|---|
| **PLSR** (used) | **6.36 percentage points** |
| PCR (benchmark) | 7.29 pp |
| Persistence ("same as last lot") | 7.04 pp |

PLSR is about 9.6% more accurate than the naive persistence baseline.

### F. Monte Carlo simulation

**Task:** turn the forecast into a risk range.

- **Method:** 10,000 seeded draws resample 360 held-out forecast errors, giving a 95% range and the chance of exceeding the 50% tolerance.
- **Example:** nominal → range 5.7% to 34.9%, 0% chance of exceeding tolerance. Thermal drift → 24.5% to 53.7%, 7.4% chance.
- This is a simulation, not an accuracy metric, so it has no percentage score.

### G. Rule engines (deterministic, no ML)

- **Severity rules:** by defect class, never by confidence.
- **SOP engine:** recommends an action; an engineer must approve.
- **Verification:** 20 clean parts gives an upper bound of 13.9% on the defect rate (one-sided 95%).

## 3. Tools used per task

| Task | Tool or model | Key result |
|---|---|---|
| Image quality check | OpenCV gate | 99.6–100% good images accepted, ≥94% of bad ones rejected |
| Casting anomaly detection plus heatmap | PatchCore (PyTorch, ResNet18) | 95.6% AUROC, 0% false alarms, 77.5% recall |
| New-casting onboarding | Few-shot PatchCore | 91.9% AUROC from 60 good images in about 4 s |
| Steel defect detection | YOLO11n (Ultralytics) | 72.4% mAP50, 3.4 ms per image |
| Root cause | XGBoost | 97.0% accuracy (synthetic) |
| Explanation | Exact TreeSHAP | Exact feature contributions |
| Next-lot forecast | PLSR (scikit-learn) | 6.36 pp error, 9.6% better than baseline |
| Risk range | Monte Carlo (NumPy) | 10,000 reproducible simulations |
| Evidence integrity | SHA-256 audit chain | Any inconsistent edit is detected |
| Reliability | unittest, puppeteer | 73 tests plus 21 end-to-end checks pass |

## 4. Say these honestly

- Vision results are on **proxy datasets** (castings, steel), not brake discs.
- XGBoost and PLSR scores are on **synthetic** process data.
- PatchCore's 100% precision rests on only 78 normal test images, so it's a small sample.
- The NEU test set was also used for earlier baseline reporting, so it isn't a fully fresh blind test.
