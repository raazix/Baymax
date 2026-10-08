# Track 3 requirements audit

Reviewed 2026-10-09 against `Singularity_2026_Hackathon_Problem_Statements.pptx`, slides 15-19, and current code/artifacts. This replaces the scaffold checklist. A working demo is not validated automotive performance.

## Required deliverables (slide 17)

| Deliverable | Coverage | Evidence and remaining gap |
|---|---|---|
| Detection, classification and localisation for one chosen component | Partial | Actual YOLO NEU steel-surface boxes and PatchCore industrial component anomaly grids work. Brake discs remain our selected target, but neither model is validated on them. The bottle is a demo prop. No validated brake-disc crack/dent/corrosion/porosity/deformation classifier or segmentation model. |
| Dashboard: highlighted image, type, location and Low-to-Critical severity | Partial | Image overlays, proxy classes, image coordinates, optional scale/geometry, deterministic triage and synthetic rotor severity work. Unknown anomalies require review; their score cannot identify a crack or establish critical severity. Physical severity rules are demonstration rules, not approved automotive limits. |
| Root cause with confidence and process linkage | Partial | XGBoost plus exact TreeSHAP provide hypotheses and uncalibrated classifier probability; confidence now appears in the dashboard. Images use simulated temperature/pressure/speed/vibration presets. Imported sensor records are not joined to individual inspections. Inspection time is not necessarily production time. Real causal confidence is not established. |
| Machine/batch future-risk view | Partial | PLSR, PCR benchmark and Monte Carlo produce synthetic next-lot forecasts. The forecast is an expected defect fraction, not calibrated probability that a machine produces any defect next cycle. Sensor trends are displayed separately; uploads do not use imported chronological history in their forecasts. |
| Recommended corrective action | Implemented for demo | Cause-specific deterministic SOPs, engineer approval, distinct follow-up inspections and auditable verification. No demonstrated real production prevention experiment. |
| Quality-alert summary for affected batches | Missing dedicated view | Individual inspection history and spoken alerts exist. There is no grouped batch alert showing unique affected part counts, severity, machines, cause, risk and action state. A production animation is not this deliverable. |

Slide 15 asks teams to choose **one component**, with brake discs as an example. Brakes are not mandatory in the brief, but remain the user's selected target. Do not quietly relabel proxy results as brake-disc validation.

## Evaluation rubric (slides 18-19)

| Criterion | Weight | Evidence / limitation |
|---|---:|---|
| Detection and localisation | 25 | Real proxy evaluation and overlays. NEU rebalanced test precision 67.65%, recall 70.97%, mAP50 72.37%, mAP50-95 42.43%. Crazing recall only 28.04%. This split was already used for baseline reporting; not a fresh blind test. These are steel-surface results, not brake-disc accuracy. |
| Severity and root cause | 25 | Severity separated from model confidence; replay distinguishes cosmetic scratch and critical crack. Proxy severity uses class and relative image size. RCA probability is explicitly uncalibrated, trained on synthetic processes. Factory-grounded severity/cause evidence is absent. |
| Predictive risk | 15 | Ordered synthetic lot train/calibration/test splits, PLSR versus PCR/persistence, residual-based simulations. PLSR test MAE about 0.0636 on synthetic defect fraction. Need machine/lot-specific imported history, trend-to-forecast linkage and defensible event-probability semantics. |
| Recommendation and dashboard | 15 | Image, defect, cause, forecast and SOP appear in console; approval/verification work. Confidence visible with limitation. Production animation demonstrates response but does not prove prevention. |
| Technical robustness | 10 | Image hashes, model provenance, input/quality checks, hash-chained audit, state tests and chronological process splits. Physical-part identities unavailable for NEU grouping; real camera/part severity and cross-domain generalisation unvalidated. |
| Innovation | 5 | Actual PatchCore, grounded NVIDIA assistant, Supermemory, ElevenLabs voice and sampled OpenCV camera analysis. Bottle tracking is 2D, not validated 6-DoF surface AR. |
| Demo and storytelling | 5 | Inspection -> probable cause -> SOP -> approval -> verification -> audit. Three.js lifecycle illustrates casting, machining, inspection, review and dispatch, with critical hold and voice/visual notification. Synthetic critical replay explicitly labeled. |

No completion percentage or predicted judging score is justified by this audit. The brief does not require AR, PLC control, a specific model family, or a 30-60 ms latency target.

## Model evidence and interpretation

- NEU: `models/yolo/neu_yolo11n_rebalanced/test_comparison.json`. Original classes: crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches. Do not substitute crazing for a validated brake crack, or pitting for validated porosity.
- MPDD metal plate: 27/71 defect images detected (38.0% recall), 44 missed, 0/26 normal false alarms. Not a validated brake-disc detector.
- MVTec metal nut: 81/93 defect images detected (87.1% recall), 12 missed, 0/22 normal false alarms, image AUROC 0.9892. These numbers describe nuts only.
- RCA: `models/xgboost/metadata.json`, `analytics/xgboost_rca.py`. Synthetic classifier probability does not establish causal confidence.
- Forecast: `models/plsr/training_report.json`, `analytics/plsr_forecast.py`. Missing prior lots use persistence assumptions. Monte Carlo intervals/tolerance breach estimates are simulated model outputs.
- Process import: `analytics/historical_sensors.py`, `apps/dashboard/app/HistorySensors.tsx`. Descriptive trends, not joined inference inputs.
- Severity: `vision/severity.py`. Physically scaled replay rules and conservative proxy rules have distinct scopes.

## Remaining priorities

1. Validate one component with honest labels, ground-truth localisation and class-wise errors. If brakes remain the target, obtain suitable brake-disc data; more proxy training cannot establish brake performance.
2. Join actual production time, machine and lot identity to imported telemetry, then feed ordered prior lots into forecasting. Keep demo sources explicit. Calibrate confidence/event probability against held-out outcomes before claiming factory probabilities.
3. Add required grouped affected-batch summary: unique part counts, classes/severity, machines, time window, evidence links and outstanding actions.
4. Rehearse the batch story with model evidence. Critical replay can demonstrate stop/voice/approval; do not describe it as a detected real bottle crack or proven prevention.

The 25 + 25 + 15 points for detection, severity/cause and history-backed risk should take priority over further AR extras.
