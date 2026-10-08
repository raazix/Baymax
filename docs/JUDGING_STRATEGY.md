# LineGuard: judging strategy

How to aim the demo at each judging criterion, ordered by weight, with what to show and what to say.

| Weight | Criterion |
|---|---|
| 25% | Functional prototype |
| 20% | AI/ML and integration performance |
| 15% | Technical |
| 15% | Problem–solution alignment |
| 10% | Innovation and features |
| 10% | Scalability and practical usability |
| 5% | UI/UX |

## 1. Functional prototype (25%): prove it runs live end to end

- Use **Presenter mode** and click through steps 1 to 4 without leaving the app: inspect, approve, verify 20 parts, audit trail.
- **Say:** "Everything you're seeing is live: real images, real models, real database, no mock-ups."
- **Backup:** warm up the models first, and keep a short screen recording in case of a failure on stage.

## 2. AI/ML and integration performance (20%): lead with honest numbers

- **PatchCore** (held-out test set):

  | Measure | Result |
  |---|---|
  | AUROC | 0.956 |
  | False alarms on normal parts | 0% |
  | Defects caught | 77.5% |

- **YOLO11n:** 72% mAP50 on 6 steel defect classes.
- **XGBoost:** 97% accuracy on its synthetic test set.
- **Integration:** one upload drives five models in sequence (PatchCore or YOLO → XGBoost → TreeSHAP → PLSR → Monte Carlo).
- **Speed:** about 15 to 70 ms per image on an RTX 4050.

## 3. Technical (15%): show the engineering depth in about 30 seconds

- **Training method:** PatchCore learns from normal images only, and its threshold was chosen on validation data, never the test set.
- **Explanations:** exact TreeSHAP explains each prediction.
- **Uncertainty:** 10,000 seeded Monte Carlo draws produce a risk range.
- **Evidence integrity:** every image gets a SHA-256 fingerprint and every step goes into a hash-chained audit trail.
- **Engineering:** quality gate, 73 automated tests, and a FastAPI backend with a Next.js dashboard.

## 4. Problem–solution alignment (15%): tie it to Track 3

- **Problem:** a defect is found, but nobody knows why, and the next lot repeats it.
- **Solution:** the system finds the flaw, finds the probable cause, and stops the next one, with an engineer always approving.
- **Show:** the same image under **Nominal** and then **Thermal drift**. The cause and the action change, which shows vision linked to process data.

## 5. Innovation and features (10%)

- **Closed loop:** detect, explain, forecast, act, verify, audit.
- **Train PatchCore tab:** a new casting can be onboarded from about 20 good images in seconds, with no defect data needed.
- **What-if lab:** live XGBoost, TreeSHAP and Monte Carlo.

## 6. Scalability and practical usability (10%)

- **Onboarding:** a new part needs only normal images, trained in seconds.
- **Inputs:** works on any image size or aspect ratio and handles webcam frames.
- **Architecture:** a modular API; the database can move to PostgreSQL (configured, untested); no PLC control, so it's safe to add to an existing line.
- **Roadmap:** real brake-disc images, live sensor feeds, camera calibration.

## 7. UI/UX (5%)

Don't spend time here. Presenter mode and the clear heatmap already carry it.

## Avoid these mistakes

- Don't claim brake-disc accuracy. Say "proxy datasets, real pipeline".
- Don't call the analytics real factory data. Say "trained on synthetic process data; plugs into real sensors next".
- If asked about weaknesses, mention three:
  - PatchCore misses about 22% of defects.
  - Few-shot custom models catch fewer defects.
  - "0/20 defective" is not proof of a fix.

  Volunteering these tends to raise technical scores.

## Suggested 7-minute order

| Time | Section |
|---|---|
| 30 s | Problem |
| 3 min | Presenter steps 1 to 4 |
| 45 s | Nominal vs Thermal drift comparison |
| 1 min | What-if lab |
| 45 s | Training a new casting |
| 1 min | Numbers and roadmap |

## Quick reference: Nominal vs Thermal drift

These are the two simulated **process context** presets. A photo can't tell you what the furnace was doing, so the operator picks one, and the screens label it as simulated.

| | Nominal | Thermal drift |
|---|---|---|
| Temperature | about 705 °C | about 758 °C |
| Pressure | 101 bar | 113 bar |
| Vibration | 2.1 mm/s | 4.1 mm/s |
| XGBoost cause | nominal process (about 99%) | thermal process drift (about 99%) |
| Next-lot forecast | about 20% defect fraction | about 39% |
| Chance of exceeding 50% tolerance | about 0% | about 7% |
| Recommended action | generic: hold and investigate | specific: check the thermocouple and temperature-control loop |

Upload the same defective image under each preset: the vision result stays identical, but the cause, risk and action change.
