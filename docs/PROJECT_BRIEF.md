# LineGuard Project Brief

**Purpose:** Hackathon prototype for evidence-linked manufacturing quality inspection, root-cause analysis (RCA), future-risk review and corrective-action tracking (Singularity 2026, Track 3). Brake discs are the selected target; steel-surface and other metal-component datasets are proxies where labelled as such.

## Product flow

An operator uploads one image or captures it with a webcam/phone. LineGuard stores the image and provenance, checks image quality, runs available vision models, measures findings where calibration permits, assigns deterministic demo severity, attaches machine/lot context, computes RCA and forecast evidence, recommends an SOP and waits for an engineer decision. Decisions and follow-up verification are auditable.

A critical demo finding holds the animated production line, displays an alert and can play ElevenLabs speech. Approval does not itself restart the line; the operator must explicitly resume. This is a browser simulation. No PLC or factory equipment is controlled.

## Architecture and stack

- **Dashboard:** Next.js 16, React 19, TypeScript, IBM Plex Sans. Motion for React provides transitions; Three.js renders lifecycle/rotor scenes; Radix Dialog provides accessible sheets; Sonner notifications; Lucide icons; QRCode React phone pairing; OpenCV.js sampled camera tracking/heatmap.
- **Responsive UI:** desktop inspection workspace, mobile bottom navigation, full-screen camera flow, reduced-motion support, and on-demand 3D loading.
- **API:** Python, FastAPI, Pydantic, Uvicorn. Next.js forwards API requests to FastAPI.
- **Vision and analytics:** PyTorch, torchvision, Ultralytics YOLO, OpenCV, NumPy, scikit-learn, XGBoost, SHAP/TreeSHAP.
- **Persistence:** SQLAlchemy with SQLite by default and PostgreSQL via psycopg/DATABASE_URL. Tables cover inspections, parts, telemetry, model runs, jobs, actions, revisions and audit events. PostgreSQL has been exercised locally; production schema migrations remain future work.
- **Async analytics:** database-backed jobs and a separate analytics worker.
- **Phone demo:** local Next.js production server, Cloudflare temporary HTTPS tunnel, QR/link pairing and six-digit code. Signed Secure HttpOnly sessions protect phone access. Laptop, API and tunnel must stay running.
- **Configuration:** provider keys and database URLs stay server-side in ignored .env/environment configuration, never dashboard NEXT_PUBLIC variables or Git.

## End-to-end pipeline

1. **Image input:** bounded JPEG/PNG upload or webcam/phone capture; bytes and SHA-256 are recorded.
2. **Quality gate:** blur and bright/specular-pixel checks. Thresholds are provisional and need camera/lighting calibration. Rejected images remain evidence and do not proceed to inference.
3. **Model selection:** configured component-specific proxy model. The dashboard currently defaults to the MPDD metal-plate PatchCore proxy; this does not make it a brake-disc model.
4. **Vision:** YOLO11n trained on NEU-DET returns source-class bounding boxes. Compact PatchCore uses frozen ImageNet ResNet-18 layer 2/3 features and a normal-image memory bank to produce anomaly scores/maps. SAHI-style slicing is an inference method, not another model. MPDD and MVTec component-specific PatchCore artifacts also exist.
5. **Geometry/metrology:** calibration, polar-location and measurement helpers exist. Reliable physical dimensions require valid scale calibration and part geometry. An image-only outline is not a physical measurement.
6. **Severity:** deterministic rules separate severity from model confidence. Current thresholds are demo/proxy rules, not approved automotive limits.
7. **Traceability:** inspection, part, lot, machine, image/model provenance and historical sensor readings are represented. CSV imports power descriptive trends; imported readings are not yet joined into inspection forecasts.
8. **RCA:** synthetic process/spatial features feed XGBoost; exact TreeSHAP shows feature contributions. This is a ranked hypothesis, not proven cause; probabilities are uncalibrated.
9. **Forecast:** PLSR predicts a future defect fraction; PCR is a benchmark. Monte Carlo residual bootstrap supplies simulated uncertainty. Training history is synthetic; missing history uses documented assumptions.
10. **Action/evidence:** deterministic SOP, engineer approval/reject/escalate, verification against at least 20 distinct later parts, evidence packet and hash-linked audit history.
11. **Assistant/alerts:** optional NVIDIA Nemotron explains stored evidence; it cannot determine severity, approve actions or control equipment. ElevenLabs provides critical-alert speech and speech transcription. Supermemory retrieves precedents from approved action notes; PostgreSQL remains canonical storage.

## Models and data

| Model/data | Role | Scope and evidence |
|---|---|---|
| YOLO11n NEU-DET | Six steel-surface classes and bounding boxes | Trained/fine-tuned proxy. Rebalanced checkpoint improved validation crazing recall from 30.3% to 35.7%; available test report has mAP50 72.37%, mAP50-95 42.43%. Not brake-disc performance. |
| Casting PatchCore | Unsupervised anomaly scores from normal casting features | Casting proxy, not validated on brakes. |
| MPDD metal-plate PatchCore | Component-specific anomaly scores | Metal plate proxy; current dashboard default. |
| MVTec metal-nut PatchCore | Component-specific anomaly detection | Nut subset only: report says 81/93 defects detected and 0/22 normal false alarms. Not brake-disc metrics. |
| XGBoost + TreeSHAP | RCA category and feature contributions | Trained on synthetic process scenarios; not real factory causality. |
| PLSR / PCR | Next-lot defect-fraction forecast / benchmark | Synthetic history only; not calibrated event probability. |
| Monte Carlo | Forecast uncertainty and simulated breach estimate | Conditional on synthetic model/residual assumptions. |
| NVIDIA Nemotron | Natural-language explanation of saved evidence | Explanatory only; human review remains necessary. |
| ElevenLabs | Text-to-speech alert and speech-to-text questions | Backend provider calls; requires private keys/quota/network. Browser autoplay can be blocked. |

Available datasets include NEU-DET (1,800 labelled steel images), MPDD, MVTec AD component subsets and casting proxy images. No annotated brake-disc segmentation/class dataset has established reliable brake-disc accuracy.

The supplied Downloads/bottle/dents folder now contains **24 valid images**: correct (6), defects (6), rusted (12), with no exact duplicate hashes. There are only six candidate normal examples, below the PatchCore custom trainer minimum of 20 confirmed normal images. An earlier isolated experiment used three plausible normal references and produced exploratory scores only; it has no calibrated threshold and is not active. Inventory/contact sheet: data/raw/brake_disc_candidates/.

## Integrations and interfaces

- PostgreSQL/SQLite through SQLAlchemy and DATABASE_URL.
- NVIDIA Inference API for assistant answers/summaries.
- ElevenLabs TTS and STT.
- Supermemory retrieval and approved-action summary memory; no source images stored there.
- Cloudflare quick tunnel and QR pairing for temporary phone access.
- FastAPI OpenAPI/Swagger at /docs; dashboard proxies /api to backend.
- Model checkpoints live under models/. Large datasets, local DB and secrets are excluded from Git and must be restored on a fresh machine.

## AR and 3D status

Three.js renders the simulated production lifecycle and rotor geometry. The camera overlay uses 2D image tracking and can show a sampled approximate heatmap. It is not validated 6-DoF AR and has no verified camera-to-disc registration or physical surface anchoring. A bottle is a demo prop, not a brake-disc scan.

## Limitations and next work

1. Collect more brake-disc images with confirmed labels and preferably masks/boxes; split by physical part/source and evaluate on independent parts.
2. Provide at least 20 distinct confirmed normal images for the PatchCore workflow; 50–100 varied normal examples are a more useful start. Defect images are also needed for independent evaluation.
3. Establish camera/scale calibration and brake-specific metrology/severity criteria with engineering review.
4. Join real telemetry to machine/lot/inspection time and use ordered historical lots in forecasting.
5. Add the grouped affected-batch alert view required by Track 3.
6. Replace asserted engineer names with authenticated identities before real deployment.
7. Treat model results, forecasts and assistant prose as demo evidence until independently validated.

## Run locally

From the repository root, start the API:

    .venv\Scripts\python.exe -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000

In another terminal, start the dashboard:

    cd apps/dashboard
    npm.cmd run dev

Open http://localhost:3000. For the protected temporary phone demo, run:

    powershell -File scripts/start_phone_demo.ps1

Stop it with scripts/stop_phone_demo.ps1. See [README](../README.md), [Track 3 audit](TRACK3.md), [backend operations](BACKEND.md), [assistant integrations](ASSISTANT.md), [mobile demo](MOBILE_UI.md) and [handoff](../HANDOFF.md).
