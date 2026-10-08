# LineGuard

Evidence-linked quality inspection scaffold for Singularity 2026 Track 3. Next.js + TypeScript dashboard, FastAPI backend, modular vision/analytics/action packages, and persistent SQLAlchemy storage.

## Run locally

From `D:\baymax`:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd D:\baymax\apps\dashboard
npm.cmd install
npm.cmd run dev
```

Open http://localhost:3000. API documentation: http://127.0.0.1:8000/docs. Next.js proxies `/api` to FastAPI; set `BACKEND_URL` in the dashboard environment if needed. This scaffold is intended for a local demo. Engineer names are recorded assertions, not authenticated identities.

## Demo walkthrough

1. Click **Demonstrate critical defect** in the production lifecycle scene (explicitly synthetic replay). Inspect highlighted synthetic geometry, dimensions, severity, telemetry, trained synthetic RCA explanation and simulated future risk.
2. Enter an engineer name and approve, reject or escalate. Decisions persist and repeat decisions are blocked.
3. After approval, replay 20 normal parts to exercise evidence-linked verification. This verifies demo criteria, not a real production fix.
4. Open **Evidence & audit** for the image hash, event history and complete inspection JSON.
5. Try unknown anomaly, cosmetic scratch and rejected capture scenarios.
6. Use **Scan with camera** for a webcam capture through the inspection pipeline. The bottle camera view supports sampled PatchCore heatmaps; it is 2D tracking, not validated 3D surface AR.
7. A critical result stops the Three.js production simulation and shows a notification with ElevenLabs audio. Approve as an engineer, then explicitly resume the simulation. No PLC commands are sent. See [production demo](docs/PRODUCTION_DEMO.md).

## Honest implementation status

Implemented: scenario fixtures, bounded JPEG/PNG ingestion, webcam quality gate, frame/model-run evidence storage, a lazy NEU proxy detector adapter, scale conversion, polar geometry, versioned demo severity rules, spatial features, seeded Monte Carlo, demo SOPs, normalised traceability storage, durable analytics jobs, guarded approval/verification transitions, audit integrity checks, optional API authentication, evidence packets, and a Next.js interface.

Trained and integrated: YOLO11n NEU proxy detection, compact PatchCore casting anomaly detection, XGBoost with exact TreeSHAP on synthetic process scenarios, PLSR next-lot fraction forecasting with PCR benchmark, and Monte Carlo residual bootstrap. Model artifacts and dataset provenance accompany results. Replay rotor detections remain fixtures; analytics training is synthetic, and confidence is uncalibrated.

Pending: validated brake-disc classes/segmentation and severity, production camera/threshold calibration, joined historical telemetry forecasting, grouped batch alerts, 6-DoF AR, authenticated approval and production validation. SAHI proxy inference and sampled camera analysis are implemented. `vision/adapters.py` defines model interfaces and fails explicitly when unconfigured. No brake-disc accuracy or latency guarantee is claimed.

SQLite is the default. PostgreSQL can be started with `docker compose up -d`, then set `DATABASE_URL` to `postgresql+psycopg://lineguard:local-demo-only@localhost:5432/lineguard` before launching FastAPI. Normalised part/telemetry/action tables accompany complete inspection snapshots. Additive backfill supports the initial local schema; versioned production migrations remain future work. Use one API process and one analytics worker for the demo. The current local API uses PostgreSQL; replay persistence, approval and alert retrieval were exercised against it. Configure credentials privately through DATABASE_URL, never in Git.

Backend is the current priority. See [backend contracts and worker operations](docs/BACKEND.md). Start deferred analytics independently with `python -m analytics.worker`. NVIDIA evidence-grounded explanations, Supermemory and ElevenLabs voice are integrated; they require private backend credentials. Critical speech reads stored evidence and does not ask an LLM to set severity or approve actions. See [Track 3 coverage and remaining gaps](docs/TRACK3.md).

## Your local datasets

A YOLO11n NEU proxy detector is trained and fine-tuned. The selected rebalanced checkpoint improved validation crazing recall from **30.3% to 35.7%**, with a small overall mAP50-95 improvement. It is active through `models/yolo/active.json`; the test set was not used for fine-tuning selection. [Fine-tuning results and limitations](docs/FINE_TUNING.md). The earlier baseline scored 73.4% test mAP50; that archived result does not describe the new checkpoint. [Baseline report](docs/NEU_BASELINE.md).

`casting_512x512.zip`: 1,300 casting proxy images (781 defective, 519 normal); no masks/boxes or process logs. Extracted under `data/raw/casting_512x512`. Brake discs remain the target component, per user instruction.

`NEU-DET.zip`: 1,800 steel-surface images with bounding-box XML annotations. Prepare YOLO **detection** data:

```powershell
python scripts/prepare_neu.py
```

Preparation validates image/box dimensions, corrects pairing despite missing `.jpg` in some XML fields, groups exact duplicates into a single split, and records a seeded manifest. No segmentation masks are fabricated. Random image-group splits cannot establish brake-disc transfer performance or physical-part generalisation.

Optional ML setup and training (PyTorch must be CUDA-enabled for GPU training):

```powershell
python -m pip install -r requirements-ml.txt
python scripts/check_gpu.py
python scripts/train_yolo.py --task detect --data data/processed/neu-det/dataset.yaml --epochs 30 --batch 4 --device 0
```

Training may download pretrained weights. Preserve NEU's six source labels. NEU boxes do not support pixel-accurate metrology or a trained brake-disc crack class. Use genuine polygon labels with `--task segment` for YOLO11n-Seg.

See [dataset recommendations](docs/DATASETS.md) and [Track 3 priorities](docs/TRACK3.md).

## Checks

```powershell
python -m unittest discover -s tests -v
cd apps/dashboard
npm.cmd run typecheck
npm.cmd run build
```


## Repository and local artifacts

Create a local environment first with `python -m venv .venv`. Dataset archives, trained binary weights, training runs, and the inspection database are excluded from Git. Preserve these on the training laptop; a fresh clone requires training or restoring the artifacts before real-model inference. See `HANDOFF.md` for current checkpoints and `docs/MPDD.md` for component training. In the dashboard choose **Component anomaly (PatchCore)**, select the matching MPDD component, and upload its image. These are component proxies, not validated brake-disc models.
