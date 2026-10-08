# Backend contracts and operations

## Start

```powershell
python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

The API runs independently of Next.js. Swagger UI is `/docs`; machine-readable schemas are `/openapi.json`. Default storage is `data/lineguard.db`. Normalised tables cover parts, telemetry, risk/RCA, corrective actions, frames, model runs, revisions, jobs and audit events; inspection snapshots keep a reconstructable evidence payload.

Schema creation and additive backfill support the initial local scaffold. They do not replace a versioned migration tool for production schema changes. PostgreSQL is supported by configuration but has not been integration-tested here.

## Durable asynchronous analytics

`POST /api/replay` body:

```json
{"scenario":"thermal_drift","seed":42,"analytics_mode":"deferred"}
```

The inspection persists with `action.status=awaiting_analytics` and an `analytics_job_id`. Analytics do not run in this request. Launch a separate worker:

```powershell
python -m analytics.worker
```

`GET /api/jobs/{id}` returns queued/running/completed/failed. `POST /api/jobs/{id}/retry` requeues failed jobs. After a stopped/crashed worker, run `python -m analytics.worker --recover` to requeue interrupted jobs. Only recover when the previous worker is stopped. Run one worker for this scaffold; there are no distributed leases or heartbeat recovery yet.

Results are persisted before job completion. Recovered execution skips already committed analytics, preventing duplicate analytic events. Inline replay mode remains available for compatibility. With trained artifacts present, new replays use XGBoost/exact TreeSHAP and PLSR with Monte Carlo residual bootstrap. Training data are synthetic; outputs are not production-validated. PCR is a benchmark only. See [RCA](RCA.md) and [forecasting](FORECAST.md).

Each inspection pins the model and metadata hashes before analytics run. The worker refuses to substitute changed artifacts. Existing inspections pinned to `demo-heuristic-v1` retain the explicit heuristic implementation. If trained artifacts are absent when an inspection is created, the demo heuristic is clearly identified; configured but broken trained models fail with 503 rather than silently falling back.

Train reproducible synthetic analytics in the local environment:

```powershell
.venv\Scripts\python.exe scripts/train_xgboost.py
.venv\Scripts\python.exe scripts/train_forecast.py
```

TreeSHAP values are raw class-margin contributions, not percentage risk contributions or causal effects. RCA classifier probabilities are uncalibrated. PLSR predicts a defect fraction, not a calibrated probability of an event. Its simulated predictive interval is conditional on fixed telemetry and synthetic residual assumptions.

The `demo-sop-v2` engine maps process hypotheses to inspection/calibration steps, recommends lot quarantine for every critical finding independently of model confidence, and retains engineer approval. Synthetic forecasts support sampling review only. See [SOP rules](SOP.md).

The live API plus a separate worker has been exercised through deferred analytics, engineer approval, a 20-part synthetic verification sample, evidence briefing and audit verification. Reproduce with `.venv\Scripts\python.exe scripts/test_analytics_api.py`; this creates explicitly synthetic test records in the local database. The report is `data/analytics_api_test/report.json`.

## Image ingestion and proxy inference

`POST /api/frames` takes raw PNG/JPEG bytes, not multipart or base64 JSON. Maximum compressed size is 8 MiB, maximum dimensions 16 megapixels. Dimensions are checked before OpenCV decoding. Invalid images return 422; excessive size returns 413. Original image bytes and SHA-256 are persisted, including rejected quality captures.

`GET /api/frames/{id}` returns metadata; `/image` serves the original bytes. A failed quality gate does not invoke inference. Capture thresholds remain provisional.

Set `LINEGUARD_PROXY_WEIGHTS` to a trained NEU YOLO detection `best.pt` and optionally `LINEGUARD_MODEL_DEVICE=0` for CUDA. `POST /api/proxy/frames/{id}/detect` requires a quality-passed frame and loads the model lazily. Missing dependencies/weights return 503. Generic COCO models and incorrect class lists are rejected. Each successful run persists its image reference, predictions, model hash and proxy limitations; retrieve it at `/api/proxy/runs/{id}`.

An evaluated model can also be registered with `scripts/activate_proxy.py`. At backend startup, `models/yolo/active.json` supplies a workspace checkpoint path, expected hash, evaluation reference and device. Registered weights must match the evaluated hash. Environment configuration overrides the registry.

This endpoint returns NEU source-label bounding boxes. It does not assign rotor severity, create masks, claim physical size, or claim brake-disc validation. Actual rotor segmentation/SAHI remains pending.

`POST /api/proxy/frames/{id}/anomaly` runs the separate casting-proxy PatchCore variant. It enforces the quality gate and persists image/model hashes and a raw28×28 anomaly-distance grid. Set `LINEGUARD_MODEL_DEVICE=0` for CUDA. It does not fuse predictions from the differently trained NEU detector. [Results and high false-alarm limitation](PATCHCORE.md).

## Decisions and audit

`POST /api/inspections/{id}/decision` requires a pending action and an engineer name. Approve/reject/escalate are mutually exclusive; concurrent updates use an optimistic revision check and the local repository serialises same-process writes. Repeated decisions return 409. There is no automatic process-control operation.

`POST /api/inspections/{id}/verification` requires an approved action and at least 20 distinct subsequent quality-passed parts from the same machine and source. Repeated inspections of the same part do not count as independent parts. It records evidence IDs and observed sample rate. With zero defects, it also reports the exact one-sided 95% binomial upper bound; this assumes independent samples. Synthetic verification never proves a production fix.

Captured defects, scale, telemetry, identity, source, image hash and model provenance are immutable through repository updates. Events append in the same transaction as the result and action projections. `/api/inspections/{id}/audit` verifies an event hash chain and a digest of immutable inspection evidence. This detects inconsistent edits; it is not externally signed proof against a privileged editor who rewrites the whole ledger.

Optional `LINEGUARD_API_TOKEN` requires Bearer authentication for `/api/*` except health. Default local mode is unauthenticated. This shared token is not per-engineer identity, authorisation roles or electronic-signature compliance. The current dashboard does not send this token automatically.

## Retrieval and optional intelligence

`GET /api/inspections?limit=50&offset=0&lot_id=B127&machine_id=M-04` supports bounded paging and traceability filters. `/api/models` reports model/integration readiness; `/api/health` checks DB connectivity. Responses carry a server-generated request ID.

`GET /api/inspections/{id}/briefing` exposes a deterministic evidence packet with stable references and JSON pointers. No external LLM or memory service is called.

Future LLM role: render a plain-language explanation from this packet and retrieved, approved SOP documents. Require structured output and citations, validate every referenced evidence ID, preserve simulated/unverified labels, and fall back to deterministic content on errors. Severity, forecasts, approvals and model scores remain upstream facts.

Future Supermemory role: optional retrieval index over versioned approved SOPs and completed engineer-verified incidents. Scope retrieval by component, plant, machine, SOP version and verification status. Keep canonical records in PostgreSQL and cite their IDs. Indexing must be explicit, with an outbox, retries and deletion tracking; never silently transmit camera images or raw telemetry. Unverified RCA hypotheses must not become verified incident memories. No credential or live integration is required for the current backend.
