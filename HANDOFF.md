# LineGuard â€” project handoff and checkpoint tracker

Last updated: 2026-10-08T19:50:35+05:30. Workspace: `D:aymax`, PowerShell, Windows. Read this file first when continuing with Claude, then verify current processes/artifacts.

## User intent and constraints

### Latest change: camera heatmap alignment (2026-10-09)

Camera lag correction: the full-screen analyzed snapshot was covering the live video, creating a frozen-looking preview. The video now stays live at a requested 30 fps, with only the contour outline over it; the matching heatmap snapshot lives in a separate result canvas in the side panel. OpenCV work is capped at 10 Hz on a <=320-pixel-wide tracking frame, with kernel reuse, hidden-tab pausing and an overlay pixel-ratio cap of 1.5. Reduced tracking coordinates map back to full-resolution camera pixels before inference; source-resolution changes are handled. The result canvas is updated once per model result instead of repainted every animation frame. The regression script includes a reduced-to-native crop mapping check. Physical webcam smoothness remains to be confirmed on the user's device.

The bottle camera view now runs fresh PatchCore inference on a PNG crop from the exact OpenCV camera frame. It no longer receives or stretches the uploaded inspection grid. Local/stored/inference hashes plus frame/model IDs must match. The returned grid maps to the crop's original image coordinates and is rendered over the matching analyzed snapshot, with transparent below-threshold patches. Snapshots refresh automatically after sequential requests; each is labeled with capture time, score, threshold and image hash. This is sampled camera analysis, not 3D AR surface registration. Custom `bottle` is preferred if registered, with a selector to change models. Accuracy on the user's steel bottle is still unvalidated. Camera quality gate remains active; failures clear heatmaps, pending requests abort on close/model change, and late responses are ignored. Regression script: `node scripts/test_bottle_heatmap.cjs`; checks passed. Real API PNG/quality/inference round trip on a local normal bottle sample passed with matching hashes, 37x28 grid, score 2.726 and threshold 3.759. See `docs/AR.md`. Historical descriptions of stretching the uploaded grid below are superseded.

- Hackathon: Singularity 2026, **Track 3 automotive quality inspection**. The local PPT was reviewed earlier; see `docs/TRACK3.md`.
- Target remains **brake discs**. Casting component images and NEU steel-surface images are **proxy datasets**, never validated brake-disc evidence.
- User has webcam and RTX 4050 Laptop GPU, 6 GB VRAM. Initially iGPU-only; switching modes and restarting enabled CUDA.
- User initially had about 18 hours until deadline; remaining time must be reconfirmed rather than inferred.
- Use **Next.js + TypeScript**, not Vite. Familiar industrial dashboard. Current priority: finish a strong backend before further UI work.
- Use the local `.venv`; do not replace the installed CUDA PyTorch. User authorized downloads, training, and subagents for backend work. Do not ask again for routine authorized work.
- Never let ML confidence decide severity; deterministic engineering demo rules do. No PLC control. Engineer approval is required before action/verification.
- LLM/Supermemory were discussed but are optional and **not integrated**. Grounded briefing endpoint is ready; store canonical evidence in DB and do not index unverified hypotheses as verified incidents.
- Current user requests: **test fine-tuned model, then work on PatchCore**, and keep this handoff updated with context/checkpoints for Claude.

## Architecture and implementation truth

Vision: camera â†’ quality gate â†’ calibration/polar map â†’ slicing/known detection + unknown anomaly â†’ spatial merge â†’ metrology â†’ severity. Deep analytics: traceability/spatial fingerprints â†’ XGBoost + TreeSHAP (ranked association, not causal proof) â†’ PLSR primary / PCR benchmark â†’ Monte Carlo â†’ SOP â†’ engineer approval â†’ verification â†’ audit. AR is a later overlay, not part of inference.

There are two separate paths today:

1. **Synthetic rotor replay:** fixture detections/geometry, measured-size calculations using synthetic scale, deterministic severity, synthetic telemetry, trained synthetic analytics, action/audit workflow.
2. **Real proxy image inference:** persisted JPEG/PNG frames â†’ actual YOLO NEU bounding boxes. No fabricated masks, physical size or rotor severity. PatchCore casting proxy is trained and integrated through its separate anomaly endpoint.

Feature coverage estimates communicated: ~70% hackathon backend and ~50â€“60% whole proposed pipeline. These are rough feature estimates, not production readiness.

## Environment / commands

`.venv` was created with system site packages, but CUDA torch/torchvision/scipy/xgboost were installed locally. Torch `2.6.0+cu126`, torchvision `0.21.0+cu126`, Ultralytics `8.3.253`, XGBoost `3.4.1`. SciPy had broken global metadata; local install fixed it. Wheels cached in `data/wheels`; avoid deleting large caches without need. No git repository currently exists.

```powershell
$env:LINEGUARD_MODEL_DEVICE = '0'
.venv\Scripts\python.exe -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
.venv\Scripts\python.exe -m analytics.worker
.venv\Scripts\python.exe -m unittest discover -s tests -v
cd apps/dashboard
npm.cmd run dev
npm.cmd run typecheck
npm.cmd run build
```

API docs: `http://127.0.0.1:8000/docs`; dashboard: `http://localhost:3000` when started. At this checkpoint API PID **25636** (exec session 14315), worker exec session **81570**. Check actual command lines before stopping processes; never kill arbitrary Python processes. Dashboard dev server is not currently started. API was restarted after PatchCore integration with LINEGUARD_MODEL_DEVICE=0 (PatchCore normalizes this to cuda:0).

Default DB `data/lineguard.db` SQLite, SQLAlchemy normalised parts/telemetry/risk/actions plus snapshots, frames, model runs, jobs, revisions, audit events. PostgreSQL config exists, not tested. One API process and one worker; no distributed leases. API token optional `LINEGUARD_API_TOKEN`; engineer names are asserted, not authenticated identities.

## Checkpoints

- [x] Modular Python backend and Next.js scaffold
- [x] Bounded image ingestion and provisional measured quality gate
- [x] NEU exact-hash-group split and YOLO baseline training (30 epochs)
- [x] Fine-tuning: conservative candidate rejected; rebalanced candidate validation-selected
- [x] New checkpoint tested on all 270 existing test images and live API
- [x] XGBoost + exact native TreeSHAP trained on synthetic scenarios
- [x] PLSR primary / PCR benchmark trained on ordered synthetic lots
- [x] Seeded 10,000 residual-bootstrap simulations, honest interval assumptions
- [x] SOP v2, approval, subsequent-part verification, audit evidence hashes
- [x] Durable analytics worker, retries and crash recovery
- [x] 52 automated tests passed; Next.js typecheck and build passed
- [x] Live worker/API approval + 20 synthetic re-inspections + briefing/audit passed
- [x] PatchCore normal casting memory bank, validation threshold, test benchmark and persisted anomaly API
- [ ] **Next priority: independently calibrated camera quality gate and anomaly operating point; genuine brake-disc evidence**
- [ ] Camera-specific calibration, actual rotor/marker detection
- [ ] Brake-disc segmentation and genuine mask metrology
- [ ] SAHI integration and spatial merge
- [ ] AR, authenticated engineer identities, production migrations/deployment
- [ ] Optional LLM and approved-memory retrieval
- [ ] Real production/brake-disc validation

## Datasets and splits

- `casting_512x512.zip`: 1,300 casting impeller images, `ok_front` 519 and `def_front` 781, 512Ã—512, no masks/boxes/process logs. Extracted `data/raw/casting_512x512`. Prior exact-hash audit found no duplicates. Deterministic normal-only training and held-out validation/test are complete, with hashes recorded in data/processed/casting/manifest.json. Do not train memory bank on defective/validation/test images.
- `NEU-DET.zip`: 1,800 200Ã—200 steel-surface images with XML boxes, original labels `crazing`, `inclusion`, `patches`, `pitted_surface`, `rolled-in_scale`, `scratches`. `scripts/prepare_neu.py` corrected XML filename pairing and valid boxes, groups one exact duplicate within a split. Seed42, 1260 train / 270 val / 270 test. `data/processed/neu-det/dataset.yaml` and `manifest.json`. No polygon masks invented.
- Neither dataset has physical-part identity grouping; random image/hash groups cannot prove production generalisation.

## YOLO model state and honest results

Baseline: `models/yolo/neu_yolo11n/weights/best.pt`, SHA256 `f61fd25e5521e9ee78580b0d4e4671af0931aeb1b08b01296ea2e42d28fd49e9`. Original test mAP50 **73.44%**, mAP50-95 **43.61%**, crazing recall **29.91%**, crazing AP50 **40.52%**. Details `docs/NEU_BASELINE.md` and run `evaluation.json`.

**Active**: `models/yolo/neu_yolo11n_rebalanced/weights/best.pt`, SHA256 `1ee2e1428f35009ff097b8167085b76bc32d54adf6067c206eb7fa86caa537f8`, registered by `models/yolo/active.json`. Twelve-epoch rebalanced fine-tune, original training crazing images sampled three times (1,680 samples / 1,260 unique), mosaic off. Selection declared before training: validation overall mAP50-95 AND crazing AP50 must exceed baseline. Validation crazing recall **30.28 â†’ 35.75%**, AP50 **45.46 â†’ 48.05%**, overall mAP50-95 **43.05 â†’ 43.15%** (tiny gain). Conservative 16-epoch candidate `neu_yolo11n_finetuned` rejected because crazing AP50 worsened.

User then requested testing. New active test comparison: mAP50 **72.37%**, mAP50-95 **42.43%**, crazing recall **28.04%**, crazing AP50 **34.60%**. **Worse than baseline on this comparison**, already reported candidly. Keep validation selection separate from test reporting; do not silently select/tune on these test results. Test was previously seen for baseline reporting, so this is not a fresh blind holdout. No claim that validation improvement generalised.

Reports:

- `models/yolo/neu_yolo11n_rebalanced/evaluation.json`: validation selection, preserve it
- `models/yolo/neu_yolo11n_rebalanced/test_comparison.json`: reused test benchmark
- `data/model_api_test_rebalanced/report.json`: all 270 raw uploads, quality gate, frozen conf0.25/IoU0.50, exact hashes and persisted-result checks
- `data/model_api_test_rebalanced/examples.jpg`: first sorted test image per class, green GT/red predictions, no cherry-picking
- Baseline API reports preserved in `data/model_api_test`; fine-tune docs `docs/FINE_TUNING.md`

**Major quality-gate limitation:** current thresholds Laplacian varianceâ‰¥120 and grayâ‰¥250 fractionâ‰¤8% reject **122/270** legitimate NEU test patches; only148 reach inference. Thresholds were NOT changed to optimise test scores. API metrics on accepted frames are conditional and not overall mAP. Need calibration from independent training/capture data, not test feedback.

Completed rebalanced live API run:148 persisted inference results, conditional boxprecision62.39% / recall59.82%, median warm HTTPinference53.5ms/p9569.9ms excluding upload/quality. Details `docs/YOLO_RETEST.md`. This is a functional benchmark, not a manufacturing performance claim.

## Trained analytics and safety contracts

- `analytics/service.py` composes XGBoost + PLSR. Artifact hashes (including RCA metadata) pinned when creating inspections. Deferred worker refuses changed models; legacy `demo-heuristic-v1` inspections retain explicitly labelled heuristics. Missing trained artifacts at creation use explicit demo fallback; broken configured models fail closed.
- XGBoost `models/xgboost/{model.ubj,metadata.json,synthetic_history.npz}`, `scripts/train_xgboost.py`. 500 synthetic lots Ã—12 parts; chronological whole-lot train0â€“399/test400â€“499. Synthetic test accuracy97%, macroF1 .9693. This is synthetic scenario recognition, not manufacturing accuracy. TreeSHAP values signed raw class margins, NOT percentages. Probabilities uncalibrated and no causal confidence.
- PLSR `models/plsr/forecast.joblib`, PCR `models/pcr/forecast.joblib`, `scripts/train_forecast.py`. Chronological 60/20/20 split, 1078 train/360 residual-calibration/360test, preprocessing fitted only train. Synthetic test MAE PLSR6.36 percentage points, PCR7.29, persistence7.04. Predicts defect fraction, not calibrated event probability. Without two previous lots, current telemetry fills history and assumption is labelled.
- Monte Carlo independently resamples held-out synthetic calibration residuals conditional on fixed input, clips to[0,1], 10k seeded simulations. Simulated predictive interval, not confidence interval; excludes parameter/input uncertainty.
- `actions/sop_engine.py` v2: critical means lot quarantine independently of confidence/cause; class-specific inspection/calibration; nominal/unknown with defect means manual investigation. Forecast supports synthetic sampling review only; no PLC.
- Verify at least20 distinct subsequent parts, same machine/source, passed quality, after approval. Zero20 yields one-sided95% upperdefectrate13.91%, not production-fix proof.
- Immutable evidence plus append-only transactionally projected audit hash chain. Detects inconsistent edits, not externally signed tamper-proof ledger.
- `GET /api/inspections/{id}/briefing` deterministic grounded JSON pointers; no LLM/memory network calls.

## PatchCore completed state and next resume boundary

Both subagents finished. Runtime: `vision/patchcore_anomaly.py`; preparation/download/training: `scripts/prepare_casting.py`, `scripts/download_patchcore_backbone.py`, `scripts/train_patchcore.py`. Tests: `test_casting_split.py`, `test_patchcore.py`, `test_patchcore_api.py`. Documentation: `docs/PATCHCORE.md`.

Frozen compact variant: official ImageNet ResNet18 (no random backbone), uncropped resize224 and RGB normalization, pooled layer2/layer3 features aligned28x28, concatenated384dim. Train363 normal images only; validation78normal/390defect; test78normal/391defect. Exact-hash groups cannot cross splits; physical-part identities and near-duplicate checks remain unavailable. 284,592 normal patches reduced through a seeded50,000-candidate pool and64dim projected greedy selection to1024x384 bank. Exact chunked nearest-neighbour Euclidean distances; maximum patch distance image score. This variant omits canonical neighborhood score reweighting; no claim of exact canonical performance.

Artifact `models/patchcore/casting_proxy.pt`, SHA256 `492ac07823a95b40289d9e62717872e1d613200ea2c3ec628c29ecbb185e1b56`. Official backbone `models/patchcore/backbone/resnet18-f37072fd.pth`, SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`; local provenance records filename published hash prefix verification. Runtime does not download. Threshold **2.156587839126587**, strict `score > threshold`, chosen by validation F1 before saving/final test. Artifact format_version1; torch.load(weights_only=True), finite bank/threshold/provenance and workspace-local backbone hash validated. Original scores are distances, not probabilities. 28x28 grid is not a segmentation mask; no mm metrology.

Full test469: AUROC **0.9558**, precision **91.73%**, recall **96.42%**, F1 **94.01%**, confusion `[[44,34],[14,377]]` (actual normal/defect rows). **False alarms34/78 normal =43.6%**, a major limitation. Validation/test defect prevalence is83%, so F1-selected sensitivity is not representative of production costs. Do not lower/raise threshold based on this test; a new operating criterion needs new independent calibration/untouched test evidence. Reports `models/patchcore/evaluation.json`, `metadata.json` include raw scores/hashes, train-only coreset and split provenance.

API **POST `/api/proxy/frames/{id}/anomaly`**: existing frame, passed quality; missing404/rejected422/unavailable503. Persists into existing ModelRun table; **GET `/api/proxy/runs/{id}`** retrieves. Response pins image SHA and artifact SHA, threshold, distance grid, casting-domain warning, no severity/metrology. `/api/models` reports readiness. Do not fuse with the NEU detector (different training domains). Shared `LINEGUARD_MODEL_DEVICE=0` must normalize to `cuda:0` for Torch; a regression test covers the initial device-setting mismatch found and fixed during live testing.

Live full test:469uploads,327 accepted/142rejected by unchanged provisional quality gate. All327 results persisted and matched frozen benchmark scores within0.001 and exact hashes. Conditional accepted confusion `[[24,21],[11,271]]`, recall96.10%, normal false alarms46.67%; these exclude rejected images. Median inference HTTP14.5ms/p9520.2ms exclude separate upload/quality request. Reports `data/patchcore_api_test/report.json`, first sorted quality-passed examples `examples.png`. Functional performance is not a production latency guarantee.

Training CLI refuses existing artifact/report overwrites; for another experiment use `--output-dir models/patchcore/experiment_02`. API test also preserves reports; use `scripts/test_patchcore_api.py --output-dir data/patchcore_api_test_02` for a new run. Changing model artifacts requires API restart and independent evaluation. Original model comparison reports remain preserved.

Immediate next objectives: (1) choose an engineer-defined acceptable false-alarm/recall tradeoff and collect new calibration evidence; (2) calibrate image quality on independent real webcam captures (current gate rejects valid casting and NEU images); (3) collect/label brake-disc images and geometry; (4) integrate slicing/fusion only once models share the intended domain. Optional LLM/memory and AR remain later. Latest request is fully handled; no training job remains running.

References: https://github.com/amazon-science/patchcore-inspection and https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf.

## Important continuation rules

1. Read recent reports; preserve existing validation/test reports instead of overwriting them.
2. Do not use test feedback for threshold/model tuning or make proxy/fixture data look like genuine rotor findings.
3. Keep runtime dependencies/weights local and startup checks fail closed.
4. Restart API after active model changes; lazy-loaded detector keeps the old checkpoint otherwise.
5. Existing tests use temporary repositories; live test scripts intentionally add synthetic records to local DB.
6. Run meaningful tests after changes; last complete suite:52 passing, log `data/backend-test.log`.
7. Keep this handoff current after each checkpoint. End-of-session state: all current model testing/PatchCore/handoff objectives complete; API and analytics worker remain running.

## LATEST CHECKPOINT ? read before continuing

Updated: 2026-10-08T22:41:31+05:30. User requested fixing dataset quality rejection and PatchCore false alarms, then interrupted to request a quick handoff, then added MPDD and requested inspection/training. Preserve ALL THREE objectives; current task is inspecting MPDD while finishing integration checks for the calibration changes.

COMPLETED since previous checkpoint:
- Train-only named quality profiles (`neu_proxy`, `casting_proxy`) in models/quality; explicit camera profile remains provisional. Development quality holdout clean acceptance: NEU251/252 (99.6%), casting73/73 (100%). Severe corruption rejection NEU94?100%, casting100%; not real-webcam validation or an untouched quality benchmark (development holdout inspected during algorithm refinement). Six quality tests passed according to agent.
- Quality profile identities: NEU7c758249b545756d475be2d388688cd5ca3fa365163eb6d7f49b0d0dd1b851cf; casting60c6f197fd88bc59b47f363cb6b90038c42716be1e4034f88304e91aeb7a918a. Public functions check_frame(frame,profile='camera'), profile_status(), exception QualityCalibrationUnavailable. See docs/QUALITY_GATE.md.
- Low-FPR casting PatchCore calibration in models/patchcore/calibrated_low_fpr. Threshold2.322272300720215 selected only VALIDATION normals with<=5% false-alarm budget. Same trained bank/backbone. Validation3/78 alarms (3.85%) and307/390 defect recall (78.72%). Reused test0/78 alarms and303/391 recall (77.49%,88 missed defects). Original recall96.42%; this is a SUBSTANTIAL recall tradeoff, not a free accuracy improvement. See docs/PATCHCORE_CALIBRATION.md; five calibration tests passed.
- Derived artifact SHA2562e737624006144655e604614d8f8e2c0e78ec9d732d76cfb671150c6ab9e5ebd. models/patchcore/active.json EXISTS and registers it; activation completed even though the tool call was interrupted. Runtime supports registry expected-hash verification and environment overrides.
- API CODE now supports POST /api/frames?quality_profile=camera|neu_proxy|casting_proxy, explicit domain guard per detector,503 for absent/corrupt quality calibration,422 for invalid/tiny images, and persisted quality provenance for YOLO as well as PatchCore. /api/camera/quality remains camera-only. /api/models includes quality_profiles.

NOT YET VERIFIED / REQUIRED NEXT:
- Running API PID25636 likely still OLD code and loaded ORIGINAL casting threshold. Check process; restart with local venv and LINEGUARD_MODEL_DEVICE=0 before testing the new calibration/registry/profile flow.
- Run full unittest suite after the changes; last ROOT complete suite was52 tests BEFORE new calibration work. Agent-only tests do not substitute integrated validation.
- Run full NEU API check with --quality-profile neu_proxy into a NEW output directory. Run PatchCore API check with --quality-profile casting_proxy --benchmark models/patchcore/calibrated_low_fpr/evaluation.json and NEW --output-dir. Script supports original raw-score reference (same bank) and verifies calibration source hashes. Reports must preserve previous evidence.
- Add focused API checks for unknown/mismatched profiles, provenance persistence, unavailable calibration503 and registry hash mismatch fail-closed. Recheck analytics worker/approval/audit and dashboard build when changed contracts warrant it.
- Report new clean acceptance plus corrupted capture rejection and defect-recall tradeoff honestly. Camera remains provisional until representative webcam captures exist.
- Inspect newly added MPDD and train separately; no automatic cross-component pooling or brake-disc validation claims.

Current root listing shows only `Unconfirmed 543308.crdownload` (~53MB) as a new dataset candidate; its filename suggests a browser download may still be incomplete. Inspect archive completeness and file contents before training. Do not rename/move an actively downloading file. Both prior subagents finished their scoped work; forecast_backend subsequently hit usage limit. Root must verify available files, not rely on a still-running agent.


## FINAL MPDD / CALIBRATION CHECKPOINT ? 2026-10-08T22:55:56+05:30

MPDD.zip complete and CRC-checked during extraction:1,346 images,282 genuine defect masks,6 categories. Original test458 images retained untouched for threshold selection. Train888 normals split80/20 by exact-hash groups, no cross-split exact duplicates. Manifest:data/processed/mpdd/manifest.json. All6 separate PatchCore banks trained under models/patchcore/mpdd/<category>/casting_proxy.pt with evaluations and hashes. Thresholds calibrated from withheld training normals only at empirical5%FPR target; several models have poor recall. Full metrics:docs/MPDD.md. Connector92.86% recall; black21.28%, brown47.06%, white73.33%, plate38.03%, tubes34.78%. No pixel-metric, production or brake-disc validation claims.

Runtime now accepts mpdd_proxy metadata. API GET /api/patchcore/models lists6MPDD models; POST /api/proxy/frames/{id}/anomaly?patchcore_model=mpdd_connector (and mpdd_<category> for others) selects the matching component model. Artifacts pinned against evaluation hashes. Existing default casting model preserved. Six live model smoke checks PASSED:source/category/hash/score/persisted JSON verified, report data/mpdd_api_test/report.json. API restarted PID33736, execsession7187, GPU0; existing analytics worker remains.

Calibrated quality live checks COMPLETE:NEU267/270 accepted(previous148), casting469/469 accepted(previous327). Reports:data/model_api_test_calibrated/report.json and data/patchcore_api_test_calibrated/report.json. Low-FPR casting model active via models/patchcore/active.json:0/78 normal false alarms, recall77.49% (88 missed defects) on REUSED testbenchmark. Do not hide recall tradeoff. Current quality code includes generic camera-v2 and explicit open-domain fallbacks from other workspace changes; camera remains not physically calibrated. Preserve those existing changes including custom normal-image training, upload inspection, demo catalog and what-if analytics.

Final automated suite82 tests PASS (data/final-backend-test.log). All6 MPDD APIchecksPASS. Dashboard was not rebuilt in this checkpoint because no frontend changes were made. Remaining:independent real webcam/engineering calibration, more robust MPDD operating points using new validation evidence, real brake-disc dataset/segmentation, SAHI/fusion, AR. No training jobs remain running. This checkpoint supersedes earlier pending integration notes.


## Dashboard MPDD integration
All six MPDD component proxy models appear under Component anomaly (PatchCore) > Component. Matching-component and brake-disc validation caveats are displayed. Next.js production build and TypeScript checks passed. All six dashboard upload endpoint selections verified against their artifact hashes; local report: data/mpdd_dashboard_upload_test.json. Source repository excludes local datasets, binary model artifacts and inspection database; these remain on the laptop.

## Rounded dashboard refinement
Updated shared CSS radii, inset graphite navigation, spacing, inputs, evidence tables and mobile control stacking. Next.js production build passes. Desktop screenshot with live MPDD inspection reviewed; screenshots remain in data/browser-check/rounded-ui (ignored by Git). Mobile screenshots captured with Chrome CLI, whose window-size capture may clip at minimum browser width; exact device emulation unavailable in this session. No backend/model changes.


## Cleaner decision console
Default view shows image/finding, cause hypothesis, simulated risk and action. Native keyboard-accessible details controls preserve pipeline evidence, measurement notes, telemetry/TreeSHAP, forecast intervals, full recommendation, model status and history. Settings retain process preset/diameter and matching-component warnings. Pipeline jumps open relevant evidence. No LLM added: analytics/briefing.py already provides a deterministic evidence packet for an optional grounded explanation layer. SAHI is listed in requirements-ml.txt but not wired into YOLO inference; PatchCore uses local overlapping square crops only. Build/TypeScript pass.


## NVIDIA / ElevenLabs integration
Implemented read-only InspectionAssistant with text and microphone questions, transcript review, speech controls. Default NVIDIA Nemotron3Ultra550B-A55B (verified catalog and live inference), ElevenLabs Flashv2.5 + Sarah + Scribev2. Keys stored ONLY in ignored .env; do not echo or commit. core/config.py now loads root .env without overriding shell values, python-dotenv added. Full implementation/status/API/security/limits in docs/ASSISTANT.md. Responses save separate hashed evidence snapshots in data/assistant; original inspection audit untouched. Live explanation + TTS + STT roundtrip passed (data/assistant-live-test.json), 88 backend tests pass, dashboard build passes. API restarted PID34748/session99789 onGPU0. Real microphone hardware interaction remains user-browser check. LLM prose not guaranteed factual despite constraints/citation validation; engineer review remains required. No supermemory or PLC integration added.


## SAHI backend completed
SAHI0.12.8 installed without altering torch/ultralytics and pinned in requirements-ml.txt. Optional inference_mode=full/sliced on NEU proxy detection and upload endpoints; defaults full. tile_size128-2048 (default512), overlap0-.5 (default.2), max64tiles. Reuses verified loaded model; BGR->RGB; class-aware GREEDYNMM/IOS.5 with full-image pass. Settings saved in model-run and upload context. PatchCore untouched, SAHI+casting422. All91 tests pass; GPU4-tile smoke, both endpoints, saved evidence and valid audit passed (data/sahi-live-test.json). docs/SAHI.md includes usage/limitations. API PID30840/session3084/GPU0. Dashboard remains full-image mode; no accuracy gain or latency claim. This supersedes older SAHI-pending notes.


## Dashboard SAHI selector
Dashboard offers full-image (default) and SAHI sliced choices when YOLO11n selected. Tile sizes 256/384/512/768/1024, overlaps10-50%. Selected settings sent on raw-image inspection upload and shown in resulting part strip; calibration/settings collapse has tile-count warning. Follow-up images reuse original inference settings. Live localhost upload verified sliced with tile256/overlap.3, persisted model run and valid audit; an image that fit in one tile still records sliced mode. Production Next build and TypeScript pass. Test report data/dashboard-sahi-test.json.

## 2026-10-09 image-first / sensor-history checkpoint

User target: mixed metal plates and brake discs; only one image should be needed. Dashboard upload now hides model/process/scale choices, defaults to `mpdd_metal_plate`, and runs quality gate -> MPDD metal-plate PatchCore proxy -> measurement/triage -> synthetic-input analytics -> automatic NVIDIA evidence summary when configured. High/critical proxy-rule findings request ElevenLabs speech automatically when configured. Human approval remains required; LLM is explanation-only, never sets severity, disposition, or approval. The app labels a round outline as a geometric hint only, not semantic part identification. The same MPDD metal-plate model is currently applied to both; no brake-disc-trained model exists. This is a demo workflow, not a validated brake-disc inspection.

Neon PostgreSQL is read from ignored root `.env` via `DATABASE_URL`; 12 tables bootstrapped remotely, with 0 historical sensor files/readings as of this checkpoint. New `sensor_datasets` and `historical_sensor_readings` tables support validated, idempotent CSV import (UTF-8, ISO timestamps with timezone, max 5 MiB / 50k rows), plus catalog/trend API and dashboard tab. CSV fields: `timestamp,machine_id,sensor,value,unit` and optional `lot_id`. Imported history is kept separate from synthetic per-inspection telemetry; forecasting does not yet consume imported history. Do not commit or print `.env`; rotate the Neon password because its connection URI was pasted into chat.

Severity proxy rules v2 use class plus normalized detection-box area/side fraction for known NEU classes. PatchCore remains `anomaly_unclassified` -> `review_required`; distance scores are not probabilities and do not set severity. Triage thresholds are provisional, image-relative, and not production-calibrated. MPDD metal_plate test recall remains 38.03% (44/71 defect images missed) despite AUROC 1.0; threshold is selected on held-out normals for low false alarms, not tuned on test defects. Do not claim precision/safety for brake discs.

Optional Supermemory module searches prior approved-action notes for assistant context and stores only engineer-approved action summaries (no source images). It is enabled from the ignored local `.env`; authenticated read-only search returned HTTP 200. No test memory was written. The API key was pasted in chat; rotate it after the demo. NVIDIA auto summary uses evidence citations and marks historical notes as non-evidence. ElevenLabs speech is configured. AR is still not implemented: current Three.js rotor is replay-only and not registered to a physical object; real overlay needs disc-specific images, physical scale/geometry and camera-to-part registration/marker calibration.

Validation completed: `python -m compileall -q analytics core database apps/api vision`; Next.js production build / TypeScript; live API health (`database=postgresql`), sensor catalog, MPDD catalog, dashboard HTTP 200 and image-first UI text. No sensor rows or sample inspections were inserted. App services running on localhost:3000 and localhost:8000. Current source commits pushed as the configured project owner: `48e7407` (history/severity/assistant) and `ece63d8` (image-first UI); subsequent shape-hint copy edits need a commit/push.

Dataset leads to review/send:
- Direct visual brake-disc: Mendeley `Thermal Brake Disc Fissures` (160x120 railway thermal images, three fissure classes and masks; license CC BY-NC-ND 4.0; limited resolution/domain).
- Direct commercial lead: Mobiusi `Brake Disc Detection Dataset` (listed 5k JPG/PNG/JSON, defect type/location/severity, proprietary paid/commercial AI training terms; get access/license before use).
- MPDD source README has >1k industrial metal-part images with pixel masks and a download link; useful surface-proxy benchmark only.
- MVTec AD is a general anomaly benchmark (>5k images, 15 categories, pixel masks); not brake-specific and non-commercial license.

Next: ingest licensed brake-disc data when user sends it; make a labeled split grouped by physical disc/lot; train/evaluate disc-specific model and sensitivity-focused operating point without tuning on test; implement domain routing only after measured data; then register validated 2D/3D findings to camera for AR. User said mixed part types, so don't silently treat geometric roundness as definitive material/part classification.


## 2026-10-09 thermal brake-disc dataset inspection

User added root `dataset.zip` (268,325,348 bytes). ZIP CRC check passes. Archive has `dataset/images/` and `dataset/masks/`, 27,700 files each (55,400 files; about 289.5 MB uncompressed). All 27,700 image/mask filename stems match one-to-one; image files include 1,764 original-looking PNGs and 25,936 JPEG augmentation variants. There are 998 numeric source-ID groups (001–999, with missing IDs); keep every variant under the same numeric ID in one split to prevent leakage. No train/validation/test folders, class metadata, or README are included. Samples are 256x256 RGB images with grayscale masks; sampled masks use values 0/1, so a loader must threshold `> 0` or rescale masks for visualization.

The Mendeley listing describes raw thermal brake-disc frames at 160x120 and three fissure classes (penetrating, incipient, superficial), while this local archive is 256x256 and exposes only binary masks. Treat it as a processed/augmented fissure-segmentation set; do not infer the three subclasses from filenames. It can support a separate thermal fissure segmentation model (e.g. U-Net), but it is not a YOLO class dataset and is unrelated to the RGB MPDD metal-plate PatchCore bank. No training or active pipeline change was made. Need source/class mapping or original data before reporting class-specific severity; split by numeric ID and preferably use the base captures with on-the-fly augmentation.


## 2026-10-09 brake-disc dataset sourcing decision

User rejected the local thermal fissure archive for the target project and declined paid Mobiusi data. Keep both out of training and active inference. The public GitHub repository `susie1214/Vision_Inspection` describes burr/scratch/hole labeling for brake-disc production, but its visible repository listing does not establish that a labeled dataset is actually included; do not treat it as a ready training dataset without checking the actual files and license. The images.cv "disk brake" set appears to classify the component itself rather than surface defects, so it is not suitable for defect detection. FaultSeg on Zenodo has labeled train-wheel defects including cracks/scratches, but is only a domain proxy and not automotive brake-disc data. No verified, free, directly applicable brake-disc defect dataset was identified in this review. Continue using MPDD metal plate strictly as the labeled proxy demo, clearly disclose its 38.03% held-out defect-image recall and its lack of brake-disc validation, and seek permissioned brake-disc images with instance masks/class labels or collect a small target-specific set.


## 2026-10-09 3D heatmap preview

Dashboard now shows a Three.js procedural-rotor preview for uploaded PatchCore inspections. It projects the PatchCore anomaly-distance grid into a rotor-shaped translucent color layer (blue=at/below the PatchCore threshold, amber/red=above) and provides a heatmap visibility toggle. This static preview is illustrative only: the grid is not a pixel mask and the rotor is procedural.

The former AR.js/Hiro-marker prototype has been replaced with a markerless bottle overlay demo. Open **Try bottle AR demo** from an uploaded casting inspection and allow webcam access. Browser-side OpenCV.js finds a tall centered contour, smooths its bounding region, and draws a translucent heatmap grid over it. No printed marker is required. The runtime is served locally from the installed npm package by `/api/opencvjs` (about 13 MB before compression). TypeScript, production build, zero-vulnerability production audit, and HTTP 200/runtime payload smoke check pass. A live webcam/steel-bottle lock test remains to be done on the user's device. See `docs/AR.md` for setup and limits.

The bottle contour is only a 2D screen-space estimate, not bottle recognition, 6-DoF pose tracking, or a reconstruction of its curved surface. The heatmap comes from a separate inspection and is illustrative; it is not registered to physical bottle defects. The current PatchCore grid is coarse, not a pixel mask; model is MPDD metal-plate proxy with poor recall and no brake-disc validation. This is a visualization demo, not production inspection. Do not use for safety/acceptance decisions.

### AR blank-screen fix

Latest bottle OpenCV startup correction: `@techstark/opencv-js` 5 exports a Promise, including as the browser global `window.cv`. Await that Promise before checking/using `Mat`; polling `window.cv.Mat` alone causes a false initialization timeout. The loader now resolves the runtime Promise and reports actual rejection errors. `VideoCapture.read` also needs a preallocated `CV_8UC4` Mat matching explicit video width/height; fixed both. Runtime Promise resolution and grayscale conversion were exercised locally; production build/TypeScript pass. Physical webcam tracking still requires device testing.

User reported nothing appeared in the live AR view. Root cause: AR.js appends its webcam video directly to `document.body` at z-index 1000 while the dashboard AR modal was at z-index 999, covering the renderer and controls. Raised the modal stacking level above the video and gated tracker updates until `ArToolkitContext.init` completes, preventing update calls before calibration/controller initialization. Production build passes; local camera and marker-lock behavior still need a browser/device check after refresh.


## 2026-10-09 MVTec archive integrated

User added `archive.zip` (5.27 GB uncompressed). Inspected archive metadata/readme/license: MVTec AD 2019 with 15 categories, train/good, test/defect folders, and pixel masks. License is CC BY-NC-SA 4.0; commercial use requires MVTec permission. Do not commit archive/extracted data/model weights.

Used only `metal_nut` as a separate benchmark (not brake disc). Extracted 430 category files to ignored `data/processed/mvtec_ad/metal_nut`; trained local PatchCore custom model `mvtec_metal_nut` on 176 train normals with 44 deterministic held-out train normals; official test set kept untouched until evaluation. Artifact SHA-256 `e9054240bd9d5f963560fc446e244d9736676710fb019fc6d84e85058bdc250e`; threshold 2.5225. Official test results: image AUROC0.9892; 81/93 anomaly recall (87.1%), 12 missed, 0/22 normal false alarms; defect recall bent24/25, color19/22, flip23/23, scratch15/23. Interpolated64x64 patch-grid pixel AUROC0.9726, not native segmentation accuracy. Local report: `data/processed/mvtec_ad/metal_nut/evaluation.json`; model/artifact registration appears in `/api/patchcore/models`.

Added repeatable train/evaluate scripts and docs/MVTEC.md. Dashboard advanced selector exposes MVTec model; default remains MPDD metal_plate. 3D rotor/AR panel is hidden for MVTec metal_nut, because its heatmap cannot be honestly registered to brake rotor geometry. This benchmark does not imply brake-disc performance.
