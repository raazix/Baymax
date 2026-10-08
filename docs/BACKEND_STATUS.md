# Current demo coverage

This is a hackathon prototype. Feature completion does not establish production readiness.

| Pipeline stage | Implemented behavior | Remaining gap |
| --- | --- | --- |
| Camera/quality | Webcam capture UI, bounded raw-image upload, measured blur/clipping gate | Thresholds need camera-specific calibration; the current gate rejects many valid NEU patches |
| Calibration/polar/metrology | Deterministic geometry calculations exercised on synthetic rotor fixtures | Automatic marker/rotor detection and measurements from real segmentation masks |
| Detection | Trained six-class NEU YOLO11n bounding-box proxy, versioned inference storage | Genuine brake-disc labels, segmentation, SAHI integration and transfer validation |
| Unknown anomalies | Trained normal-only casting PatchCore variant, validation-selected threshold and persisted anomaly endpoint | High normal false-alarm rate; new calibration, brake-disc validation and cross-model fusion |
| Severity | Versioned deterministic demo rules, independent of confidence | Engineer-approved limits and genuine measurement uncertainty |
| Traceability | Parts, lot/machine references, telemetry, frames, runs and inspection snapshots | Live process data collection; PostgreSQL integration/migrations |
| Spatial fingerprint | Deterministic distribution features | Paired diagnosed histories for training RCA on spatial features |
| RCA/explanation | Trained XGBoost and exact TreeSHAP with pinned model/metadata hashes | Real diagnosed production labels, probability calibration and causal validation |
| Forecast | Trained PLSR primary; PCR benchmark; chronological synthetic evaluation | Production lot histories and real forecast validation |
| Uncertainty | 10,000 seeded residual-bootstrap simulations | Validated production residuals, parameter/input uncertainty and interval coverage |
| SOP/action | Cause-specific inspection steps, critical lot-containment recommendation, engineer transitions | Approved plant SOP documents and authenticated engineer identity |
| Verification/audit | Subsequent-part checks, sample-rate evidence, event/evidence integrity verification | Real corrective experiments and externally anchored audit/signatures |
| Async API | FastAPI, persistent queued jobs, retry and crash-recovery behavior | Distributed leases, operational monitoring and deployment hardening |
| UI | Next.js industrial dashboard; trained/legacy contribution display compatibility | Live operator workflow integration and AR |
| Optional intelligence | Deterministic evidence briefing with citations | LLM rendering and approved-memory retrieval are not integrated |

Validation: 52 automated tests passed. A live API and separate worker passed deferred analytics, approval, 20 synthetic re-inspections, briefing and audit verification (`data/analytics_api_test/report.json`). Full proxy API evaluations covered270 NEU and469 casting test images. The Next.js type check and production build passed. Synthetic verification is not evidence of a production fix.

Useful endpoints: `http://127.0.0.1:8000/docs`, `/api/models`, `/api/inspections/{id}/briefing`, `/api/inspections/{id}/audit`.
