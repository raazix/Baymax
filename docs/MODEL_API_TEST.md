# Frozen-model API test

Tested all 270 held-out NEU images through raw upload, quality gate, GPU inference and result retrieval. Original weights and confidence threshold were unchanged.

| Check | Result |
|---|---|
| Frames uploaded | 270 |
| Quality accepted / rejected | 148 / 122 |
| Persisted prediction runs verified | 148 |
| Box precision on accepted frames | 67.2% |
| Box recall on accepted frames | 61.3% |
| Median inference-request latency | 48.5 ms |
| p95 inference-request latency | 67.5 ms |
| Invalid image / excessive size / missing frame | Correctly rejected: 422 / 413 / 404 |

Precision/recall use confidence >= 0.25 and same-class greedy IoU >= 0.50 matching. They are not mAP, and are conditional on quality-passed frames. Latency covers local HTTP inference request, decode, model prediction and result commit; it excludes upload/quality and includes no network camera transport.

## Finding: quality gate over-rejects the proxy dataset

The fixed Laplacian variance >=120 and bright-pixel fraction <=8% thresholds reject 122 of 270 images: all 45 inclusion images, 35 scratches, 28 pitted surface and 14 patches. High recall on accepted patches does not establish performance on the excluded images. Gate thresholds need calibration on separate representative capture-quality data, for the actual camera, lighting, resolution and component. They were not lowered using held-out test feedback.

## Finding: defect misses remain

The frozen model misses many crazing regions. Example visuals are the first sorted test image from each class, without selecting successful detections. Green boxes are ground truth; red boxes are predictions. Dataset-image localisation cannot validate brake-disc safety or millimetre metrology.

The API integration is functional; production readiness is not established. Original held-out model mAP remains 73.4% at IoU 0.50 and 43.6% across IoU 0.50?0.95, evaluated outside the capture quality gate.

## Artifacts

- `data/model_api_test/report.json`: full per-image and per-class evidence.
- `data/model_api_test/examples.jpg`: ground truth versus predictions.
- `scripts/test_model_api.py`: repeatable integration test command.

Existing 12 backend tests also passed.
