# Optional SAHI YOLO inference

Installed and pinned SAHI 0.12.8. The trained six-class NEU YOLO11n detector remains a steel proxy; sliced inference does not establish brake-disc accuracy or produce segmentation masks. No accuracy improvement has been measured.

Full-image inference remains the default. Opt in through either endpoint:

```text
POST /api/proxy/frames/{frame_id}/detect?inference_mode=sliced&tile_size=512&overlap=0.2
POST /api/inspections/upload?model=neu&inference_mode=sliced&tile_size=512&overlap=0.2
```

Upload accepts a raw JPEG/PNG body. Parameters: inference_mode full/sliced, tile_size 128-2048 pixels, overlap 0-0.5. Requests above 64 tiles fail with 422 before tiled inference. SAHI mode on casting/PatchCore is rejected; existing PatchCore square-crop behavior is unchanged. Dashboard still uses default full-image mode; use API docs for sliced mode.

SAHI receives RGB converted from OpenCV BGR. It reuses the already verified loaded YOLO weights and the existing detector lock. Each tile uses model input size 640 and confidence threshold 0.25. A full-image pass is included for larger defects. Class-aware GREEDYNMM uses intersection-over-smaller (IOS) threshold 0.5, merges overlapping detections, and returns full-image-coordinate boxes clipped to image boundaries. Nearby same-class defects can be merged at this operating point; validate on labelled deployment images before adopting it.

Model-run evidence stores mode, SAHI version, tile size/count, overlap, merge settings, confidence and measured inference duration. Upload inspection context retains the same settings and links the persisted model run. Sliced mode generally costs more latency; it does not meet a verified 30-60ms end-to-end budget.

Validation: 91 backend tests pass. Tests cover tile-coordinate shifts, merging same-class duplicates without merging distinct classes, BGR/RGB conversion, clipping and workload/input limits. GPU smoke test on one quality-passed 200x200 NEU image: four 128px tiles, three final detections; full mode also returned three. Both API endpoints and persisted evidence passed, inspection audit valid. Functional report: ignored data/sahi-live-test.json. This is functional evidence, not an accuracy benchmark.

[SAHI source and documentation](https://github.com/obss/sahi).


## Dashboard
Select **Steel defects - YOLO11n**, then choose **SAHI sliced - small details** as YOLO inference. Open Inspection settings to choose tile size and overlap. Upload a NEU steel image. The saved inspection summary shows SAHI and tile count; detailed inference config is in the inspection context and model-run record. Follow-up verification reuses the original mode and tile settings. Dashboard full-image mode is the default.
