# Bottle camera heatmap

The view uses OpenCV.js to find a tall centered contour, then submits a lossless PNG crop from that exact webcam frame to the camera quality gate and PatchCore. It never uses the uploaded inspection's heatmap.

## Try it locally

1. Start the API and dashboard and open a casting inspection.
2. Choose **Bottle camera heatmap**, allow camera access, and center the full upright bottle against a plain contrasting background. Use localhost or HTTPS.
3. Captures are analyzed automatically, with a single request at a time, 2.5 seconds between completed requests and a 45-second request timeout.
4. The custom `bottle` model is selected when registered; otherwise the dashboard's selected model is used. You can change models in the viewer. All models remain experimental for your steel bottle.

## Registration and evidence

The PNG crop is hashed locally. Stored-frame and inference hashes, frame ID and model name must match before rendering. The grid is drawn at the crop's original coordinates with the source frame's camera-cover transform. Only above-threshold patches are colored. Quality failures prevent model inference. Failures clear the heatmap; closing or switching models aborts requests and ignores late results.

The view displays the **exact analyzed snapshot**, its crop outline, capture time, score, threshold and image hash prefix. It holds that snapshot until the next result. This is sampled camera analysis, not real-time 3D surface registration. A delayed result is never drawn over different moving pixels. The API stores crop images and model runs for traceability.

## Limits and validation

Contours estimate a 2D region, not bottle identity or 6-DoF pose. PatchCore grids are coarse feature distances rather than pixel segmentation masks. Background and reflections can trigger scores. Accuracy on the actual steel bottle needs representative normal training images and labeled held-out evaluation.

OpenCV is served locally at `/api/opencvjs`; its Promise export is awaited. Run `node scripts/test_bottle_heatmap.cjs` for regression checks covering crop bounds, camera-cover mapping, threshold transparency, invalid grids, image identity, quality gating and the request contract. These pass, along with a real API round trip on a local normal bottle image and the production build. Physical webcam operation and localization still require device testing.
