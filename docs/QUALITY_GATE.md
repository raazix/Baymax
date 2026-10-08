# Explicit capture-quality profiles

`check_frame(frame, profile='camera')` retains camera default behavior and adds explicit `neu_proxy` and `casting_proxy` calibration. `profile_status()` reports readiness, version, and calibration SHA256. Unknown profiles or malformed frames raise `ValueError`; absent/corrupt calibration raises `QualityCalibrationUnavailable`. Metadata validates finite numeric thresholds, range/order constraints, dimension bounds, version, and profile identity. The cache uses actual file contents plus digest, so same-mtime replacements cannot carry stale provenance.

Run `.\.venv\Scripts\python.exe scripts/calibrate_quality.py` to reproduce local artifacts under `models/quality`. Calibration reads original **training images only**, never the detector's validation/test splits. NEU: 1,260 training images, hash-group 1,008 fit / 252 quality validation. Casting: 363 normal training images, hash-group 290 fit / 73 quality validation. Exact-byte duplicate hashes cannot cross the quality split. Physical-part identity and near-duplicate independence remain unverified. Every source path/hash and split is recorded in the artifact. Dataset originals are presumed usable, not operator quality-labelled ground truth.

Each proxy uses grayscale INTER_AREA resize to224x224. Blur checks combine Laplacian variance normalized by image variance and a multiscale energy ratio: Laplacian variance after Gaussian sigma0.7 divided by Laplacian variance after additional sigma3 smoothing. Initial single-scale analysis proved vulnerable to uint8 quantization after blur; multiscale denoising was added during development. Quality validation was inspected during algorithm development, so these are development validation results, not independent final quality-system certification. Threshold values are derived from fitting images' fixed robust quantiles, without selecting values from detector test results.

Fitting rules: lower1% multiscale quantile times0.9; lower0.5% normalized-Laplacian quantile times0.5; contrast lower0.5% quantile times0.7 (minimum2); mean exposure lower0.5% minus10 and upper99.5% plus5; clipping fractions upper99.5% plus0.1. Contrast and mean use0..255 gray units, ratios are dimensionless, fractions are0..1. These tolerances preserve textured or naturally bright proxy surfaces; white surface patterns cannot be identified as physical specular reflection from clipping alone.

| Development quality-validation outcome | NEU (252 originals) | Casting (73 originals) |
| --- | ---: | ---: |
| Original image accepted | 251/252 (99.6%) | 73/73 (100%) |
| Gaussian sigma3 rejected | 248/252 (98.4%) | 73/73 (100%) |
| Gaussian sigma5 rejected | 238/252 (94.4%) | 73/73 (100%) |
| Severe gain0.12 underexposure rejected | 252/252 (100%) | 73/73 (100%) |
| Severe +180 clipped exposure rejected | 237/252 (94.0%) | 73/73 (100%) |
| Uniform128 blank rejected | 252/252 (100%) | 73/73 (100%) |

Synthetic blur is applied at224px; exposure transformations clip back to uint8. Stronger blur can produce quantized contour energy, so measured rejection is not monotonically perfect. Some very dark originals become naturally bright-looking after the exposure perturbation; they remain indistinguishable without reference capture/lighting evidence. Rejected usable originals should be reviewed, not silently accepted. No passing image is guaranteed defect-visible, glare-free, or otherwise safe.

Proxy frames must retain calibrated source dimensions: NEU200x200 or casting512x512, with aspect ratio0.8..1.25. Profiles are requested explicitly, never selected automatically from appearance. The API additionally prevents applying a NEU-quality profile to a casting-model run and vice versa. Camera thresholds remain provisional Laplacian variance>=120 and grayscale bright-clipping fraction<=0.08; these were not lowered globally. Real webcam capture calibration still requires labelled sharp/blurred/exposure frames under actual optics and lighting.

`specular_fraction` remains a compatibility alias for fraction grayscale>=250. Responses name it `saturation_fraction` too and explicitly state it is **not physical specular reflection**. Proxy responses include normalized metrics, rejection reasons, thresholds, version, content SHA256, and camera-validation limitation. All outputs serialize with strict JSON; tests cover corruption, profile absence, unchanged camera behavior, hash splits, and identity replacement.
