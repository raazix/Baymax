# PatchCore lower false-alarm operating point

The lower false-alarm threshold substantially reduces normal-image alarms **at
the cost of missing more defects**. It preserves the original normal memory bank
and ImageNet backbone; no retraining or changes to original reports occurred.

| Split | Original normal false alarms | Calibrated normal false alarms | Original defect recall | Calibrated defect recall |
|---|---:|---:|---:|---:|
| Validation | 32/78 (41.03%) | 3/78 (3.85%) | 382/390 (97.95%) | 307/390 (78.72%) |
| Reused test benchmark | 34/78 (43.59%) | 0/78 (0%) | 377/391 (96.42%) | 303/391 (77.49%) |

Calibration raises the threshold from **2.1565878391** to **2.3222723007**.
The decision remains strict `score > threshold`; equal scores are normal.
The predeclared objective was the lowest positive threshold allowing no more than
5% false alarms among validation normals. With 78 normals this permits at most
three alarms. The threshold depends only on validation-normal scores, so it
maximizes recall among thresholds meeting that empirical budget without using
test scores. Validation defect scores quantify the tradeoff after selection.
The 5% target is a configurable engineering **demo** setting, not an approved
production acceptance criterion.

Original artifact and manifest hashes are verified before deriving the new
artifact. Every saved evaluation score must match the frozen manifest's image
path, hash, label and split; duplicate or missing score records cause failure.
The derived artifact is written before computing reused test metrics. The test
benchmark was already inspected previously; these numbers are **not a new blind
evaluation**. Zero false alarms in 78 benchmark normals does not guarantee zero
population false alarms. Physical-part identity and near-duplicate leakage
remain unresolved. Neither operating point has brake-disc or production
validation, and anomaly distances are not probabilities.

Artifacts: `models/patchcore/calibrated_low_fpr/casting_proxy.pt`, `metadata.json`
and `evaluation.json`. Model SHA256:
`2e737624006144655e604614d8f8e2c0e78ec9d732d76cfb671150c6ab9e5ebd`.
The preserved bank tensor SHA256 is
`22b74949a9038d7a69ec4eae11bfd3077338e9aa4339ca1b4c45f0464942aef0`.

Run `.venv\Scripts\python.exe scripts/calibrate_patchcore.py --output models/patchcore/NEW_VERSION`
to create a new immutable operating point; an existing artifact is never
overwritten. Tests in `tests/test_patchcore_calibration.py` cover strict-boundary
ties, empirical budget enforcement, recall tradeoffs, zero distances and invalid
or single-class data.
