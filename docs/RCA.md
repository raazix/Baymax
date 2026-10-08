# Synthetic RCA backend

`analytics.xgboost_rca.analyze_rca(telemetry)` predicts a ranked process hypothesis
from temperature, pressure, vibration and machine speed. It returns class
probabilities, exact TreeSHAP feature contributions, and hashed artifact provenance.
This is a **synthetic demonstration, not evidence of manufacturing causality**.

Train with `.venv\Scripts\python.exe scripts\train_xgboost.py`.
Artifacts and generated histories live in `models/xgboost/`.
The reproducible generator samples a latent process scenario for each lot, then
adds lot and part noise and gradual temporal drift. All parts in lots 0–399 train
the model; only future lots 400–499 evaluate it. Neither lot IDs nor latent labels
are input features. Test data do not tune model parameters.

The five hypotheses are nominal process, thermal process drift, pressure
instability, tooling vibration, and speed drift. They are generator scenarios,
not real diagnosed failures. Confidence is the **uncalibrated classifier
probability**. It is not a probability of causal truth. Results outside training
feature ranges are explicitly flagged. Spatial fingerprint features are deferred
until labelled histories link defect distributions to process outcomes.

TreeSHAP uses XGBoost's native exact `pred_contribs` implementation, with tree-path
dependent background semantics. Contributions are signed changes to the selected
class's raw margin before multiclass softmax. They are not percentages and do not
sum to the reported probability. Base value plus contributions reconstructs that
class margin. The response includes the additivity residual.

`metadata.json` records seed, feature order, whole-lot split, held-out classification
report, confusion matrix, log loss, training ranges, dataset hash and model hash.
A missing or modified model raises `RCAModelUnavailable`; no heuristic silently
substitutes for it. Set `LINEGUARD_RCA_MODEL_DIR` to select a different artifact
directory. The loader caches by model and metadata content hashes, so replacing
an artifact invalidates cached provenance and predictions.

Held-out seed-42 results: **97.0% accuracy**, macro F1 **0.9693**, log loss
**0.09198**, on 1,200 parts from 100 future synthetic lots. These measure recovery
of the synthetic generator's scenarios, not production diagnostic performance.

Run `.venv\Scripts\python.exe -m unittest discover -s tests -p test_rca.py` to check chronological
separation, reproducibility, unavailable-model behavior, invalid inputs and exact
SHAP additivity against the trained artifact.
