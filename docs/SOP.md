# Deterministic demo SOP engine

`actions.sop_engine.recommend(defects, analytics)` uses `demo-sop-v2` rules.
Any critical defect recommends quarantining the affected lot regardless of RCA
class, model availability or confidence. Other observed defects recommend holding
the affected part. RCA selects an inspection and calibration recommendation for
temperature control, pressure sensing/regulation, vibration/tooling or spindle
speed feedback. A nominal or unsupported hypothesis alongside a defect requests
manual investigation rather than asserting a tooling diagnosis.

Observed defects produce `required=true, status=pending`; no observed defects
produce `required=false, status=not_required`. A high synthetic next-lot fraction
can additionally recommend engineer review of increased sampling. Forecasts and
simulated predictive intervals are labelled synthetic supporting evidence, not
causal proof or authority for a machine change. No PLC connection exists. The
numeric forecast is a predicted defect fraction, not a calibrated failure
probability. Invalid or non-finite fractions are excluded from evidence JSON.

These demonstration SOPs require engineering validation and an approved process
specification before production use. Tests cover all supported hypotheses,
critical containment at every confidence level, manual review for nominal RCA,
forecast-only sampling, missing analytics, and strict JSON serialization.
