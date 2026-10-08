# Forecast backend

PLSR is the primary next-lot defect-fraction regression. PCR is an offline comparison benchmark. These are actual fitted models trained on **synthetic process histories**, not production-validated brake-disc predictions or calibrated probabilities that an event occurs.

Train reproducibly from the workspace:

```powershell
.\.venv\Scripts\python.exe scripts/train_forecast.py --seed 2026
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_forecast.py -v
```

Artifacts are `models/plsr/forecast.joblib` and `models/pcr/forecast.joblib`; provenance and metrics are `models/plsr/training_report.json`. Only load trusted local joblib files.

The synthetic generator models correlated chronological temperature, pressure, vibration, and machine speed; a stipulated relationship produces a defective fraction from 250 simulated parts per lot. Input features comprise the current lot, the preceding lot, and a mean of two preceding lots. Targets belong to the next lot. No target defect fraction is used as an input. Rows split chronologically: 1,078 training, 360 calibration, 360 final test. Scaling and latent components fit training only. Calibration residuals are reserved for simulation, never sourced from the test split. Component counts were fixed before test evaluation.

| Model | Test MAE (fraction) |
| --- | ---: |
| PLSR (4 components) | 0.06362 |
| PCR (6 components) | 0.07289 |
| Previous observed lot baseline | 0.07040 |

MAE 0.06362 means approximately **6.36 percentage points** error on this synthetic held-out sequence. It does not establish real manufacturing accuracy. PCR performs worse than the persistence baseline and remains a benchmark.

`analytics.plsr_forecast.analyze_forecast(telemetry, seed)` returns `forecast` and `uncertainty`. Required telemetry: `temperature_c`, `pressure_bar`, `vibration_mm_s`, `machine_speed_rpm`. Optional `history` is a chronological list of preceding lot telemetry dictionaries; the latest two are used. Missing history is filled with current telemetry and explicitly reported as a persistence assumption. Missing artifacts raise `ForecastModelUnavailable`; invalid telemetry raises `ValueError`. Outputs include model version, file SHA-256, synthetic provenance, test MAE, training bounds, and an out-of-range flag.

Monte Carlo performs 10,000 reproducible draws of calibration residuals, adds them to the clipped primary prediction, and clips samples to [0,1]. The mean, central 95% **simulated predictive interval**, and probability of simulated defect fraction exceeding tolerance 0.5 all come from those same draws. The mean may differ from the point prediction due to residual bias and boundary clipping. The 0.5 tolerance is a demo configuration, not an engineering limit. It is not a confidence interval for parameters or a validated coverage guarantee. Assumptions: supplied process inputs are fixed; future errors are exchangeable with synthetic calibration errors; residual draws are independent. Input uncertainty, parameter uncertainty, distribution shift, and serial residual dependence are not modeled. Production use needs real ordered lot histories, engineering tolerance configuration, input uncertainty measurements, and forward validation of interval coverage.
