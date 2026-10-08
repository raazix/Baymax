# Track 3 alignment

Source: `Singularity_2026_Hackathon_Problem_Statements.pptx`, slides 15–19.

| Criterion | Weight | Scaffold coverage / next evidence |
|---|---:|---|
| Detection and localisation | 25 | Highlighted synthetic rotor fixture; next train/evaluate localisation on a real proxy dataset |
| Severity and root cause | 25 | Versioned deterministic demo rules, process linkage, ranked heuristic factors; next validated thresholds and trained RCA with defensible confidence |
| Predictive risk | 15 | Next-lot synthetic risk and simulated interval; next time-held-out PLSR and PCR comparison |
| Recommendation and dashboard | 15 | Inspection-to-action workflow, engineer decisions, lot/machine traceability |
| Technical robustness | 10 | Typed inputs, persisted evidence, image hash, state transitions, evidence validation; next grouped/temporal model evaluation |
| Innovation | 5 | Unknown-anomaly scenario and uncertainty flow; real PatchCore adapter pending |
| Demo and storytelling | 5 | Thermal drift replay → hold lot → approve → inspect subsequent sample |

## Suggested remaining 18 hours

1. 0–3h: run scaffold, choose/download one dataset, verify CUDA and labels.
2. 3–8h: train one small localisation baseline, preserve held-out evaluation, connect real inference.
3. 8–12h: train synthetic process models with time/batch-held-out tests, expose assumptions and model provenance.
4. 12–15h: integrate results, review error states, prepare image/model metrics.
5. 15–18h: rehearse thermal-drift story and record backup demo. Add AR only if core evidence is complete.

The brief allows historical/simulated data, but expects working models. The initial scaffold is not yet a complete hackathon submission.
