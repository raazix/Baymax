# Frozen fine-tuned checkpoint: test comparison

The validation-selected rebalanced model was frozen before this evaluation. All 270 original test images were evaluated; no model, confidence or quality thresholds changed. This split was previously used to report the baseline, so these are reused comparison results, not a fresh blind test.

| Test metric | Original baseline | Rebalanced checkpoint |
| --- | ---: | ---: |
| mAP50 | 73.44% | 72.37% |
| mAP50-95 | 43.61% | 42.43% |
| Recall | 70.65% | 70.97% |
| Crazing AP50 | 40.52% | 34.60% |
| Crazing recall | 29.91% | 28.04% |

The fine-tuned checkpoint's validation improvements did **not** establish a test improvement. Its test mAP and crazing performance are worse than the baseline. The checkpoint remains the validation-selected candidate; test scores are reported separately rather than silently selecting another model on test feedback.

The live API also exercised all 270 test images at confidence0.25 and matching IoU0.50. The unchanged provisional quality gate accepted148 and rejected122. All148 inference results were checked against persisted records and exact image/model hashes. Conditional box precision was **62.39%**, recall **59.82%** among accepted images. These are not mAP, and they exclude292 ground-truth instances in rejected images. Invalid uploads returned422, oversized413, missing frames404. Median warm inference request was53.5ms, p9569.9ms, excluding the upload/quality request; this is a local measurement, not a latency guarantee.

Reports: `models/yolo/neu_yolo11n_rebalanced/test_comparison.json` and `data/model_api_test_rebalanced/report.json`. Visual evidence: `data/model_api_test_rebalanced/examples.jpg`, first sorted test image per class, with green ground truth and red predictions. No brake-disc validation, pixel-accurate masks or physical measurements are implied.
