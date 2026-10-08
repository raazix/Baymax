# NEU-DET baseline evaluation

YOLO11n detection, 30 completed epochs on RTX 4050 (6 GB). Original six steel-surface labels. Seeded image/hash-group split: 1,260 train, 270 validation, 270 test. Best checkpoint selected on validation; final metrics below are held-out test results.

Overall precision: 67.7%; recall: 70.7%; mAP@50: 73.4%; mAP@50?95: 43.6%.

| Class | Precision | Recall | mAP@50 | mAP@50?95 |
|---|---:|---:|---:|---:|
| crazing | 55.2% | 29.9% | 40.5% | 18.0% |
| inclusion | 66.9% | 76.4% | 74.7% | 39.6% |
| patches | 76.1% | 82.4% | 86.0% | 55.3% |
| pitted_surface | 86.1% | 81.0% | 87.2% | 59.8% |
| rolled-in_scale | 55.0% | 59.8% | 58.1% | 33.4% |
| scratches | 67.2% | 94.4% | 94.1% | 55.5% |

Crazing recall is only 29.9%, so this baseline misses many crazing instances. Dataset contains defect images; false-positive behaviour on defect-free brake discs has not been established. No physical-part IDs are available for grouped production validation.

This is steel-surface proxy detection, not brake-disc validation, segmentation, calibrated metrology or production safety approval.

Batch/worker settings changed on checkpoint resumption using observed throughput; the schedule is included in the evaluation report. Training checkpoints completed successfully; a missing SciPy import interrupted optional plotting after training. SciPy was installed locally, training curves regenerated, and the saved weights evaluated separately without retraining.

Artifacts: `models/yolo/neu_yolo11n/weights/best.pt`, `evaluation.json`, `results.csv`, `results.png`, and `models/yolo/neu_yolo11n_test` diagnostic plots. `models/yolo/active.json` binds backend activation to the evaluated checkpoint hash.

API integration checked on training images, separately from held-out metrics: raw upload ? quality gate ? GPU inference ? persistent model-run evidence. A crazing example returned no boxes; a scratch example returned actual boxes. These are integration examples, not extra test metrics.
