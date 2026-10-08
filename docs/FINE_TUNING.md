# Validation-only fine-tuning outcome

Two runs started from the original 30-epoch YOLO11n NEU checkpoint. The existing test set was not evaluated or used for selection during fine-tuning. These are validation metrics on 270 images, not new blind test results or brake-disc performance.

| Validation metric | Baseline | Selected rebalanced model |
| --- | ---: | ---: |
| Overall mAP50-95 | 43.05% | 43.15% |
| Overall mAP50 | 77.36% | 77.06% |
| Overall recall | 72.25% | 74.04% |
| Crazing AP50 | 45.46% | 48.05% |
| Crazing recall | 30.28% | 35.75% |
| Crazing AP50-95 | 16.74% | 18.56% |

The first conservative run stopped after 16 epochs. It improved overall mAP50-95 to 43.86% but reduced crazing AP50 to 40.00%; the predefined selection rule rejected it.

The second run completed 12 epochs with 1,680 training samples: 1,260 unique original training images, with each of the 210 crazing source images repeated three times. It used AdamW with initial learning rate 0.0003, mosaic disabled, batch 8, 640px input, two workers and seed 42. Repetition stayed entirely within the original training split; validation images and annotations remained unchanged.

Before either run, the selection rule required both overall validation mAP50-95 and crazing AP50 to exceed the baseline. The second candidate passed and was registered in `models/yolo/active.json`. Overall gains are small, mAP50 decreased slightly, and crazing detection remains weak. Repeated validation-based model selection does not provide independent proof of generalisation.

Selected weights: `models/yolo/neu_yolo11n_rebalanced/weights/best.pt`, SHA256 `1ee2e1428f35009ff097b8167085b76bc32d54adf6067c206eb7fa86caa537f8`. Reports and curves remain in both run directories. The original baseline test report remains an archived baseline benchmark; it does not describe the newly selected model. Source labels remain the six NEU steel-surface categories, without invented masks or brake-disc class mappings.

The backend was restarted after activation. A functional inference test uses the first quality-passed validation image in filename order, checks the active model hash and persisted result equality, and writes `api_smoke.json` in the selected run directory. It is not an accuracy benchmark. The current camera quality gate still needs calibration and was not altered to improve model scores.
