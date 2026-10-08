# Casting-proxy PatchCore baseline

An actual normal-only memory-bank anomaly detector has been trained on the local `casting_512x512.zip`. It uses a frozen, official ImageNet-pretrained ResNet18. This is a **casting component proxy**, not validated brake-disc inspection. No random backbone was used and inference never downloads weights.

Reproduce from the workspace `.venv`:

The fitted baseline already exists. Another experiment must use a new `--output-dir`, such as `models/patchcore/experiment_02`; the trainer refuses to overwrite existing artifacts/evaluations. Set `LINEGUARD_PATCHCORE_WEIGHTS` to another experiment's artifact only after review.

```powershell
.\.venv\Scripts\python.exe scripts/prepare_casting.py
.\.venv\Scripts\python.exe scripts/download_patchcore_backbone.py
.\.venv\Scripts\python.exe scripts/train_patchcore.py --device cuda:0 --batch-size 8 --seed 42
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_casting_split.py -v
```

Official source: https://download.pytorch.org/models/resnet18-f37072fd.pth. Download size is 46,830,571 bytes. Verified official filename hash prefix `f37072fd`; full locally recorded SHA256 is `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. Download provenance is recorded under `models/patchcore/backbone/provenance.json`. The prefix is the official published integrity reference; the full digest is a local record rather than a separately published full-checksum assertion.

`data/processed/casting/manifest.json` lists every image's workspace-relative path, label, split, and SHA256. Exact-byte duplicate groups stay in one split; contradictory duplicate labels are rejected. Counts:

| Split | Normal | Defective | Purpose |
| --- | ---: | ---: | --- |
| Training | 363 | 0 | Normal memory bank only |
| Validation | 78 | 390 | Supervised image threshold selection |
| Test | 78 | 391 | Frozen held-out image evaluation |

The split uses seed 42. Physical part IDs are unavailable. Near-duplicate detection has **not** been performed; related physical parts or visually near-identical images may span splits despite exact-hash grouping. Test results therefore cannot establish part-independent production generalization.

Training and runtime share `vision/patchcore_anomaly.py` preprocessing and extraction: uncropped BGR-to-RGB, resize to 224x224, ImageNet normalization, frozen ResNet18 layer2/layer3 features, local average pooling, bilinear layer3 alignment, concatenated 384-dimensional embeddings on a 28x28 patch grid. A seeded 50,000-candidate cap subsamples the 284,592 training patches, then approximate greedy farthest-point selection in a 64-dimensional Gaussian projection retains 1,024 original embeddings. This compact variant is not a complete canonical PatchCore reproduction: it uses a smaller backbone and image scores are the maximum Euclidean nearest-normal-patch distance without PatchCore neighborhood reweighting.

The threshold maximizes validation F1; ties prefer precision and then the larger threshold. The runtime rule is strictly `score > threshold`. Both classes are used only for validation calibration, while the memory bank contains normal training images only. The artifact is saved with the chosen threshold **before test images are loaded**. No test threshold tuning was performed. High defective prevalence makes F1 favor sensitivity and does not reflect expected production prevalence.

Frozen test results on 469 images:

| Metric | Result |
| --- | ---: |
| AUROC | 0.9558 |
| Average precision | 0.9914 |
| Precision | 0.9173 |
| Recall | 0.9642 |
| F1 | 0.9401 |
| True normal / false alarm | 44 / 34 |
| Missed defect / detected defect | 14 / 377 |

**34 of 78 normal test images trigger false alarms (43.6%).** This threshold is a sensitive demo baseline, not suitable for autonomous production acceptance. A production operating point requires engineering costs, representative prevalence, and new calibration evidence; changing threshold after viewing this test needs a new untouched test set.

Artifacts: `models/patchcore/casting_proxy.pt`, `metadata.json`, and `evaluation.json`. The evaluation includes raw image scores, predictions, image hashes, validation/test metrics, split provenance, and artifact hash. Artifact format is version 1 with a CPU float32 1024x384 bank, positive distance threshold, verified backbone path and hash, and casting proxy metadata. The fitted artifact hash is `492ac07823a95b40289d9e62717872e1d613200ea2c3ec628c29ecbb185e1b56`. Runtime outputs carry artifact and backbone evidence through provenance.

Scores are feature distances, **not probabilities or calibrated confidence**. The 28x28 distance grid is not a segmentation mask. There are no ground-truth masks, so no pixel metrics, millimetre metrology, or defect-class-specific unknown-anomaly performance is claimed. This baseline detects departures from normal casting appearance; it does not establish safety severity or unknown brake-disc defect recognition.

## Live API validation

Set `LINEGUARD_MODEL_DEVICE=0` before API startup for the RTX4050; the adapter normalizes this shared setting to PyTorch `cuda:0`. `POST /api/proxy/frames/{id}/anomaly` requires an existing quality-passed frame. Missing frame returns404, rejected capture422, unavailable model503. Runs persist raw28×28 scores, threshold, image hash and artifact hash; retrieve at `/api/proxy/runs/{id}`. It does not combine casting anomalies with NEU steel detections.

All469 test images were exercised through the live endpoint without threshold changes. The provisional gate accepted327/rejected142. All327 stored outputs matched benchmark scores within0.001 and their exact artifact/image hashes. Accepted-image confusion matrix `[[24,21],[11,271]]`, conditional recall96.10% and normal false alarms46.67%; these metrics exclude rejected images. Median GPU inference HTTPrequest14.5ms and p9520.2ms exclude the separate upload/quality request and do not establish a full-pipeline latency guarantee. Reports: `data/patchcore_api_test/report.json`; figures: `examples.png`.

Algorithm references: [official PatchCore implementation](https://github.com/amazon-science/patchcore-inspection) and [CVPR2022 paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf).
