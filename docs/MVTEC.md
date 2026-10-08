# MVTec AD metal-nut benchmark

`archive.zip` is MVTec AD 2019. It contains 15 industrial object/texture categories, train/good images, test images, and ground-truth anomaly masks. It is not a brake-disc dataset.

## Local training and evaluation

Only the `metal_nut` category is used for this project benchmark. Training uses the official `train/good` images, with the existing custom PatchCore trainer holding out a deterministic 44/220 normal images for threshold calibration. Official test images/masks are untouched until evaluation. The archive and extracted data stay local and ignored by Git.

```powershell
.\.venv\Scripts\python.exe scripts\train_mvtec_metal_nut.py --archive archive.zip --device cuda:0
.\.venv\Scripts\python.exe scripts\evaluate_mvtec_metal_nut.py
```

Trained on RTX 4050 GPU: 176 memory-bank normal images, 44 held-out normals, threshold 2.5225. On the official 115-defect / 22-good test split: image AUROC 0.9892; at the held-out-normal threshold, 81/93 defects detected (87.1% recall), 12 missed, and 0/22 good images flagged. Defect-class recall: bent 24/25, color 19/22, flip 23/23, scratch 15/23. Interpolated 64x64 patch-grid pixel AUROC is 0.9726; this is not native-resolution segmentation accuracy.

The separate model is registered as `mvtec_metal_nut`, and can be chosen in the dashboard's advanced PatchCore selector for MVTec-like metal-nut images. It does not replace the MPDD default. The dashboard keeps the procedural brake-rotor AR view disabled for this model because a nut's image grid must not be drawn as if it were registered to a disc.

## License and limits

The included archive states CC BY-NC-SA 4.0 and requests citation of Bergmann et al., “A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection,” CVPR 2019. Commercial use requires contacting MVTec. The archive, extracted images, and trained model artifacts are not committed. Follow the archive's [CC BY-NC-SA 4.0 terms](https://creativecommons.org/licenses/by-nc-sa/4.0/).

These metrics describe only MVTec metal nuts under its dataset setup. They do not indicate performance on metal plates or brake discs. The model is a benchmark option, not an automotive inspection model.
