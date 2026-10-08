# MPDD component anomaly baselines

Six **separate** compact PatchCore models were trained on MPDD normal components.
They are manufacturing-component proxies, not brake-disc models. Their observed
operating-point recall varies sharply; several models miss most defects. These
are experimental baselines, not production quality gates.

| Component | Test AUROC | Test AP | Detected defects | Normal false alarms |
|---|---:|---:|---:|---:|
| bracket_black | 0.7201 | 0.8076 | 10/47 (21.28%) | 1/32 (3.13%) |
| bracket_brown | 0.9178 | 0.9150 | 24/51 (47.06%) | 2/26 (7.69%) |
| bracket_white | 0.8622 | 0.9020 | 22/30 (73.33%) | 4/30 (13.33%) |
| connector | 1.0000 | 1.0000 | 13/14 (92.86%) | 0/30 (0%) |
| metal_plate | 1.0000 | 1.0000 | 27/71 (38.03%) | 0/26 (0%) |
| tubes | 0.7980 | 0.8968 | 24/69 (34.78%) | 2/32 (6.25%) |

AUROC and AP describe ranking across thresholds. They do not establish high
recall at the selected threshold: metal_plate ranks these test images perfectly
but misses 44/71 defects at its conservative calibration threshold.

The original 888 official-training normals were split into 710 bank-training and
178 normal-only validation images, with seeded exact-hash groups. The official
458 test images were preserved: 176 normals and 282 defects, with genuine defect
masks. Each category's memory bank uses only its own training normals; no banks
are pooled across MPDD categories, casting data, or NEU steel images.

Calibration uses only the withheld training normals. The deterministic objective
is the lowest positive distance threshold that permits at most `floor(N*.05)`
validation-normal false alarms under strict `score > threshold`. This 5% target
is a configurable demo criterion, not an engineering-approved safety limit or a
population guarantee. Validation sets contain only 11–58 normals. Metal_plate
has 11 validation normals, so it permits zero alarms; one high normal score sets
a conservative threshold. Test false-alarm rates can exceed 5%, as observed for
brown, white, and tubes. No thresholds were changed after inspecting test results.

All models use the same verified local ImageNet ResNet18 frozen layer2/layer3
feature extractor, local 3x3 pooling, 224x224 per-crop ImageNet normalization,
and exact nearest-normal-patch Euclidean distance. Shared `square_crops` ensures
consistent handling of non-square images. Each 1,024-patch bank is an approximate
greedy coreset from a seeded 50,000-candidate cap and 64-dimensional random
projection, retaining the original 384-dimensional embeddings. The image score
is the maximum patch distance over all crops; scores are not probabilities.

Image and mask bytes are checked against the frozen manifest. The manifest
validator rejects duplicate paths, cross-split/category/label exact-hash leakage,
defective training/validation images, and modifications to official test origin.
Each bank and threshold are saved **before that category's test images are
loaded**. The test is evaluated once after freezing. The tubes process was
restarted after an interrupted first feature pass, before it had saved any bank
or loaded test images. Completed category artifacts were preserved.

Pixel metrics are deferred: authentic masks are retained and their hashes
verified, but raw anomaly grids are not validated segmentation masks. No pixel
performance, defect subtype recognition, physical dimensions, or engineering
severity is claimed. Physical-part identities and near-duplicate leakage beyond
exact image hashes remain unresolved. Connector's strong result uses only 14
defective test images and must not be generalized to unseen product domains.

Artifacts live in `models/patchcore/mpdd/<category>/casting_proxy.pt`, alongside
`metadata.json` and `evaluation.json` with thresholds, confusion matrices,
calibration counts, raw validation/test scores, image identities, and hashes.
The legacy filename supports the shared artifact loader; metadata explicitly
identifies `data_source=mpdd_proxy` and the component category.

Train with `.venv\Scripts\python.exe scripts/train_mpdd.py --device cuda:0`.
Select categories with `--categories connector metal_plate`; use a fresh
`--output-root` to preserve existing artifacts. All four tests pass in
`tests/test_mpdd_training.py`.
