# NEU training run

## Validation-only fine-tuning

`.venv\Scripts\python.exe scripts/finetune_yolo.py` starts from the 30-epoch baseline and trains conservatively. `--strategy rebalanced` runs a short alternate pass with each original crazing training image sampled three times and mosaic disabled. Validation and test sets are unchanged. Image repetitions remain wholly within the training split.

Before training, the script declares its activation rule: overall validation mAP50-95 and crazing AP50 must both improve over the baseline. Otherwise the existing active model is retained. Candidate reports, weights and curves live under `models/yolo/neu_yolo11n_finetuned` and `models/yolo/neu_yolo11n_rebalanced`. The test set is not run or used for candidate selection. A newly activated checkpoint requires restarting the API to update its already-loaded model.

CUDA-enabled PyTorch 2.6.0, TorchVision 0.21.0 and Ultralytics 8.3.253 are installed in the workspace `.venv`. Official wheel SHA-256 checks passed; wheels are cached in `data/wheels`. The pretrained YOLO11n checkpoint is in `models/yolo/yolo11n.pt`. GPU access was confirmed after switching the laptop out of iGPU-only mode: RTX 4050, 6 GB VRAM.

Run in your normal laptop PowerShell from `D:\baymax`:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_neu_training.ps1
```

This creates/uses the workspace virtual environment, reuses the installed CUDA 12.6 PyTorch or verified cached wheels, checks CUDA, trains YOLO11n for up to 30 epochs (early stopping after 10 epochs without improvement), saves checkpoints every 5 epochs, and evaluates the best validation-selected checkpoint on the held-out test set once. NVIDIA drivers must support the selected CUDA runtime. The script fails rather than quietly switching to CPU if CUDA is unavailable. No system execution policy is changed; Bypass applies only to this process.

Defaults: 640px input, batch 4, workers 0 for Windows, seed 42. Preserve original six NEU labels and prepared 1,260/270/270 train/validation/test split. Test images and exact duplicate groups are kept separate from training. No physical-part metadata is available for stronger grouping.

Outputs under `models/yolo/neu_yolo11n` (a numeric suffix is added for repeat runs):

- `weights/best.pt`: validation-selected model.
- `weights/last.pt`: resume checkpoint.
- `results.csv` and training curves.
- `evaluation.json`: held-out overall and per-class metrics, weight/manifest hashes, GPU and provenance.
- A separate test evaluation directory with diagnostic plots.

If CUDA runs out of memory, rerun with batch 2:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_neu_training.ps1 -Batch 2
```

Resume an interrupted run once dependencies are installed:

```powershell
.\.venv\Scripts\python.exe scripts/train_yolo.py --task detect --data data/processed/neu-det/dataset.yaml --resume models/yolo/neu_yolo11n/weights/last.pt --device 0
```

Use the exact backend configuration printed on completion, then restart the API with the virtual environment Python:

```powershell
$env:LINEGUARD_PROXY_WEIGHTS = 'D:/baymax/models/yolo/neu_yolo11n/weights/best.pt'
$env:LINEGUARD_MODEL_DEVICE = '0'
.\.venv\Scripts\python.exe -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

Stop the existing API process on port 8000 first. Upload a JPEG/PNG to `/api/frames`, then call `/api/proxy/frames/{id}/detect`. This model is a steel-surface proxy, not a validated brake-disc detector or segmentation/metrology model.

Alternatively, register the evaluated model once:

```powershell
.\.venv\Scripts\python.exe scripts/activate_proxy.py --report models/yolo/neu_yolo11n/evaluation.json
```

The backend then reads `models/yolo/active.json` at startup, verifies the checkpoint hash before loading, and uses the registered GPU device. Explicit environment variables take precedence. An already running API needs a restart to read a newly activated model.
