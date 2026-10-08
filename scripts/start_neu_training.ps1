param([int]$Epochs = 30, [int]$Batch = 4)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    python -m venv --system-site-packages .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
$env:PIP_CACHE_DIR = Join-Path $projectRoot 'data/pip-cache'
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot 'data/yolo-config'
& $pythonPath scripts/check_training_install.py
if ($LASTEXITCODE -ne 0) {
    $torchWheel = Join-Path $projectRoot 'data/wheels/torch-2.6.0+cu126-cp312-cp312-win_amd64.whl'
    $visionWheel = Join-Path $projectRoot 'data/wheels/torchvision-0.21.0+cu126-cp312-cp312-win_amd64.whl'
    if ((Test-Path -LiteralPath $torchWheel) -and (Test-Path -LiteralPath $visionWheel)) {
        & $pythonPath -m pip install $torchWheel $visionWheel
    } else {
        & $pythonPath -u scripts/download_training_wheels.py
        if ($LASTEXITCODE -ne 0) { throw 'Official CUDA wheel download failed; partial chunks are preserved for resumption.' }
        & $pythonPath -m pip install $torchWheel $visionWheel
    }
    if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
}
& $pythonPath -m pip install 'ultralytics>=8.3,<8.4'
if ($LASTEXITCODE -ne 0) { throw 'Ultralytics installation failed.' }
& $pythonPath scripts/check_gpu.py
if ($LASTEXITCODE -ne 0) { throw 'GPU preflight failed.' }
if (-not (Test-Path -LiteralPath 'data/processed/neu-det/dataset.yaml')) {
    & $pythonPath scripts/prepare_neu.py
    if ($LASTEXITCODE -ne 0) { throw 'NEU data preparation failed.' }
}
New-Item -ItemType Directory -Path 'models/yolo' -Force | Out-Null
$trainingLog = Join-Path $projectRoot ('models/yolo/training-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
$priorErrorPreference = $ErrorActionPreference
try {
    # tqdm may write normal progress to stderr; preserve it without treating it as a fatal PowerShell error.
    $ErrorActionPreference = 'Continue'
    & $pythonPath -u scripts/train_yolo.py --task detect --data data/processed/neu-det/dataset.yaml --epochs $Epochs --batch $Batch --device 0 --workers 0 --weights models/yolo/yolo11n.pt 2>&1 | Tee-Object -FilePath $trainingLog
    $trainingExitCode = $LASTEXITCODE
} finally { $ErrorActionPreference = $priorErrorPreference }
if ($trainingExitCode -ne 0) { throw "Training failed. Inspect $trainingLog; checkpoints are preserved." }
Write-Output 'Training and held-out evaluation completed. See models/yolo/neu_yolo11n*/evaluation.json.'
