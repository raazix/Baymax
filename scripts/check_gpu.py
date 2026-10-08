"""Optional training-environment preflight after installing ML dependencies."""
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
else:
    raise SystemExit('CUDA GPU is not available. Switch out of iGPU-only mode and restart before GPU training.')
