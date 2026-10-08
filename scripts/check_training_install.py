"""Cheap package metadata check used by the local setup launcher."""
import sys
from importlib.metadata import version, PackageNotFoundError
try:
    ready = version('torch') == '2.6.0+cu126' and version('torchvision') == '0.21.0+cu126'
except PackageNotFoundError:
    ready = False
sys.exit(0 if ready else 1)
