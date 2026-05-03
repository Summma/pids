#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv-h100
source .venv-h100/bin/activate
python -m pip install --upgrade pip wheel setuptools

# H100 path. CUDA-enabled torch wheels are resolved by pip on Linux if the base
# image already has a compatible CUDA runtime; otherwise install the platform's
# recommended torch wheel first and re-run this script.
python -m pip install -r ml/requirements-h100.txt

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
PY
