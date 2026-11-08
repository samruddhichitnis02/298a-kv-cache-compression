#!/usr/bin/env bash
set -euo pipefail

# Run from members/vedant inside the lab container. No Conda or sudo is needed.
# --system-site-packages keeps the lab image's torch 2.7.1+cu128 (built for the RTX 5090)
# and transformers 5.2.0; pip then only adds the small extras (pytest, rouge, ...).
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[dev,hf,plots,score]'
mkdir -p results/environment
nvidia-smi > results/environment/nvidia_smi.txt
python -m pip freeze > results/environment/python_packages.txt
python - <<'PY'
import json
import torch
from pathlib import Path

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Install the lab-approved CUDA PyTorch build before proceeding.")
# Execute an actual CUDA operation; a visible device alone does not prove kernel support.
x = torch.ones((16, 16), device="cuda")
assert (x @ x).sum().item() == 4096
info = {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
        "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
        "compiled_architectures": torch.cuda.get_arch_list()}
Path("results/environment/cuda_preflight.json").write_text(json.dumps(info, indent=2) + "\n")
print(json.dumps(info, indent=2))
PY
python -m pytest -q
echo "CUDA preflight passed. Next: hf auth login (do not paste the token into code)."
