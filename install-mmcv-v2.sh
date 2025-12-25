#!/usr/bin/env bash
set -e

ENV_NAME=nerfstream

echo "============================================"
echo " Fixing MuseTalk inside conda env: $ENV_NAME"
echo " Updated working configuration - all versions tested"
echo " PyTorch 2.0.1+cu117, mmcv 2.1.0, mmpose 1.2.0"
echo "============================================"

# ---------------------------------------------
# 1. Activate existing environment
# ---------------------------------------------
#source ~/miniforge/etc/profile.d/conda.sh
#conda activate $ENV_NAME

# ---------------------------------------------
# 2. Pin pip and upgrade build tools
# ---------------------------------------------
python -m pip install --upgrade pip setuptools wheel

# ---------------------------------------------
# 3. Install PyTorch 2.0.1 with CUDA 11.7
#    This version is compatible with mmcv pre-built wheels
# ---------------------------------------------
pip uninstall -y torch torchvision torchaudio || true
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
  --index-url https://download.pytorch.org/whl/cu117

python - << 'EOF'
import torch
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPUs:", torch.cuda.device_count())
EOF

# ---------------------------------------------
# 4. Install OpenMMLab packages via mim
# ---------------------------------------------
pip install --no-cache-dir -U openmim

# Remove any conflicting packages first
pip uninstall -y mmdet mmcv mmcv-full mmpose mmengine chumpy || true

# Install mmengine (must be 0.9.1 for mmcv 2.1.0 compatibility)
mim install mmengine==0.9.1

# Install mmcv 2.1.0 with pre-built wheel for CUDA 11.7 + PyTorch 2.0.1
# Note: Using PyTorch 2.0.0 wheels which are compatible with 2.0.1
mim install "mmcv>=2.0.0,<2.2.0" \
  -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html

# Install mmdet (3.3.0 is compatible with mmcv 2.1.0)
mim install "mmdet>=3.1.0,<3.4.0"

# Install mmpose 1.2.0 (tested and working version)
pip install mmpose==1.2.0

# ---------------------------------------------
# 6. Verify OpenMMLab stack
# ---------------------------------------------
python - << 'EOF'
import mmengine, mmcv, mmpose
from mmpose.apis import inference_topdown, init_model
print("MMEngine:", mmengine.__version__)
print("MMCV:", mmcv.__version__)
print("MMPose:", mmpose.__version__)
print("Checking mmcv._ext...")
import mmcv._ext
print("mmcv._ext OK!")
EOF

# ---------------------------------------------
# 7. Install MuseTalk dependencies with compatible versions
# ---------------------------------------------
# Install compatible versions of huggingface ecosystem
pip install \
  "diffusers>=0.20.0,<0.21.0" \
  "huggingface-hub>=0.16.0,<0.18.0" \
  "transformers>=4.30.0,<4.35.0" \
  "accelerate>=0.20.0,<0.25.0"

# Fix filelock if corrupted (common issue)
pip uninstall -y filelock || true
pip install "filelock>=3.20.0"

# Install other MuseTalk dependencies
pip install \
  sentencepiece \
  mediapipe \
  facexlib \
  basicsr \
  einops \
  librosa \
  scipy \
  opencv-python \
  tqdm \
  rich

# Fix numpy version (required for xtcocotools compatibility)
pip install --force-reinstall "numpy==1.26.4" || true

# Optional but recommended
pip uninstall -y tensorflow || true

# ---------------------------------------------
# 8. Final sanity check
# ---------------------------------------------
python - << 'EOF'
import torch
import mmcv
import mmpose

# Check PyTorch CUDA
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA NOT AVAILABLE"

# Check mmcv._ext
print("MMCV:", mmcv.__version__)
import mmcv._ext
print("mmcv._ext: OK")

# Check mmpose APIs
print("MMPose:", mmpose.__version__)
from mmpose.apis import inference_topdown, init_model
print("mmpose APIs: OK")

# Check diffusers
from diffusers import AutoencoderKL
print("diffusers: OK")

print("\n✅ All critical dependencies verified!")
EOF

echo "============================================"
echo " Installation complete!"
echo " Environment '$ENV_NAME' is MuseTalk-ready"
echo " Run: conda activate $ENV_NAME"
echo " Then: python genavatar_musetalk.py --help"
echo "============================================"
