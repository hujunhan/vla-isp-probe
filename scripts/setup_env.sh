#!/usr/bin/env bash
# E001 environment setup: OpenVLA-OFT + LIBERO-plus. [ai-scaffold]
set -euxo pipefail
source "$(dirname "$0")/env.sh"
source "$CONDA_SH"
conda create -n oft python=3.10.14 -y
conda activate oft
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
cd $OPENVLA_OFT_ROOT && pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation || echo "FLASH_ATTN_FAILED"
cd $LIBERO_PLUS_ROOT && pip install -e .
pip install -r $OPENVLA_OFT_ROOT/experiments/robot/libero/libero_requirements.txt
pip install -r $LIBERO_PLUS_ROOT/extra_requirements.txt
# (huggingface_hub CLI ships with huggingface_hub 0.36.2; do not -U)
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pip freeze > $LOG_DIR/e001-pip-freeze.txt
echo SETUP_DONE
# Pins found necessary in E001 (unpinned resolution breaks torch 2.2 / transformers 4.40 / robosuite 1.4.1):
#   numpy 2.x breaks torch 2.2 + TF 2.15; tensorflow-metadata 1.21 needs newer protobuf;
#   wandb 0.28 needs protobuf>4.21; mujoco 3.14 fails robosuite's joint-type assert;
#   accelerate 1.15 calls .to() on bnb-quantized models inside from_pretrained.
pip install "numpy==1.26.4" "huggingface_hub==0.36.2" "opencv-python==4.10.0.84" \
  "tensorflow-metadata==1.14.0" "wandb==0.16.6" "mujoco==3.2.6" "accelerate==0.30.1" "bitsandbytes==0.44.1"
# LIBERO-plus's editable install doesn't resolve; set PYTHONPATH=$LIBERO_PLUS_ROOT at run time.
# Assets: unzip assets.zip, then move .../LIBERO-plus-0/assets to LIBERO-plus/libero/libero/assets.
