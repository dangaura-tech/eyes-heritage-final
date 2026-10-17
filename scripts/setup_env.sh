#!/bin/bash
set -e
cd ~/heritage
export PATH=$HOME/.local/bin:$PATH
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
rm -rf .venv
uv venv -p 3.12 .venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install "transformers>=4.56" timm peft rasterio pyproj shapely geopandas opencv-python-headless scikit-image scikit-learn lightgbm catboost pandas pyarrow matplotlib tqdm optuna huggingface_hub accelerate tabulate scipy
python -c "import torch;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"
echo SETUP_DONE
