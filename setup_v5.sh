#!/usr/bin/bash

set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

cd /workspace
apt-get autoremove -y

apt-get update --allow-unauthenticated --allow-insecure-repositories
apt-get install python3.10-venv -y --allow-unauthenticated 

python3 -m venv --system-site-packages /venv
export PATH="/venv/bin:$PATH"
echo 'export PATH="/venv/bin:$PATH"' > ~/.bashrc

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv pip install -U pip setuptools 
uv pip install wheel torch==2.10 transformers==5.6.1 pyarrow==23 flashinfer-python torchvision==0.25 vllm==0.19.1 pandas==2.3.3
uv pip install --no-build-isolation flash-attn==2.8.3

uv pip install -e ./shadowkv-opencompass
ln -sfn /workspace/shadowkv-opencompass/opencompass /venv/lib/python3.10/site-packages/opencompass

mkdir shadowkv-models/3rdparty
git clone https://github.com/NVIDIA/cutlass.git ./shadowkv-models/3rdparty/cutlass
MAX_JOBS=16 uv pip install --no-build-isolation --no-deps -e ./shadowkv-models

git clone https://github.com/Dao-AILab/fast-hadamard-transform.git fast-hadamard-transform
cd fast-hadamard-transform
uv pip install --no-build-isolation .
