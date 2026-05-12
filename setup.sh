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

pip install -U pip setuptools
pip install wheel torch==2.7 transformers==4.52.4 pyarrow==23 flashinfer-python torchvision==0.22 vllm==0.9 pandas==2.3.3

pip install ./shadowkv-opencompass

mkdir shadowkv-models/3rdparty
git clone https://github.com/NVIDIA/cutlass.git ./shadowkv-models/3rdparty/cutlass
MAX_JOBS=16 pip install --no-build-isolation ./shadowkv-models

rm -rf ./shadowkv-opencompass ./shadowkv-models

git clone https://github.com/Dao-AILab/fast-hadamard-transform.git fast-hadamard-transform
cd fast-hadamard-transform
MAX_JOBS=16 pip install --no-build-isolation .
