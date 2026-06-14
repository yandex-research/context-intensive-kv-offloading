# KV Cache Offloading for Context-Intensive Tasks
Supplementary code for ["KV Cache Offloading for Context-Intensive Tasks"](https://arxiv.org/pdf/2604.08426) <a href='https://arxiv.org/pdf/2604.08426'><img src='https://img.shields.io/badge/ArXiv-PDF-red' height="25"></a> &nbsp; 


The Text2JSON dataset gathered and used in the experiments is available as [`./text_to_json.jsonl.gz`](https://github.com/yandex-research/context-intensive-kv-offloading/blob/eddd76f9d494feb90cf1f8d4a66688b729ce48d6/text_to_json.jsonl.gz). The nearest update will move the dataset to HF and provide detailed evaluation config for ShadowKV and improved variants on Text2JSON and other benchmarks integrated into [OpenCompass](https://github.com/open-compass/opencompass/tree/12462107fe746db536bc4b44bb6b58f0736251fe).


# Evaluation

## Transformers==4.52.4

To run the experiments, you first need to build a docker container with our environment:

`docker build -t eval_image .`

Run the image with:

`docker run -it -d --gpus all --volume /path/to_your_downloaded_models:/mnt/LLM eval_image`

And then exec into container via:

`docker exec -it container_id bash`

Note: if you wish to forego mounting downloaded models and download them on the go, remove the `volume` flag and unset `HF_HOME` variable inside the container.

Inside the image, run `opencompass run_cfg.py` with appropriate `CUDA_VISIBLE_DEVICES`. We ran our experiments on A100-80G GPUs.

To run the validation with YAKV on MultiNeedle, Text2JSON and LongProc datasets and Qwen3-30B-A3B-Instruct-2507, Qwen3-4B-Instruct-2507, Llama-3.1-8B-Instruct and Llama-3.2-3B-Instruct run
```
CUDA_VISIBLE_DEVICES=0 opencompass -w outputs run_cfg.py
```
from `/workspace/`. If you wish to use some subset of models or dataset, modify the [run_cfg.py](https://github.com/yandex-research/context-intensive-kv-offloading/blob/main/run_cfg.py) accordingly. You can change sparse budget, chunk size, HIGGS grid, etc by changing corresponding values from the same `run_cfg.py`

## Transformers==5.6.1

You can also use a newer transformers version. The setup is the same, just use Dockerfile_v5 instead of Dockerfile:

`docker build -t eval_image -f Dockerfile_v5 .`

Then do `docker run` and `docker exec` just like commands above.

## Baselines

You can evaluate different baselines, like as ArkVale and LRQK too.

To do this, follow the installation steps below.

### ArkVale 

Git repo that will be cloned is a fork of official repo. The fork is different in 3 things:
* it supports newer version of transformers (so you must install transformers==5.6.1)
* it does not offload generated tokens to have a more fair comparison with ShadowKV and YAKV (upper bound of ArkVale is being evaluated)
* it supports additional group_size (from GQA) values, so we could do the evaluation on models like Llama-3.2-3B with group_size=3

#### Setup
```
mkdir baselines
git clone -b GQA_3_fix https://github.com/AndreyBocharnikov/ArkVale.git baselines/ArkVale
cd baselines/ArkVale
git submodule update --init --recursive --depth 1
cd source
TORCH_CUDA_ARCH_LIST="8.0" uv pip install --no-build-isolation -e .
cd ..
```
Change `TORCH_CUDA_ARCH_LIST` according to the GPU you are using, `TORCH_CUDA_ARCH_LIST="8.0"` is for A100.

#### Launch
Just like with YAKV, you can controll what models and datasets are used from `run_config.py` file. 

`python opencompass_run.py run_config.py -w outputs`

### LRQK

TODO

# GPU Inference

We have separate detailed instructions for compiling and running GPU inference experiments in [`./inference`](./inference), see [README.md](./inference/README.md) inside.


# Code Ownership

Our evaluation code uses significant parts of [OpenCompass](https://github.com/open-compass/opencompass/) and implements our benchmarks and offloading methods as components. We also include ShadowKV implementation using the original [ByteDance-Seed/ShadowKV](https://github.com/ByteDance-Seed/ShadowKV/) project.
In turn, our GPU inference code modifies the [mini-sglang](https://github.com/sgl-project/mini-sglang/) framework - a minimal version of [SGLang](https://github.com/sgl-project/sglang).
We do not own the [OpenCompass](https://github.com/open-compass/opencompass/) codebase or the [mini-sglang](https://github.com/sgl-project/mini-sglang/) framework and we are immensely grateful to their authors for their implementations. Our own code implements YAKV and additional evaluation benchmarks within these frameworks. The specific versions that we use are under [Apache-2.0 License for OpenCompass](https://github.com/open-compass/opencompass/blob/7f8eee472581f42e014163a5a8c98cb049786dd2/LICENSE), [MIT License for mini-sglang](https://github.com/sgl-project/mini-sglang/commit/34fe3f31fd12f26b0a8f7ed5044b292f493df5a0), and [Apache-2.0 License again for ShadowKV](https://github.com/ByteDance-Seed/ShadowKV/blob/71bb0e9953ca56efa9689d3d8321b7f95b8d0694/LICENSE).
