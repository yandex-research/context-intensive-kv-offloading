FROM nvcr.io/nvidia/pytorch:24.10-py3
COPY setup.sh setup.sh
COPY shadowkv-models /workspace/shadowkv-models
COPY shadowkv-opencompass /workspace/shadowkv-opencompass
COPY needlebench longproc text2json.jsonl /root/.cache/opencompass/data/
COPY run_cfg.py /workspace/run_cfg.py
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV HF_HOME=/mnt/LLM
RUN chmod +x setup.sh
RUN ./setup.sh
