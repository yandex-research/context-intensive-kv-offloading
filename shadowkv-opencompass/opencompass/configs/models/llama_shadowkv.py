from opencompass.models import LlamaShadowKV


models = [
    dict(
        path="meta-llama/Llama-3.1-8B-Instruct",
        type=LlamaShadowKV,
        run_cfg=dict(num_gpus=1),
        batch_size=1,
        sparse_budget=10240,
        local_budget=32,
        outlier_budget=384,
        max_length=128 * 1024,
        chunk_size=8,
        rank=160
    )
]
