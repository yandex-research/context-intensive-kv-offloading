from opencompass.models import Qwen3ShadowKV


models = [
    dict(
        path="Qwen/Qwen3-8B",
        type=Qwen3ShadowKV,
        run_cfg=dict(num_gpus=1),
        batch_size=1,
        sparse_budget=10240,
        local_budget=32,
        outlier_budget=384,
        max_length=68 * 1024,
        chunk_size=8,
        rank=160,
    )
]
