from opencompass.models import HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr='llama-3_1-8b-hf',
        path='meta-llama/Meta-Llama-3.1-8B-Instruct',
        max_out_len=1024,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
        model_kwargs=dict(torch_dtype=torch.bfloat16),
    )
]
