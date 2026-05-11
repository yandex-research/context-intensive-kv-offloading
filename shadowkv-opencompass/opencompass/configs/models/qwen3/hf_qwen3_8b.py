from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen3-8b',
        path='Qwen/Qwen3-8B',
        max_out_len=4096,
        max_seq_len=69632,
        batch_size=3,
        run_cfg=dict(num_gpus=1),
        chat_template_kwargs=dict(enable_thinking=False),
        model_kwargs=dict(attn_implementation="flash_attention_2"),
    )
]
