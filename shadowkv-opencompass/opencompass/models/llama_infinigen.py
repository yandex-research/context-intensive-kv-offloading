from opencompass.registry import MODELS
from shadowkv_models.llama_infinigen import patch_llama_model
from .huggingface_above_v4_33 import HuggingFacewithChatTemplate


@MODELS.register_module()
class LlamaInfiniGenwithChatTemplate(HuggingFacewithChatTemplate):
    """Model wrapper for HuggingFace models designed for chat.

    Args:
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'mid' represents the part of input to
            truncate. Defaults to 'none'.
    """

    def __init__(
        self,
        *args,
        # InfiniGen parameters
        skewing_matrix_path: str | None = None,
        partial_weight_path: str | None = None,
        partial_weight_ratio=0.2,
        capacity=1.0,
        alpha=99,
        budget=0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = patch_llama_model(
            self.model,
            skewing_matrix_path,
            partial_weight_path,
            partial_weight_ratio,
            capacity,
            alpha,
            budget,
        )
