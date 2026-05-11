import torch

from .higgs import QuantizerBase


class FP8Quantizer(QuantizerBase):
    def __init__(self, origin_type):
        super().__init__()
        self.type = origin_type

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        input_type = x.dtype
        assert input_type == self.type
        return x.to(torch.float8_e4m3fn)

    @torch.no_grad()
    def dequantize(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.type)
