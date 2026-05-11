import os
import torch

from collections import namedtuple
from fast_hadamard_transform import hadamard_transform
from minisgl.shadowkv_kernels import higgs_quantize, higgs_dequantize, higgs_dequantize_full


def is_power_of_two(x: int) -> bool:
    return (x & (x - 1)) == 0


def quantize(x: torch.Tensor):
    assert is_power_of_two(x.shape[-1])


def get_grid(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    file = f"{os.path.dirname(os.path.abspath(__file__))}/grids/EDEN4-256.pt"
    grid = torch.load(file, map_location="cpu").to(device=device, dtype=dtype)

    return grid


QuantizedTensor = namedtuple("QuantizedTensor", ["idx", "scales"])


class HiggsQuantizerCUDA:
    def __init__(self, hadamard_groupsize: int, edenn_d: int, edenn_n: int, dtype, device):
        if edenn_d != 4 or edenn_n != 256:
            raise NotImplementedError

        self.grid = get_grid(device=device, dtype=dtype)
        self.channel_size = hadamard_groupsize
        self.dtype = dtype
        self.hadamard_scale = 1 / hadamard_groupsize

    def quantize(self, x) -> torch.Tensor:
        assert len(x.shape) == 2, x.shape
        channel_size = x.shape[1]
        assert channel_size == self.channel_size
        assert x.dtype == self.dtype, (x.dtype, self.dtype)
        assert x.is_contiguous()

        x = x.to(dtype=torch.float32)
        scales = torch.linalg.norm(x, axis=-1, keepdim=True)
        x = hadamard_transform(x) / scales

        idx = higgs_quantize(x.to(self.dtype), self.grid)

        return QuantizedTensor(idx=idx, scales=scales.squeeze(dim=-1).to(self.dtype))

    def dequantize(self, quantized: QuantizedTensor) -> torch.Tensor:
        assert quantized.idx.is_contiguous()
        assert quantized.scales.is_contiguous()

        x = higgs_dequantize(
            quantized.idx,
            quantized.scales,
            self.grid,
        )

        x = hadamard_transform(x, scale=self.hadamard_scale).to(self.dtype)

        return x

    def full_dequantize(self, quantized: QuantizedTensor, out: torch.Tensor):
        higgs_dequantize_full(quantized.idx, quantized.scales, self.grid, out, self.hadamard_scale)
