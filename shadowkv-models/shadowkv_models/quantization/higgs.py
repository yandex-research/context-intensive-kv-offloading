import torch
import pathlib
import warnings
import functools
from typing import TypeVar
from functools import partial
from collections import namedtuple

from fast_hadamard_transform import hadamard_transform


def rotate_half(
    x: torch.Tensor
) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def flash_attn_llama(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_prefill: bool
) -> torch.Tensor:
    Tq, Tk = q.size(-2), k.size(-2)
    is_causal = bool(is_prefill and (Tq == Tk))
    return torch.nn.functional.scaled_dot_product_attention(
        q.contiguous(), k.contiguous(), v.contiguous(),
        attn_mask=None, dropout_p=0.0, is_causal=is_causal
    )


grids_folder = pathlib.Path(__file__).parent.resolve() / "higgs_grids"

GRIDS = {}
# Read files in the folder and read grids in the EDEN{DIM}_{SIZE}.pt format
for file in grids_folder.iterdir():
    if file.suffix == ".pt":
        try:
            if file.name.startswith("EDEN"):
                dim, size = map(int, file.stem[4:].split('-'))
            elif file.name.startswith("QUIPSHARP"):
                dim, size = map(int, file.stem[9:].split('-'))
            else:
                raise ValueError("Could not parse grid file name")
        except ValueError:
            warnings.warn(f"failed to parse grid {file}")
            continue
        GRIDS[dim] = GRIDS.get(dim, {})
        if size in GRIDS[dim]:
            warnings.warn(f"Got multiple grids for {dim=} {size=}, overriding with {file}")
        GRIDS[dim][size] = torch.load(file, map_location='cpu')

GRID_NORMS = {k1: {k2: torch.linalg.norm(GRIDS[k1][k2], dim=1) ** 2 for k2 in v1.keys()} for k1, v1 in GRIDS.items()}
print("FOUND GRIDS", GRIDS.keys())

@functools.lru_cache()
def get_grid(dim: int, size: int, device: torch.device) -> torch.Tensor:
    return GRIDS[dim][size].to(device)


@functools.lru_cache()
def get_grid_norms_squared(dim: int, size: int, device: torch.device) -> torch.Tensor:
    return torch.linalg.norm(get_grid(dim, size, device), dim=1).square()


def entropy(idx):
    _, counts = torch.unique(idx, return_counts=True)
    counts = counts.to(torch.float)
    return -torch.sum(counts / len(idx) * torch.log2(counts / len(idx))).item()


def pad_to_block(tensor, dims, had_block_size, value=0):
    pad_dims = [0 for _ in range(2 * len(tensor.shape))]
    for dim in dims:
        size = tensor.shape[dim]
        next_multiple_of_1024 = ((size - 1) // had_block_size + 1) * had_block_size
        delta = next_multiple_of_1024 - size
        pad_dims[-2 * dim - 1] = delta

    return torch.nn.functional.pad(tensor, pad_dims, "constant", value)

class QuantizerBase:
    QuantizedState = TypeVar('QuantizedState')

    def quantize(self, x: torch.Tensor) -> QuantizedState: ...

    def dequantize(self, quantized: QuantizedState) -> torch.Tensor: ...

    def quantize_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x)).to(dtype=x.dtype, device=x.device)

QuantizedTensor = namedtuple("QuantizedTensor", ["idx", "scales"])


class HiggsQuantizer(QuantizerBase):
    def __init__(self, hadamard_groupsize: int, edenn_d: int, edenn_n: int, channel_size: int = 1024, chunk_size: int = 64) -> None:
        """
        HIGGS vector quantization.
        :param hadamard_groupsize: perform random hadamard transform to groups of this many vectors
        :param edenn_d: quantization grouop dimension
        :param edenn_n: quantization lattice size
        :param channel_size: channel size of keys and values, used to trim padding 
        :param chunk_size: chunk size is used to avoid memory demanding matmul and split the input into chunk to perform multiple smaller matmuls
        """
        super().__init__()
        self.hadamard_groupsize = hadamard_groupsize
        self.channel_size = channel_size
        self.grid = partial(get_grid, dim=edenn_d, size=edenn_n) # grid of shape [edenn_d, edenn_n]
        self.grid_norm = partial(get_grid_norms_squared, dim=edenn_d, size=edenn_n)
        self.edenn_d = edenn_d
        self.chunk_size = chunk_size
        self.hadamard_scale = 1 / hadamard_groupsize

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> QuantizedTensor:
        """
        x.shape - [B, C]
        """
        batch_size = x.shape[0]
        channel_size = x.shape[1]
        assert channel_size == self.channel_size, "channel size from __init__ does not match the channel size of quantize argument. Make sure you create HiggsQuantizer with correct channel size"
        device = x.device
        x = x.to(dtype=torch.float32)
        x = pad_to_block(x, [1], self.hadamard_groupsize)
        mult = x.shape[1] // self.hadamard_groupsize
        x = x.reshape(-1, mult, self.hadamard_groupsize)
        scales = torch.linalg.norm(x, axis=-1) # [B, mult]
        x = hadamard_transform(x) / scales[:, :, None]

        x = pad_to_block(x, [2], self.edenn_d).reshape(batch_size, mult, -1, self.edenn_d)

        result_idx = torch.empty((batch_size, mult, x.shape[2]), dtype=torch.uint8)
        for i, chunk in enumerate(torch.split(x, self.chunk_size, dim=0)):
            chunk_idx = torch.argmax(2 * chunk @ self.grid(device=device).T - self.grid_norm(device=device), dim=-1) # [B, mult, pad(pad(C)) / mult / d]
            result_idx[i * self.chunk_size: (i + 1) * self.chunk_size] = chunk_idx

        return QuantizedTensor(
            result_idx,
            scales
        )

    @torch.no_grad()
    def dequantize(self, quantized: QuantizedTensor) -> torch.Tensor:
        """
        quantized.idx shape is [B, padded_d(padded_had(C)) // d]
        quantized.scale shape is [B, padded_had(C) // hadamard_groupsize]
        """
        idx = quantized.idx
        scales = quantized.scales
        device = scales.device
        x = self.grid(device=device)[idx.int()].flatten(start_dim=2)  # [b, mult, C / mult / d, d] -> [b, mult, C / mult]

        # Cut the padded values
        x = x[..., :self.hadamard_groupsize]

        x = (x * scales.unsqueeze(dim=2)).half()  # [b, mult, C / mult] * [b, mult, 1]
        
        x = hadamard_transform(x, scale=self.hadamard_scale).flatten(start_dim=1)  # [b, mult, C / mult] => [b, C]
        
        return x[:, :self.channel_size]
