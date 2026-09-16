from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                return fn(self, *args, **kwargs)

        return wrapper

    return decorator


def get_tensor_capacity(x: torch.Tensor, base: int = 2**30) -> float:
    return x.numel() * x.element_size() / base


def print_object_cuda_tensors_capacity(obj) -> str:
    res = 0.0
    out = []
    for name in obj.__dict__:
        item = getattr(obj, name)
        if isinstance(item, torch.Tensor) and item.is_cuda:
            cap = get_tensor_capacity(item)
            out.append(f"{name} {tuple(item.shape)} {item.dtype}: {cap} GiB")
            res += cap

    out.append(f"Total: {res}")
    return "\n".join(out)
