import torch
from collections import namedtuple

from .higgs import QuantizerBase


# FP4 E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit.
# Positive values (sign=0), codes 0-7:  0, 0.5, 1, 1.5, 2, 3, 4, 6
# Negative values (sign=1), codes 8-15: 0, -0.5, -1, -1.5, -2, -3, -4, -6
NVFP4_MAX = 6.0
_FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

# Midpoints between consecutive positive FP4 values used for nearest-value rounding.
_FP4_ABS_THRESHOLDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]

FP8_E4M3_MAX = 448

NVFP4QuantizedTensor = namedtuple("NVFP4QuantizedTensor", ["data", "scales"])


class NVFP4Quantizer(QuantizerBase):
    def __init__(self, block_size: int = 16, channel_size: int = 1024) -> None:
        """
        Block-wise NVFP4 (E2M1) quantization.
        :param block_size: number of elements per quantization block; each block shares one scale
        :param channel_size: original channel width, used to trim padding on dequantize
        """
        super().__init__()
        self.block_size = block_size
        self.channel_size = channel_size
        self._lut = torch.tensor(_FP4_LUT, dtype=torch.float32)
        self._thresholds = torch.tensor(_FP4_ABS_THRESHOLDS, dtype=torch.float32)

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> NVFP4QuantizedTensor:
        """
        x.shape = [B, C]
        Returns NVFP4QuantizedTensor:
          data:   uint8 [B, ceil(C/block_size)*block_size/2], two FP4 codes per byte
          scales: float32 [B, ceil(C/block_size)], per-block scale = max_abs / FP4_MAX
        """
        B, C = x.shape
        assert C == self.channel_size
        device = x.device
        x = x.to(torch.float32)

        pad = (self.block_size - C % self.block_size) % self.block_size
        if pad > 0:
            x = torch.nn.functional.pad(x, (0, pad))
        C_padded = x.shape[1]
        num_blocks = C_padded // self.block_size

        x_blocked = x.view(B, num_blocks, self.block_size)

        max_abs = x_blocked.abs().amax(dim=-1)                    # [B, num_blocks]
        scales = (max_abs / NVFP4_MAX).clamp(min=1e-8)            # [B, num_blocks]
        global_scale = FP8_E4M3_MAX / scales.max()
        # Cast scales to E4M3
        scales = (scales * global_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX) \
            .to(torch.float8_e4m3fn) \
            .to(torch.float32) \
            .div(global_scale)

        x_norm = (x_blocked / scales.unsqueeze(-1)).clamp(-NVFP4_MAX, NVFP4_MAX)

        # Determine sign bit and magnitude code for each element.
        sign = (x_norm < 0).to(torch.uint8)
        thresholds = self._thresholds.to(device)
        mag_code = torch.searchsorted(
            thresholds.contiguous(), x_norm.abs().reshape(-1).contiguous()
        ).to(torch.uint8).view(B, num_blocks, self.block_size)

        # 4-bit code: bits[3] = sign, bits[2:0] = magnitude index (0-7)
        codes = ((sign << 3) | mag_code).view(B, C_padded)        # [B, C_padded]

        # Pack two 4-bit codes per byte: low nibble = even index, high nibble = odd index.
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)

        return NVFP4QuantizedTensor(packed, scales)

    @torch.no_grad()
    def dequantize(self, quantized: NVFP4QuantizedTensor) -> torch.Tensor:
        """
        Returns float16 tensor of shape [B, channel_size].
        """
        packed = quantized.data    # [B, C_padded//2]
        scales = quantized.scales  # [B, num_blocks]
        device = scales.device
        B = packed.shape[0]

        lows  = (packed & 0x0F).to(torch.long)
        highs = ((packed >> 4) & 0x0F).to(torch.long)

        C_padded = packed.shape[1] * 2
        codes = torch.empty(B, C_padded, dtype=torch.long, device=device)
        codes[:, 0::2] = lows
        codes[:, 1::2] = highs

        lut = self._lut.to(device)
        values = lut[codes].view(B, scales.shape[1], self.block_size)
        values = (values * scales.unsqueeze(-1)).view(B, C_padded)

        return values[:, :self.channel_size].half()
