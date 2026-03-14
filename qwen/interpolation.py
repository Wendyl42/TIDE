import torch
import logging
from torch import Tensor
from dataclasses import dataclass
from .math import find_correction_range, linear_ramp_mask, get_default_temperature, tuning_temperature

TRAIN_SEQ_LEN = 64
MAX_TXT_TOKENS = 512

# logging.basicConfig(level=logging.ERROR)
rope_logger = logging.getLogger("rope_logger")
rope_logger.addHandler(logging.NullHandler())

@dataclass
class InterpolationOptions:
    interpolation: str = 'no'
    txt_interpolation: str = 'no' # 'no', 'clip', 'yarn`
    alpha: float = 1.0
    beta: float = 32.0
    low_temperature: float | None = None
    high_temperature: float = 1.0
    low_alpha: float = 0.6
    high_alpha: float = 0.2
    dype: bool = False
    dytemp: str = 'no'

def naive_rope(
    dim: int,
    pos: Tensor,
    theta: float,
    freqs_dtype=torch.float32,
):
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))
    )  # [D/2]

    freqs = torch.outer(pos, freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs)

    return freqs

def yarn_rope(
    dim: int,
    pos: Tensor,
    theta: int,
    S: float,
    alpha: float,
    beta: float,
    temperature: float | None,
    timestep: float,
    dytemp: str,
    low_alpha: float,
    high_alpha: float,
    freqs_dtype=torch.float32,
):
    assert dim % 2 == 0
    assert S >= 1.0
    assert alpha < beta

    if temperature is None:
        temperature = get_default_temperature(S)

    inv_freqs = theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)
    inv_sqrt_t = tuning_temperature(dytemp, temperature, 1.0, timestep, 1.0 / inv_freqs , low_alpha, high_alpha)

    freqs_extrapolation = 1.0 / inv_freqs
    freqs_interpolation = 1.0 / (S * inv_freqs)

    low, high = find_correction_range(beta, alpha, dim, theta, TRAIN_SEQ_LEN)
    interpolation_mask = linear_ramp_mask(low, high, dim // 2, freqs_dtype).to(pos.device)
    freqs = freqs_interpolation * interpolation_mask + freqs_extrapolation * (1 - interpolation_mask)

    freqs = torch.outer(pos, freqs)
    freqs = torch.polar(torch.ones_like(freqs) * inv_sqrt_t, freqs)
    return freqs
