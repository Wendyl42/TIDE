import torch
from torch import Tensor
from dataclasses import dataclass
from .math import find_correction_range, linear_ramp_mask, get_kappa_t, find_newbase_ntk, get_default_temperature, tuning_temperature

TRAIN_SEQ_LEN = 64

@dataclass
class InterpolationOptions:
    interpolation: str = 'no'
    alpha: float = 1.0
    beta: float = 32.0
    low_temperature: float | None = None
    high_temperature: float = 1.0
    low_alpha: float = 0.6
    high_alpha: float = 0.2
    dype: bool = False
    dytemp: str = 'no'
    train_seq_len = 64

def rope(
    dim: int,
    pos: Tensor, # [seq_len]
    theta: float, # base, default 10000.0
    opts: InterpolationOptions,
    timestep: float = 1.0,
    freqs_dtype=torch.float32
):
    assert dim % 2 == 0

    max_pos = torch.max(pos).item()
    infer_seq_len = max_pos + 1

    if infer_seq_len <= opts.train_seq_len or opts.interpolation == "no":
        return naive_rope(dim, pos, theta, freqs_dtype)

    S = infer_seq_len / opts.train_seq_len # S = L' / L
    if opts.interpolation == "pi":
        return pi_rope(dim, pos, theta, S, freqs_dtype)
    elif opts.interpolation == "ntk":
        return ntk_rope(dim, pos, theta, S, timestep, opts, freqs_dtype)
    elif opts.interpolation == "ntkbypart":
        return ntkbypart_rope(dim, pos, theta, S, opts, timestep, freqs_dtype)
    elif opts.interpolation == "yarn":
        return yarn_rope(dim, pos, theta, S, opts, timestep, freqs_dtype)

def naive_rope(
    dim: int,
    pos: Tensor,
    theta: float,
    freqs_dtype,
):
    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))
    )  # [D/2]

    freqs = torch.outer(pos, freqs)  # [seq_len, D/2]

    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()

    freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    return freqs_cos, freqs_sin

def pi_rope(
    dim: int,
    pos: Tensor,
    theta: float,
    S: float,
    freqs_dtype,
):
    assert S >= 1.0

    pos = pos / S

    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))
    )  # [D/2]

    freqs = torch.outer(pos, freqs)  # [seq_len, D/2]

    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()

    freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    return freqs_cos, freqs_sin

def ntk_rope(
    dim: int,
    pos: Tensor,
    theta: float,
    S: float,
    timestep: float,
    opts: InterpolationOptions,
    freqs_dtype,
):
    assert S >= 1.0
    theta = find_newbase_ntk(float(dim), theta, S, timestep, opts.dype)

    freqs = (
        1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))
    )  # [D/2]

    freqs = torch.outer(pos, freqs)  # [seq_len, D/2]

    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()

    freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    return freqs_cos, freqs_sin

def ntkbypart_rope(
    dim: int,
    pos: Tensor,
    theta: int,
    S: float,
    opts: InterpolationOptions,
    timestep: float,
    freqs_dtype,
):
    assert S >= 1.0

    if opts.dype:
        kappa = get_kappa_t(timestep)
        # Warning: DyPE's code implementation is different from its paper implementation. We follow their code
        alpha = opts.alpha ** kappa
        beta = opts.beta ** kappa
    else:
        alpha = opts.alpha
        beta = opts.beta

    inv_freqs = theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)
    freqs_extrapolation = 1.0 / inv_freqs
    freqs_interpolation = 1.0 / (S * inv_freqs)

    low, high = find_correction_range(beta, alpha, dim, theta, opts.train_seq_len)
    interpolation_mask = linear_ramp_mask(low, high, dim // 2, freqs_dtype).to(pos.device)
    freqs = freqs_interpolation * interpolation_mask + freqs_extrapolation * (1 - interpolation_mask)

    freqs = torch.outer(pos, freqs) # [seq_len, D/2]

    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()

    freqs_cos = freqs.cos().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1, output_size=freqs.shape[1] * 2).float()  # [S, D]
    return freqs_cos, freqs_sin

def yarn_rope(
    dim: int,
    pos: Tensor,
    theta: int,
    S: float,
    opts: InterpolationOptions,
    timestep: float,
    freqs_dtype,
):
    assert S >= 1.0

    temperature = get_default_temperature(S) if opts.low_temperature is None else opts.low_temperature

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim))
    inv_sqrt_t = tuning_temperature(opts.dytemp, temperature, 1.0, timestep, freqs, opts.low_alpha, opts.high_alpha)

    freq_cos, freq_sin = ntkbypart_rope(dim, pos, theta, S, opts, timestep, freqs_dtype)
    return freq_cos * inv_sqrt_t, freq_sin * inv_sqrt_t
