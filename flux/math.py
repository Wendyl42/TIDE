import math
import torch
from torch import Tensor

def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=64):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=64):
    low = math.floor(find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)  # Clamp values just in case

def linear_ramp_mask(min, max, dim, freqs_dtype):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=freqs_dtype) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)

    return ramp_func

def get_kappa_t(timestep: float, lambda_s: float = 2.0, lambda_t: float = 2.0):
    return lambda_s * (timestep ** lambda_t)

def find_newbase_ntk(dim: float, base: float, scale: float, timestep: float, dype: bool):
    if dype:
        kappa = get_kappa_t(timestep)
        return base * scale ** ((kappa * dim) / (dim - 2))
    else:
        return base * scale ** (dim / (dim - 2))

def find_dominant_idx(freqs: Tensor, training_len: float):
    periods = (2 * torch.pi) / freqs
    diff = torch.abs(periods - training_len)
    return torch.argmin(diff).item()

def get_mscale(scale):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0

def get_default_temperature(scale):
    inv_sqrt_t = get_mscale(scale)
    return 1.0 / (inv_sqrt_t ** 2.0)

def dyheating(low_temperature: float, high_temperature:float, timestep:float, freqs: Tensor, alpha_low=3.0, alpha_high=0.8):
    alphas = alpha_low + (alpha_high - alpha_low) * freqs
    temps = high_temperature - ((high_temperature - low_temperature) * torch.pow(timestep, alphas))
    return torch.repeat_interleave(temps, repeats=2).unsqueeze(0)

def tuning_temperature(dytemp:str, low_temperature:float, high_temperature:float, timestep: float, freqs: Tensor, alpha_low, alpha_high):
    if dytemp == 'no':
        return 1.0 / math.sqrt(low_temperature)
    elif dytemp == 'dyheating':
        return 1.0 / dyheating(low_temperature, high_temperature, timestep, freqs, alpha_low, alpha_high).sqrt()