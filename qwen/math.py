import math
import torch
from torch import Tensor

def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=64):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
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

def get_mscale(scale):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0

def get_default_temperature(scale):
    inv_sqrt_t = get_mscale(scale)
    return 1.0 / (inv_sqrt_t ** 2.0)

def cooling(low_temperature: float, high_temperature:float, timestep: float):
    return (high_temperature - low_temperature) * timestep + low_temperature

def heating(low_temperature: float, high_temperature:float, timestep: float):
    return (low_temperature - high_temperature) * timestep + high_temperature

def texture(low_temperature: float, high_temperature:float, freqs: Tensor):
    temps = (low_temperature - high_temperature) * freqs + high_temperature
    return temps

def abstraction(low_temperature: float, high_temperature:float, freqs: Tensor):
    temps = (high_temperature - low_temperature) * freqs + low_temperature
    return temps

def dyheating(low_temperature: float, high_temperature:float, timestep:float, freqs: Tensor, alpha_low=3.0, alpha_high=0.8):
    alphas = alpha_low + (alpha_high - alpha_low) * freqs
    temps = high_temperature - ((high_temperature - low_temperature) * torch.pow(timestep, alphas))
    return temps

def tuning_temperature(dytemp:str, low_temperature:float, high_temperature:float, timestep: float, freqs: Tensor, alpha_low, alpha_high):
    if dytemp == 'no':
        return 1.0 / math.sqrt(low_temperature)
    elif dytemp == 'cooling':
        return 1.0 / math.sqrt(cooling(low_temperature, high_temperature, timestep))
    elif dytemp == 'heating':
        return 1.0 / math.sqrt(heating(low_temperature, high_temperature, timestep))
    elif dytemp == 'texture':
        return 1.0 / texture(low_temperature, high_temperature, freqs).sqrt()
    elif dytemp == 'abstraction':
        return 1.0 / abstraction(low_temperature, high_temperature, freqs).sqrt()
    elif dytemp == 'dyheating':
        return 1.0 / dyheating(low_temperature, high_temperature, timestep, freqs, alpha_low, alpha_high).sqrt()