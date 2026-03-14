# TIDE

Code release for "TIDE: Text-Informed Dynamic Extrapolation with Step-Aware Temperature Control for Diffusion Transformers"

<div align="center">
  <img src="assets/collage.jpg" alt="TIDE Results" width="100%">
</div>

## Installation

Create a conda environment and install dependencies:

```bash
conda create -n tide python=3.10
conda activate tide
pip install -r requirements.txt
```

## Usage

Generate ultra-high resolution images with TIDE using the `run.py` script:

```bash
python run.py --prompt "Your text prompt here"
```

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--prompt` | Mountain Lake | Text prompt for image generation |
| `--height` | 4096 | Target height in pixels (should be a multiple of 16) |
| `--width` | 4096 | Target width in pixels (should be a multiple of 16) |
| `--steps` | 28 | Number of sampling steps |
| `--seed` | 0 | Seed for sampling |
| `--method` | `yarn` | Position encoding method: `no`, `ntk`, `ntkbypart`, or `yarn` |
| `--disable_dype` | False | Disable DyPE, which is our baseline (enabled by default) |
| `--disable_tide` | False | Disable TIDE (enabled by default) |
| `--dir` | "outputs" | Output directory |
| `--tiled_vae` | False | Enable Tiled VAE for saving CUDA memory (disabled by default) |
| `--cpu_offloading` | False | Enable CPU Offloading for saving CUDA memory (disabled by default) |

**Examples:**

```bash
# Generate 4K image with our default settings (YARN + DyPE + TIDE)
python run.py

# Use the baseline in our paper (YaRN + DyPE, no TIDE)
python run.py --method yarn --disable_tide

# Use pure FLUX.1 without any extrapolation method
python run.py --method no --disable_dype --disable_tide --width 2048 --height 2048
```

For Qwen-Image generation, use the run_qwen.py script. It accepts similar parameters, but does not support the `--disable_dype` parameter, and only support `no` and `yarn` for `--method` parameter.