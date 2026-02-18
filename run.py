import torch
import os
import math
import argparse

from PIL.PngImagePlugin import PngInfo

from flux.transformer_flux import FluxTransformer2DModel
from flux.pipeline_flux import FluxPipeline
from flux.interpolation import InterpolationOptions



def get_attn_mask(width: int, height: int, device, dtype):
    image_tokens = int((width / 16) * (height / 16))
    TEXT_TOKENS = 512
    custom_mask = torch.zeros(1, 1, 1, TEXT_TOKENS + image_tokens)
    custom_mask[:,:,:,:TEXT_TOKENS] = math.log(width / 1024.0) + math.log(height / 1024.0)
    custom_mask = custom_mask.to(device=device, dtype=dtype)
    return custom_mask

def main():
    parser = argparse.ArgumentParser(
        description='TIDE (based on FLUX.1 dev)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default="A starry night sky filled with countless stars and the Milky Way reflects on a still lake, surrounded by dramatic snow-capped mountains under a deep blue hue",
        help='Text prompt for image generation'
    )
    parser.add_argument('--height', type=int, default=4096, help='height of the sample in pixels (should be a multiple of 16)')
    parser.add_argument('--width', type=int, default=4096, help='width of the sample in pixels (should be a multiple of 16)')
    parser.add_argument('--steps', type=int, default=28, help='number of sampling steps')
    parser.add_argument('--seed', type=int, default=0, help='seed for sampling')
    parser.add_argument(
        '--method',
        type=str,
        choices=['no', 'ntk', 'ntkbypart', 'yarn'],
        default='yarn',
        help='interpolation method for positional embeddings (no, ntk, ntkbypart, or yarn)'
    )
    parser.add_argument('--disable_dype', action='store_true', help='disable DyPE (Issachar et al., 2025)')
    parser.add_argument('--disable_tide', action='store_true', help='disable TIDE (our method)')
    parser.add_argument('--dir', type=str, default="outputs", help='output directory')
    parser.add_argument('--tiled_vae', action='store_true', help='enable Tiled VAE for saving CUDA memory')
    parser.add_argument('--cpu_offloading', action='store_true', help='enable CPU Offloading for saving CUDA memory')

    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    method_name = args.method
    if not args.disable_dype:
        method_name = f"dy-{method_name}"
    if not args.disable_tide:
        method_name = f"tide-{method_name}"

    filename = f'seed_{args.seed}_method_{method_name}_res_{args.width}x{args.height}.png'

    # Save image with descriptive filename
    interpolation_opts = InterpolationOptions(
        interpolation=args.method,
        dype=not args.disable_dype,
        dytemp="no" if args.disable_tide else "dyheating" # Dynamic Temperature Control
    )

    # Load transformer with DyPE configuration
    transformer = FluxTransformer2DModel.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        interpolation_opts = interpolation_opts,
    )

    # Initialize pipeline
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        transformer=transformer,
        torch_dtype=torch.bfloat16
    )

    device = "cuda"

    if args.cpu_offloading:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    if args.tiled_vae:
        pipe.enable_vae_tiling()

    # Text Anchoring
    pixel_count = args.width * args.height
    if args.disable_tide or pixel_count <= 1024 * 1024:
        my_kwargs = None
    else:
        custom_mask = get_attn_mask(args.width, args.height, device, pipe.dtype)
        my_kwargs = {
            "attention_mask": custom_mask,
        }

    # Generate image
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        guidance_scale=3.5,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        num_inference_steps=args.steps,
        joint_attention_kwargs=my_kwargs,
        shift_mode="log"
    ).images[0]

    # metadata
    metadata = PngInfo()
    metadata.add_text("prompt", args.prompt)
    metadata.add_text("seed", str(args.seed))

    image.save(f"{args.dir}/{filename}", pnginfo=metadata)
    print(f"✓ Image saved to: {args.dir}/{filename}")

if __name__ == "__main__":
    main()
