from qwen.transformer_qwenimage import QwenImageTransformer2DModel
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from qwen.interpolation import InterpolationOptions, MAX_TXT_TOKENS

from PIL.PngImagePlugin import PngInfo

import math
import torch
import argparse
import os

def get_attn_mask(width: int, height: int, device, dtype):
    image_tokens = int((width / 16) * (height / 16))
    custom_mask = torch.zeros(1, 1, 1, MAX_TXT_TOKENS + image_tokens)
    custom_mask[:,:,:,:MAX_TXT_TOKENS] = math.log(width / 1024.0) + math.log(height / 1024.0)
    custom_mask = custom_mask.to(device=device, dtype=dtype)
    return custom_mask

def main():
    parser = argparse.ArgumentParser(
        description='TIDE (based on Qwen-Image)'
    )
    parser.add_argument('--prompt', type=str, default="A starry night sky filled with countless stars and the Milky Way reflects on a still lake, surrounded by dramatic snow-capped mountains under a deep blue hue.", help='Prompt')
    parser.add_argument('--height', type=int, default=3072, help='height of the sample in pixels (should be a multiple of 16)')
    parser.add_argument('--width', type=int, default=3072, help='width of the sample in pixels (should be a multiple of 16)')
    parser.add_argument('--steps', type=int, default=40, help='inference steps')
    parser.add_argument('--seed', type=int, default=0, help='seed for sampling')

    parser.add_argument('--method', type=str, choices=['no', 'yarn'], default='yarn', help='interpolation method for positional embeddings (no, ntk, ntkbypart, or yarn)')
    parser.add_argument('--disable_tide', action='store_true', help='disable TIDE (our method)')

    parser.add_argument('--dir', type=str, default="outputs", help='output directory')
    parser.add_argument('--tiled_vae', action='store_true', help='enable tiled vae for saving GPU memory')
    parser.add_argument('--cpu_offloading', action='store_true', help='enable CPU offloading for saving CUDA memory')

    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    method_name = args.method
    if not args.disable_tide:
        method_name = f"tide-{method_name}"

    filename = f'seed_{args.seed}_method_{method_name}_res_{args.width}x{args.height}.png'

    interpolation_opts = InterpolationOptions(
        interpolation=args.method,
        dytemp="no" if args.disable_tide else "dyheating"
    )

    transformer = QwenImageTransformer2DModel.from_pretrained(
        "Qwen/Qwen-Image",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        interpolation_opts = interpolation_opts
    )
    pipe = QwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image",
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
    if args.disable_tide or pixel_count <= 1328 * 1328:
        my_kwargs = None
    else:
        custom_mask = get_attn_mask(args.width, args.height, device, pipe.dtype)
        my_kwargs = {
            "attention_mask": custom_mask,
        }


    # Generate image
    image = pipe(
        prompt=args.prompt,
        negative_prompt="",
        height=args.height,
        width=args.width,
        true_cfg_scale=4.0,
        generator=torch.Generator(device=device).manual_seed(args.seed),
        num_inference_steps=args.steps,
        attention_kwargs=my_kwargs
    ).images[0]

    # metadata
    metadata = PngInfo()
    metadata.add_text("prompt", args.prompt)
    metadata.add_text("seed", str(args.seed))

    image.save(f"{args.dir}/{filename}", pnginfo=metadata)
    print(f"✓ Image saved to: {args.dir}/{filename}")

if __name__ == "__main__":
    main()
