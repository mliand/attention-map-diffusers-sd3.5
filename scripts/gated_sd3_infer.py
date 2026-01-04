"""
简化桥接：调用 sd-scripts_sd3.5 里的 sd3_gated_inference 进行加载与生成。
默认假设 sd-scripts_sd3.5 位于 ../sd-scripts_sd3.5；可用 --sd_scripts_path 覆盖。
"""

import argparse
import sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Bridge to run SD3.5 gated inference via sd3_gated_inference.py")
    parser.add_argument("--sd_scripts_path", type=str, default="../sd-scripts_sd3.5", help="Path to sd-scripts_sd3.5 repo")
    parser.add_argument("--ckpt_path", type=str, required=True, help="SD3.5 checkpoint (.safetensors)")
    parser.add_argument("--prompt", type=str, default="A photo of a cat")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--gate_type", type=str, default="headwise", choices=["headwise", "elementwise", "none"])
    parser.add_argument("--use_original_model", action="store_true", help="Use original MMDiT instead of gated")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--t5xxl_token_length", type=int, default=256)
    parser.add_argument("--apply_lg_attn_mask", action="store_true")
    parser.add_argument("--apply_t5_attn_mask", action="store_true")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs/gated_infer")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return parser.parse_args()


def main():
    args = parse_args()
    sd_scripts = Path(args.sd_scripts_path).resolve()
    if not sd_scripts.exists():
        raise FileNotFoundError(f"sd-scripts path not found: {sd_scripts}")

    # sd3_gated_inference imports library.* from its repo; add to sys.path
    if str(sd_scripts) not in sys.path:
        sys.path.insert(0, str(sd_scripts))

    from sd3_gated_inference import (
        load_gated_mmdit_for_inference,
        generate_image,
    )
    from library import sd3_utils, strategy_sd3
    from library.utils import load_safetensors
    from library.device_utils import get_preferred_device

    device = get_preferred_device()
    sd3_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    loading_device = "cpu" if args.offload else device

    if args.use_original_model:
        state_dict = load_safetensors(args.ckpt_path, loading_device, disable_mmap=True, dtype=sd3_dtype)
        clip_l = sd3_utils.load_clip_l(None, sd3_dtype, loading_device, state_dict=state_dict)
        clip_g = sd3_utils.load_clip_g(None, sd3_dtype, loading_device, state_dict=state_dict)
        t5xxl = sd3_utils.load_t5xxl(None, sd3_dtype, loading_device, state_dict=state_dict)
        vae = sd3_utils.load_vae(None, sd3_dtype, loading_device, state_dict=state_dict)
        mmdit = sd3_utils.load_mmdit(state_dict, sd3_dtype, loading_device)
    else:
        mmdit, state_dict = load_gated_mmdit_for_inference(
            args.ckpt_path,
            args.gate_type,
            sd3_dtype,
            loading_device,
        )
        clip_l = sd3_utils.load_clip_l(None, sd3_dtype, loading_device, state_dict=state_dict)
        clip_g = sd3_utils.load_clip_g(None, sd3_dtype, loading_device, state_dict=state_dict)
        t5xxl = sd3_utils.load_t5xxl(None, sd3_dtype, loading_device, state_dict=state_dict)
        vae = sd3_utils.load_vae(None, sd3_dtype, loading_device, state_dict=state_dict)

    for module in (clip_l, clip_g, t5xxl, vae, mmdit):
        module.to(sd3_dtype)
    if not args.offload:
        for module in (clip_l, clip_g, t5xxl, vae, mmdit):
            module.to(device)

    clip_l.eval()
    clip_g.eval()
    t5xxl.eval()
    mmdit.eval()
    vae.eval()

    tokenize_strategy = strategy_sd3.Sd3TokenizeStrategy(args.t5xxl_token_length)
    encoding_strategy = strategy_sd3.Sd3TextEncodingStrategy()

    generate_image(
        mmdit,
        vae,
        clip_l,
        clip_g,
        t5xxl,
        args.steps,
        args.prompt,
        args.seed,
        args.width,
        args.height,
        device,
        args.negative_prompt,
        args.cfg_scale,
        tokenize_strategy,
        encoding_strategy,
        args,
        sd3_dtype,
    )


if __name__ == "__main__":
    main()
