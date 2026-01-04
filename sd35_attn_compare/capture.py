import argparse
import json
from pathlib import Path
from typing import Iterable, List

import torch
from diffusers import StableDiffusion3Pipeline

from attention_map_diffusers import attn_maps, init_pipeline, save_attention_maps


def _to_list(prompts: Iterable[str]) -> List[str]:
    return list(prompts) if not isinstance(prompts, str) else [prompts]


def _resolve_dtype(dtype_str: str):
    if dtype_str.lower() in ("bf16", "bfloat16"):
        return torch.bfloat16
    if dtype_str.lower() in ("fp16", "float16", "half"):
        return torch.float16
    return torch.float32


def capture_attention(
    model_id: str,
    prompts: Iterable[str],
    out_dir: str,
    num_inference_steps: int = 15,
    guidance_scale: float = 4.5,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    save_images: bool = True,
    dump_attn_png: bool = False,
    unconditional: bool = True,
) -> None:
    """
    运行 SD3.5 large，捕获跨注意力图并保存原始张量/元数据。
    """
    prompts = _to_list(prompts)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    attn_maps.clear()

    dtype = _resolve_dtype(torch_dtype)
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    pipe = init_pipeline(pipe)

    result = pipe(
        prompts,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )

    if save_images:
        image_dir = out_path / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for idx, image in enumerate(result.images):
            image.save(image_dir / f"{idx:02d}.png")

    torch.save(attn_maps, out_path / "attn_maps.pt")

    token_ids = pipe.tokenizer(prompts)["input_ids"]
    token_ids = token_ids if token_ids and isinstance(token_ids[0], list) else [token_ids]
    tokens = [pipe.tokenizer.convert_ids_to_tokens(t) for t in token_ids]
    meta = {
        "prompts": prompts,
        "token_ids": token_ids,
        "tokens": tokens,
        "unconditional": unconditional,
    }
    with open(out_path / "tokens.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if dump_attn_png:
        save_attention_maps(attn_maps, pipe.tokenizer, prompts, base_dir=str(out_path / "attn_maps_png"), unconditional=unconditional)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SD3.5 large and capture cross-attention maps.")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-large", help="Diffusers model id")
    parser.add_argument("--prompt", type=str, nargs="+", required=True, help="Prompt(s) to render")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--steps", type=int, default=15, help="num_inference_steps")
    parser.add_argument("--guidance", type=float, default=4.5, help="guidance_scale")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="torch dtype: bfloat16|float16|float32")
    parser.add_argument("--no_images", action="store_true", help="Skip saving generated images")
    parser.add_argument("--dump_attn_png", action="store_true", help="Also dump per-token attention PNGs (large)")
    parser.add_argument(
        "--no_unconditional",
        dest="unconditional",
        action="store_false",
        default=True,
        help="Keep unconditional branch instead of dropping it.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    capture_attention(
        model_id=args.model_id,
        prompts=args.prompt,
        out_dir=args.out_dir,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        device=args.device,
        torch_dtype=args.dtype,
        save_images=not args.no_images,
        dump_attn_png=args.dump_attn_png,
        unconditional=args.unconditional,
    )


if __name__ == "__main__":
    main()
