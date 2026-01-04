"""
批量跑长提示：读取 JSON 列表的 prompt，逐条生成图片，保存注意力张量并计算首 token 指标。
输出：每个样本一个子目录，包含 image、attn_maps.pt、tokens.json、first_token_metrics.csv、layer_head_heatmap.png、timestep_curve.png。
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sd35_attn_compare.capture import capture_attention
from sd35_attn_compare.analyze import compute_first_token_metrics, visualize_first_token


def load_prompts(path: Path, limit: int) -> List[str]:
    prompts = json.loads(path.read_text(encoding="utf-8"))
    prompts = prompts[:limit]
    return prompts


def build_parser():
    parser = argparse.ArgumentParser(description="Batch generate SD3.5 attention maps and first-token metrics.")
    parser.add_argument("--prompts_json", type=str, required=True, help="Path to JSON list of prompts")
    parser.add_argument("--out_root", type=str, required=True, help="Root output directory")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="bfloat16|float16|float32")
    parser.add_argument("--limit", type=int, default=100, help="Number of prompts to run")
    parser.add_argument("--first_token_index", type=int, default=0)
    parser.add_argument("--value_col", type=str, default="first_token_mean", choices=["first_token_mean", "first_token_share"])
    parser.add_argument("--skip_images", action="store_true", help="Do not save generated images")
    parser.add_argument("--dump_attn_png", action="store_true", help="Also dump per-token attention PNGs (large)")
    parser.add_argument("--dit_path", type=str, default=None, help="Path to DiT/MMDiT weights (file or folder)")
    parser.add_argument("--clip_l_path", type=str, default=None, help="Path to CLIP-L weights (file or folder)")
    parser.add_argument("--clip_g_path", type=str, default=None, help="Path to CLIP-G weights (file or folder)")
    parser.add_argument("--t5xxl_path", type=str, default=None, help="Path to T5-XXL weights (file or folder)")
    parser.add_argument("--vae_path", type=str, default=None, help="Path to VAE weights (file or folder)")
    return parser


def main():
    args = build_parser().parse_args()
    prompts_path = Path(args.prompts_json)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompts_path, args.limit)
    print(f"Loaded {len(prompts)} prompts, running generation...")

    for idx, prompt in enumerate(prompts):
        run_dir = out_root / f"{idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{idx+1}/{len(prompts)}] prompt length {len(prompt.split())} words -> {run_dir}")

        # 1) capture attention
        capture_attention(
            model_id=args.model_id,
            prompts=[prompt],
            out_dir=str(run_dir),
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            device=args.device,
            torch_dtype=args.dtype,
            save_images=not args.skip_images,
            dump_attn_png=args.dump_attn_png,
            unconditional=True,
            dit_path=args.dit_path,
            clip_l_path=args.clip_l_path,
            clip_g_path=args.clip_g_path,
            t5xxl_path=args.t5xxl_path,
            vae_path=args.vae_path,
        )

        # 2) load and analyze first-token metrics
        attn_maps = torch.load(run_dir / "attn_maps.pt", map_location="cpu")
        with open(run_dir / "tokens.json", "r", encoding="utf-8") as f:
            token_meta = json.load(f)
        tokens = token_meta.get("tokens", [])

        df = compute_first_token_metrics(
            attn_maps=attn_maps,
            tokens=tokens,
            unconditional=True,
            first_token_index=args.first_token_index,
        )
        df.to_csv(run_dir / "first_token_metrics.csv", index=False)
        visualize_first_token(df, run_dir, value_col=args.value_col)


if __name__ == "__main__":
    main()
