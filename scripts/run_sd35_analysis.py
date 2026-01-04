import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sd35_attn_compare.capture import capture_attention
from sd35_attn_compare.analyze import compute_first_token_metrics, visualize_first_token


def build_parser():
    parser = argparse.ArgumentParser(description="End-to-end SD3.5 large attention capture + first-token analysis.")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-large")
    parser.add_argument("--prompt", type=str, nargs="+", required=True, help="Prompt(s) to render")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--first_token_index", type=int, default=0)
    parser.add_argument("--value_col", type=str, default="first_token_mean", choices=["first_token_mean", "first_token_share"])
    parser.add_argument("--skip_images", action="store_true")
    parser.add_argument("--dump_attn_png", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    capture_attention(
        model_id=args.model_id,
        prompts=args.prompt,
        out_dir=str(out_dir),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        device=args.device,
        torch_dtype=args.dtype,
        save_images=not args.skip_images,
        dump_attn_png=args.dump_attn_png,
        unconditional=True,
    )

    attn_maps = torch.load(out_dir / "attn_maps.pt", map_location="cpu")
    tokens_json = (out_dir / "tokens.json").read_text(encoding="utf-8")
    import json

    tokens = json.loads(tokens_json).get("tokens", [])
    df = compute_first_token_metrics(attn_maps, tokens, unconditional=True, first_token_index=args.first_token_index)
    df.to_csv(out_dir / "first_token_metrics.csv", index=False)
    visualize_first_token(df, out_dir, value_col=args.value_col)


if __name__ == "__main__":
    main()
