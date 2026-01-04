import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def compute_first_token_metrics(
    attn_maps: Dict,
    tokens: List[List[str]],
    unconditional: bool = True,
    first_token_index: int = 0,
) -> pd.DataFrame:
    """
    计算首 token 注意力：对空间维求均值，取指定 token 的平均注意力。
    返回包含 timestep/layer/batch/head 的 DataFrame。
    """
    records = []
    for timestep, layers in attn_maps.items():
        for layer_name, attn_map in layers.items():
            tensor = attn_map
            if unconditional and tensor.shape[0] % 2 == 0:
                tensor = tensor.chunk(2)[1]
            if tensor.dim() != 5:
                continue  # unexpected shape

            # batch x heads x H x W x tokens
            spatial_mean = tensor.mean(dim=(2, 3))  # batch x heads x tokens
            token_count = spatial_mean.shape[-1]
            token_idx = min(first_token_index, token_count - 1)
            first_token = spatial_mean[..., token_idx]
            token_avg = spatial_mean.mean(dim=-1).clamp_min(1e-8)

            for batch_idx in range(first_token.shape[0]):
                token_str = tokens[batch_idx][token_idx] if batch_idx < len(tokens) else f"token_{token_idx}"
                for head_idx in range(first_token.shape[1]):
                    value = first_token[batch_idx, head_idx].item()
                    records.append(
                        {
                            "timestep": int(timestep),
                            "layer": layer_name,
                            "batch": batch_idx,
                            "head": head_idx,
                            "first_token_mean": value,
                            "first_token_share": float(value / token_avg[batch_idx, head_idx]),
                            "token": token_str,
                        }
                    )
    return pd.DataFrame.from_records(records)


def _heatmap(df: pd.DataFrame, value_col: str, out_path: Path):
    layers = sorted(df["layer"].unique(), key=str)
    heads = sorted(df["head"].unique())
    heat = np.full((len(heads), len(layers)), np.nan, dtype=np.float32)

    grouped = df.groupby(["layer", "head"])[value_col].mean()
    for j, layer in enumerate(layers):
        for i, head in enumerate(heads):
            if (layer, head) in grouped:
                heat[i, j] = grouped[(layer, head)]

    fig, ax = plt.subplots(figsize=(0.6 * len(layers) + 2, 0.4 * len(heads) + 2))
    im = ax.imshow(heat, aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.02)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Head")
    ax.set_xticks(np.arange(len(layers)))
    ax.set_xticklabels(layers, rotation=90)
    ax.set_yticks(np.arange(len(heads)))
    ax.set_yticklabels(heads)
    ax.set_title(f"First-token attention ({value_col})")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _timestep_curve(df: pd.DataFrame, value_col: str, out_path: Path):
    curve = df.groupby("timestep")[value_col].mean().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(curve.index, curve.values, marker="o")
    ax.set_xlabel("Timestep")
    ax.set_ylabel(value_col)
    ax.set_title(f"First-token attention over timesteps ({value_col})")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def visualize_first_token(df: pd.DataFrame, out_dir: Path, value_col: str = "first_token_mean") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _heatmap(df, value_col, out_dir / "layer_head_heatmap.png")
    _timestep_curve(df, value_col, out_dir / "timestep_curve.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze first-token attention from captured attn_maps.")
    parser.add_argument("--attn_path", type=str, required=True, help="Path to attn_maps.pt")
    parser.add_argument("--tokens_path", type=str, required=True, help="Path to tokens.json")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for metrics and plots")
    parser.add_argument("--first_token_index", type=int, default=0, help="Index of the token to inspect")
    parser.add_argument(
        "--value_col",
        type=str,
        default="first_token_mean",
        choices=["first_token_mean", "first_token_share"],
        help="Which metric to visualize.",
    )
    parser.add_argument("--keep_unconditional", action="store_true", help="Do not drop unconditional branch")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    attn_maps = torch.load(args.attn_path, map_location="cpu")
    with open(args.tokens_path, "r", encoding="utf-8") as f:
        token_meta = json.load(f)

    tokens = token_meta.get("tokens", [])
    df = compute_first_token_metrics(
        attn_maps=attn_maps,
        tokens=tokens,
        unconditional=not args.keep_unconditional,
        first_token_index=args.first_token_index,
    )

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path / "first_token_metrics.csv", index=False)
    visualize_first_token(df, out_path, value_col=args.value_col)


if __name__ == "__main__":
    main()
