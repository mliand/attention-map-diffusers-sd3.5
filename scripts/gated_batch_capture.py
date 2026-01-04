"""
基于 sd-scripts 的 Gated MMDiT 推理，批量长提示，捕获首 token 注意力（平均到所有潜空间位置），保存图片、原始 logits 指标。
为避免大改源码，这里通过 monkey-patch 重写 attention 计算，获取 attention probs 并按 (layer, timestep) 存储。
输出：
- images/ : 生成图
- attn_first_token.pt : dict[timestep][layer] -> tensor (batch, heads) 首 token 平均注意力
- first_token_metrics.csv / heatmap / curve 同前
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

# --- sd-scripts path injection ---
DEFAULT_SD_SCRIPTS = Path(__file__).resolve().parents[2] / "sd-scripts_sd3.5"
SD_SCRIPTS_PATH = Path(os.environ.get("SD_SCRIPTS_PATH", DEFAULT_SD_SCRIPTS))
if SD_SCRIPTS_PATH.exists() and str(SD_SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SD_SCRIPTS_PATH))

from sd3_gated_inference import load_gated_mmdit_for_inference, generate_image  # type: ignore
from library import sd3_utils, strategy_sd3  # type: ignore
from library.device_utils import get_preferred_device  # type: ignore
import library.sd3_models as sd3_models  # type: ignore

from sd35_attn_compare.analyze import compute_first_token_metrics, visualize_first_token


# storage for first-token attention per timestep/layer
captured_first: Dict[int, Dict[int, torch.Tensor]] = {}
_capture_state = {"ctx_len": None, "layer_idx": None, "timestep": None}


def patched_attention(q, k, v, head_dim, mask=None, scale=None, mode="torch"):
    """
    Manual attention with weight capture. Returns same shape as original.
    """
    # layouts
    pre_attn_layout, post_attn_layout, _ = sd3_models.MEMORY_LAYOUTS[mode]
    qh = pre_attn_layout(q, head_dim)
    kh = pre_attn_layout(k, head_dim)
    vh = pre_attn_layout(v, head_dim)

    # qh,kh,vh: (B, heads, L, d) for torch/math; for xformers memory layout differs, we treat same here
    B, H, Lq, d = qh.shape
    scale_val = 1 / math.sqrt(d) if scale is None else scale
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale_val
    if mask is not None:
        scores = scores + mask
    attn_probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_probs, vh)
    out = post_attn_layout(out)

    ctx_len = _capture_state["ctx_len"]
    layer_idx = _capture_state["layer_idx"]
    timestep = _capture_state["timestep"]
    if ctx_len is not None and layer_idx is not None and timestep is not None:
        # attn_probs: (B, H, Lq, Lk). Queries: [ctx | latent], Keys: [ctx | latent]
        ctx_len_int = int(ctx_len)
        latent_probs = attn_probs[:, :, ctx_len_int:, :ctx_len_int]  # latents attending to text tokens
        first_tok = latent_probs[..., 0].mean(dim=2)  # (B, H)
        captured_first.setdefault(int(timestep), {})[int(layer_idx)] = first_tok.detach().cpu()

    return out


def wrap_blocks_for_capture(mmdit):
    """
    给每个 block 标注 layer_idx，并 monkey-patch sd3_models.attention。
    """
    for idx, block in enumerate(mmdit.joint_blocks):
        block.layer_idx = idx
    sd3_models.attention = patched_attention


def build_parser():
    p = argparse.ArgumentParser(description="Batch SD3.5 Gated inference with first-token attention capture.")
    p.add_argument("--ckpt_path", type=str, required=True, help="sd-scripts sd3.5 safetensors")
    p.add_argument("--prompts_json", type=str, required=True, help="JSON list of prompts")
    p.add_argument("--out_root", type=str, required=True, help="Output root dir")
    p.add_argument("--gate_type", type=str, default="headwise", choices=["headwise", "elementwise", "none"])
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--guidance", type=float, default=4.5)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--first_token_index", type=int, default=0)
    p.add_argument("--value_col", type=str, default="first_token_mean", choices=["first_token_mean", "first_token_share"])
    p.add_argument("--skip_images", action="store_true")
    return p


def load_prompts(path: Path, limit: int) -> List[str]:
    arr = json.loads(path.read_text(encoding="utf-8"))
    return arr[:limit]


def main():
    args = build_parser().parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(Path(args.prompts_json), args.limit)

    device = get_preferred_device()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    loading_device = "cpu"

    mmdit, state_dict = load_gated_mmdit_for_inference(
        args.ckpt_path,
        args.gate_type,
        dtype,
        loading_device,
    )
    clip_l = sd3_utils.load_clip_l(None, dtype, loading_device, state_dict=state_dict)
    clip_g = sd3_utils.load_clip_g(None, dtype, loading_device, state_dict=state_dict)
    t5xxl = sd3_utils.load_t5xxl(None, dtype, loading_device, state_dict=state_dict)
    vae = sd3_utils.load_vae(None, dtype, loading_device, state_dict=state_dict)

    for m in (clip_l, clip_g, t5xxl, vae, mmdit):
        m.to(dtype)
        m.to(device)
        m.eval()

    wrap_blocks_for_capture(mmdit)

    tokenize_strategy = strategy_sd3.Sd3TokenizeStrategy(256)
    encoding_strategy = strategy_sd3.Sd3TextEncodingStrategy()

    for idx, prompt in enumerate(prompts):
        run_dir = out_root / f"{idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        global captured_first
        captured_first = {}

        # set per-step hook via closure: we update timestep before mmdit call
        def tracked_generate():
            nonlocal prompt, run_dir
            # copy of generate_image loop with timestep tracking would be large;
            # instead, wrap mmdit forward to receive timestep from outer scope via global
            pass

        # We reuse generate_image from sd3_gated_inference but patch model_sampling loop via monkey-patch

        from sd3_gated_inference import do_sample as orig_do_sample  # type: ignore

        def do_sample_with_capture(*, height, width, initial_latent, seed, cond, neg_cond, mmdit, steps, cfg_scale, dtype, device, vae):
            # clone from orig_do_sample with timestep tracking
            if initial_latent is None:
                latent = torch.zeros(1, 16, height // 8, width // 8, device=device)
            else:
                latent = initial_latent
            latent = latent.to(dtype).to(device)
            # noise
            generator = torch.Generator(device)
            generator.manual_seed(seed)
            noise = torch.randn(latent.size(), dtype=latent.dtype, layout=latent.layout, generator=generator, device=device)

            model_sampling = sd3_utils.ModelSamplingDiscreteFlow(shift=3.0)
            sigmas = torch.linspace(
                model_sampling.timestep(model_sampling.sigma_max),
                model_sampling.timestep(model_sampling.sigma_min),
                steps,
                device=device,
            )
            sigmas = torch.cat([sigmas, torch.zeros_like(sigmas[:1])])
            sigmas = sigmas.to(device)

            sigma0 = sigmas[0]
            noise_scaled = model_sampling.noise_scaling(sigma0, noise, latent, False)

            c_crossattn = torch.cat([cond[0], neg_cond[0]]).to(device).to(dtype)
            y = torch.cat([cond[1], neg_cond[1]]).to(device).to(dtype)
            x = noise_scaled.to(device).to(dtype)

            for i in range(len(sigmas) - 1):
                sigma_hat = sigmas[i]
                timestep_val = model_sampling.timestep(sigma_hat).float()
                timestep = torch.FloatTensor([timestep_val, timestep_val]).to(device)
                # update capture state
                _capture_state["timestep"] = float(timestep_val)
                x_c_nc = torch.cat([x, x], dim=0)
                with torch.autocast(device_type=device.type, dtype=dtype):
                    model_output = mmdit(x_c_nc, timestep, context=c_crossattn, y=y)
                model_output = model_output.float()
                batched = model_sampling.calculate_denoised(sigma_hat, model_output, x)
                pos_out, neg_out = batched.chunk(2)
                denoised = neg_out + (pos_out - neg_out) * cfg_scale
                sigma_hat_dims = sigma_hat[(...,) + (None,) * (x.ndim - sigma_hat.ndim)]
                d = (x - denoised) / sigma_hat_dims
                dt = sigmas[i + 1] - sigma_hat
                x = x + d * dt
                x = x.to(dtype)

            latent = x
            latent = vae.process_out(latent)
            return latent

        # monkey-patch generate_image to use custom sampler
        from sd3_gated_inference import generate_image as gen_img_func  # type: ignore

        def generate_image_capture(*args, **kwargs):
            return gen_img_func(
                *args,
                **kwargs,
                sampler_fn=do_sample_with_capture,
                captured_first_out=captured_first,
                skip_image=kwargs.get("skip_image", False),
                run_dir=run_dir,
            )

        # simpler: replicate generate_image here to avoid changing signature
        # Prepare embeddings
        clip_l.to(device)
        clip_g.to(device)
        t5xxl.to(device)

        with torch.autocast(device_type=device.type, dtype=dtype), torch.no_grad():
            tokens_and_masks = tokenize_strategy.tokenize(prompt)
            lg_out, t5_out, pooled, l_attn_mask, g_attn_mask, t5_attn_mask = encoding_strategy.encode_tokens(
                tokenize_strategy, [clip_l, clip_g, t5xxl], tokens_and_masks, False, False
            )
            cond = encoding_strategy.concat_encodings(lg_out, t5_out, pooled)

            tokens_and_masks = tokenize_strategy.tokenize("")
            lg_out, t5_out, pooled, neg_l_attn_mask, neg_g_attn_mask, neg_t5_attn_mask = encoding_strategy.encode_tokens(
                tokenize_strategy, [clip_l, clip_g, t5xxl], tokens_and_masks, False, False
            )
            neg_cond = encoding_strategy.concat_encodings(lg_out, t5_out, pooled)

        mmdit.to(device)
        latent_sampled = do_sample_with_capture(
            height=args.height,
            width=args.width,
            initial_latent=None,
            seed=idx,
            cond=cond,
            neg_cond=neg_cond,
            mmdit=mmdit,
            steps=args.steps,
            cfg_scale=args.guidance,
            dtype=dtype,
            device=device,
            vae=vae,
        )
        vae.to(device)
        with torch.no_grad():
            image = vae.decode(latent_sampled)
        image = image.float()
        image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
        decoded_np = (255.0 * torch.moveaxis(image.cpu(), 0, 2)).numpy().astype("uint8")
        if not args.skip_images:
            from PIL import Image
            out_img = Image.fromarray(decoded_np)
            (run_dir / "images").mkdir(parents=True, exist_ok=True)
            out_img.save(run_dir / "images" / "img.png")

        # save captured first-token attention
        torch.save(captured_first, run_dir / "attn_first_token.pt")

        # build pseudo attn_maps compatible with analyzer: we store mean per head, no spatial dims
        # construct dummy attn_map tensor of shape (batch, heads, 1, 1, tokens)
        attn_maps_compat = {}
        for t, layers in captured_first.items():
            attn_maps_compat[t] = {}
            for layer_idx, first_tok_tensor in layers.items():
                attn_maps_compat[t][f"layer-{layer_idx}"] = first_tok_tensor[:, :, None, None, None]

        tokens = [tokenize_strategy.tokenizer.convert_ids_to_tokens(tokenize_strategy.tokenizer(prompt)["input_ids"])]
        df = compute_first_token_metrics(
            attn_maps=attn_maps_compat,
            tokens=tokens,
            unconditional=False,
            first_token_index=args.first_token_index,
        )
        df.to_csv(run_dir / "first_token_metrics.csv", index=False)
        visualize_first_token(df, run_dir, value_col=args.value_col)


if __name__ == "__main__":
    main()
