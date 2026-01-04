"""
Minimal attention capturing utilities for SD3 diffusers pipelines.

This replaces the original external `attention_map_diffusers` dependency by registering a custom
attention processor that records cross-attention probabilities (latents -> text tokens) per
transformer block and timestep.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Tuple

import torch

attn_maps: Dict[float, Dict[str, torch.Tensor]] = {}
_capture_state: Dict[str, object] = {"timestep": None, "latent_hw": None}


def _infer_hw(latent_tokens: int, latent_hw: Tuple[int, int] | None) -> Tuple[int, int]:
    if latent_hw and latent_hw[0] * latent_hw[1] == latent_tokens:
        return latent_hw
    side = int(math.sqrt(latent_tokens))
    if side * side == latent_tokens:
        return side, side
    return 1, latent_tokens


class CaptureJointAttnProcessor2_0:
    """
    Drop-in replacement for diffusers' JointAttnProcessor2_0 that also stores attention maps.
    """

    def __init__(self, layer_name: str):
        self.layer_name = layer_name

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor | None = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states
        batch_size = hidden_states.shape[0]

        # sample projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)
        if getattr(attn, "norm_k", None) is not None:
            key = attn.norm_k(key)

        # context projections
        context_len = 0
        if encoder_hidden_states is not None:
            context_len = encoder_hidden_states.shape[1]
            enc_q = attn.add_q_proj(encoder_hidden_states)
            enc_k = attn.add_k_proj(encoder_hidden_states)
            enc_v = attn.add_v_proj(encoder_hidden_states)

            enc_q = enc_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = enc_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = enc_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if getattr(attn, "norm_added_q", None) is not None:
                enc_q = attn.norm_added_q(enc_q)
            if getattr(attn, "norm_added_k", None) is not None:
                enc_k = attn.norm_added_k(enc_k)

            query = torch.cat([query, enc_q], dim=2)
            key = torch.cat([key, enc_k], dim=2)
            value = torch.cat([value, enc_v], dim=2)

        # attention
        scale = getattr(attn, "scale", None)
        attn_scores = torch.matmul(query, key.transpose(-1, -2))
        if scale is not None:
            attn_scores = attn_scores * scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = torch.softmax(attn_scores, dim=-1)

        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states_out = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)
        else:
            encoder_hidden_states_out = None

        # proj out
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # store attention maps (latents attending to text tokens)
        if encoder_hidden_states is not None and context_len > 0:
            timestep = _capture_state.get("timestep")
            latent_tokens = residual.shape[1]
            if timestep is not None:
                t_key = float(timestep)
                latent_hw = _capture_state.get("latent_hw")
                h, w = _infer_hw(latent_tokens, latent_hw if isinstance(latent_hw, tuple) else None)
                latent_probs = attn_probs[:, :, :latent_tokens, latent_tokens : latent_tokens + context_len]
                try:
                    latent_probs = latent_probs.reshape(batch_size, attn.heads, h, w, context_len)
                except Exception:
                    # fall back to flat if reshape fails
                    latent_probs = latent_probs.reshape(batch_size, attn.heads, latent_tokens, 1, context_len)
                attn_maps.setdefault(t_key, {})[self.layer_name] = latent_probs.detach().cpu()

        if encoder_hidden_states_out is None:
            return hidden_states
        return hidden_states, encoder_hidden_states_out


def init_pipeline(pipe):
    """
    Register capture processors on the SD3 transformer inside a diffusers pipeline.
    """
    attn_maps.clear()
    _capture_state["timestep"] = None
    _capture_state["latent_hw"] = None

    # wrap transformer forward to record timestep and latent grid size
    transformer = pipe.transformer
    orig_forward = transformer.forward

    def forward_with_capture(*args, **kwargs):
        if "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
        elif args:
            hidden_states = args[0]
        else:
            hidden_states = None

        timestep = kwargs.get("timestep", None)
        if timestep is None and len(args) > 1:
            timestep = args[1]
        if torch.is_tensor(timestep):
            _capture_state["timestep"] = timestep.flatten()[0].item()
        elif timestep is not None:
            _capture_state["timestep"] = float(timestep)

        if torch.is_tensor(hidden_states) and hidden_states.ndim >= 4:
            h, w = hidden_states.shape[-2], hidden_states.shape[-1]
            patch = getattr(transformer.config, "patch_size", 1) or 1
            _capture_state["latent_hw"] = (h // patch, w // patch)

        return orig_forward(*args, **kwargs)

    transformer.forward = forward_with_capture  # type: ignore[assignment]

    # replace attention processors
    processors = transformer.attn_processors
    new_processors: Dict[str, CaptureJointAttnProcessor2_0] = {}
    for name in processors.keys():
        match = re.search(r"transformer_blocks\.(\d+)", name)
        layer_name = f"layer-{match.group(1)}" if match else name
        new_processors[name] = CaptureJointAttnProcessor2_0(layer_name)

    transformer.set_attn_processor(new_processors)
    return pipe


def save_attention_maps(attn_maps: Dict, tokenizer, prompts, base_dir: str, unconditional: bool = True):
    """
    Light-weight PNG dump: for each timestep/layer, saves first-token mean attention heatmaps (avg over heads).
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    for timestep, layers in attn_maps.items():
        for layer_name, tensor in layers.items():
            data = tensor
            if unconditional and data.shape[0] % 2 == 0:
                data = data.chunk(2)[1]
            # data: batch x heads x H x W x tokens
            avg_heads = data.mean(dim=1)
            tok_idx = 0
            heat = avg_heads[..., tok_idx]
            for b_idx, prompt in enumerate(prompts):
                out_path = base / f"t{timestep:.3f}_{layer_name}_b{b_idx}_tok{tok_idx}.png"
                arr = heat[b_idx].detach().cpu().float().numpy()
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                import matplotlib.pyplot as plt

                plt.imsave(out_path, arr, cmap="magma")

