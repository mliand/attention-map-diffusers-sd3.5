# SD3.5-Large Attention Comparison

对 Stable Diffusion 3.5 **large** 版本进行跨注意力图提取与首 token 注意力（“attention sink”）分析的最小仓库。提取部分复用 `attention_map_diffusers` 的 hook，分析与可视化部分参考了 `gated_attention` 中首 token 关注度的对比思路。

## 环境
- Python 3.10+
- GPU（推荐 bfloat16/float16）
- 依赖见 `requirements.txt`。跨注意力捕获逻辑已内置，无需额外安装。

```bash
pip install -r requirements.txt
```

## 用法
### 1) 运行并捕获注意力
生成图片、保存原始跨注意力张量与 token 信息，默认模型 id 可改为官方的 SD3.5 Large。
```bash
python -m sd35_attn_compare.capture \
  --model_id stabilityai/stable-diffusion-3.5-large \
  --prompt "A capybara holding a sign that reads Hello World." \
  --out_dir outputs/sd35-large-run \
  --steps 15 --guidance 4.5
```
产物：
- `images/`：生成的图片
- `attn_maps.pt`：逐 timestep/layer 的注意力张量（CPU）
- `tokens.json`：prompt 与 token 列表

### 2) 计算首 token 注意力指标
```bash
python -m sd35_attn_compare.analyze \
  --attn_path outputs/sd35-large-run/attn_maps.pt \
  --tokens_path outputs/sd35-large-run/tokens.json \
  --out_dir outputs/sd35-large-run
```
产物：
- `first_token_metrics.csv`：timestep/layer/head 维度的首 token 平均注意力（空间平均）
- `layer_head_heatmap.png`：按 layer×head 汇总的热力图（可视化首 token 下沉）
- `timestep_curve.png`：按 timestep 平均的首 token 注意力曲线

### 3) 快速组合运行
`scripts/run_sd35_analysis.py` 将捕获与分析串起来（需要已安装 `attention_map_diffusers`）：
```bash
python scripts/run_sd35_analysis.py \
  --model_id stabilityai/stable-diffusion-3.5-large \
  --prompt "A photo of a puppy wearing a hat." \
  --out_dir outputs/sd35-large-full \
  --steps 15 --guidance 4.5
```

### 4) 使用 sd-scripts 的 Gated Inference
已将 `sd-scripts_sd3.5/sd3_gated_inference.py` 拷贝到本仓库 `scripts/`，脚本会默认尝试从 `../sd-scripts_sd3.5` 读取 `library/*` 依赖，可用环境变量 `SD_SCRIPTS_PATH` 指定原仓库路径。
需要准备的权重文件（sd-scripts 版 SD3.5）：
- DiT/MMDiT 主体：`mmdit.safetensors`（`--ckpt_path`）
- CLIP-L / CLIP-G：`clip_l.safetensors`、`clip_g.safetensors`（`--clip_l`、`--clip_g`）
- T5-XXL：`text_encoder_3.safetensors`（`--t5xxl`）
```bash
python scripts/gated_sd3_infer.py \
  --ckpt_path /path/to/mmdit.safetensors \
  --clip_l /path/to/clip_l.safetensors \
  --clip_g /path/to/clip_g.safetensors \
  --t5xxl /path/to/text_encoder_3.safetensors \
  --prompt "A photo of a cat" \
  --gate_type headwise \
  --steps 50 --dtype bf16 \
  --output_dir outputs/gated_infer
```

### 5) 批量跑长提示（100 条）
使用 `prompts_long.json`（100 条、>300 tokens）批量生成，自动保存图片、注意力张量、首 token 指标与可视化：
```bash
python scripts/batch_capture_analyze.py \
  --prompts_json prompts_long.json \
  --out_root outputs/batch_run \
  --model_id stabilityai/stable-diffusion-3.5-large \
  --steps 15 --guidance 4.5 \
  --dtype bfloat16 \
  --dit_path /path/to/mmdit.safetensors \
  --clip_l_path /path/to/clip_l.safetensors \
  --clip_g_path /path/to/clip_g.safetensors \
  --t5xxl_path /path/to/text_encoder_3.safetensors \
  --vae_path /path/to/vae.safetensors
```
> 说明：`--dit_path/--clip_l_path/--clip_g_path/--t5xxl_path/--vae_path` 为可选覆盖项，接受目录或 safetensors 单文件。若不填，默认使用 `model_id` 中的对应权重。若已有完整本地 diffusers 目录（含 VAE），可直接把 `--model_id` 指向该目录并省略覆盖项。

## 设计要点
- **提取**：直接调用 `attention_map_diffusers.init_pipeline` 注册 hook，保存原始注意力张量，确保后续可重复分析。
- **首 token 指标**：对 `attn_map`（batch×head×H×W×tokens）在空间维求均值，取指定 token 索引（默认 0），得到 layer/head/timestep 维度的首 token 平均注意力。
- **可视化**：提供简单热力图与时间步曲线，便于观察 “attention sink” 程度。

## 注意
- 请根据自身环境调整 `device` 与 `torch_dtype`，默认在 CUDA 上用 bfloat16。
- 如果使用 CFG，`--unconditional` 选项决定是否自动丢弃无条件分支（默认 True）。
- 若首 token 不是位置 0，可用 `--first_token_index` 指定实际索引。
