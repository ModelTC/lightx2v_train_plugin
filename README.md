# LightX2V 训练插件

模型代码放在独立子包中，通过 `PYTHONPATH` 暴露源码，并在 YAML 中显式启用：

```bash
export LIGHTX2V_TRAIN_ROOT=/data/wushuo/LightX2V-Train-Plugin
export LIGHTX2V_PLUGIN_ROOT=/data/wushuo/lightx2v_train_plugin
export PYTHONPATH="${LIGHTX2V_TRAIN_ROOT}/lightx2v_train:${LIGHTX2V_TRAIN_ROOT}:${LIGHTX2V_PLUGIN_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
```

```yaml
plugins:
  - lightx2v_train_plugin.wan22
```

插件包不需要安装。每次任务启动时加载选中的模块；修改源码后，下次启动生效。顶层包不导入所有模型。

## 开发检查

代码检查沿用 LightX2V 主仓的 pre-commit 配置：Ruff hook 固定为 `v0.11.0`，通用 hooks 为 `v4.5.0`，Python 目标版本为 3.11，行宽 200、缩进 4，并检查导入顺序、行尾空白、文件末尾换行、YAML/TOML/JSON 语法、大小写冲突、合并标记、调试语句和 3MB 文件大小限制。插件包名在 Ruff 中声明为 `lightx2v_train_plugin`。

```bash
pip install ruff pre-commit
pre-commit install
pre-commit run --all-files
```

`pre-commit install` 安装本地提交 hook；GitHub CI 会在 pull request 和推送到 `main` 时运行相同检查。`--all-files` 检查 Git 已跟踪文件；新增文件需要先 `git add`，或用 `pre-commit run --files path/to/file` 单独检查。

## Wan2.2 T2V A14B SFT

参考实验是 `/data/wushuo/exps/exps_1109/diffsynth_wan22_lora_8/outputs/runs/a10040g_20261110_first_epoch`。

首版以其 LoRA 配方为基线：两个专家分开训练、480×832、49 帧、20 个固定编码样本重复 100 次、8 卡每专家 250 次更新；rank/alpha 为 32/32，目标层为 `q,k,v,o,ffn.0,ffn.2`，AdamW 学习率 `1e-4`，weight decay `0.01`。前五次更新的学习率依次是 `2e-5、4e-5、6e-5、8e-5、1e-4`。

时间采样沿用 DiffSynth 的 1000 点、shift=5 时间表。high 抽取索引 `[0,417)`，low 抽取 `[417,1000)`，并使用完整时间表归一化的 loss 权重。`0.417` 是索引比例；推理时的 sigma 切换边界仍为 `0.875`。BF16 时间步、latent、noise、target 与参考路径保持一致。

分布式使用主项目已有 FSDP2；其参数分片、归约、数据顺序和显存表现与参考 ZeRO-2 不同。原生前向需要把 sigma 转回 timestep，这次往返的最大误差约为 `3.05e-5`；模型 kernel 和 VAE 解码路径也不同，不承诺逐位一致。

### 转换已有编码缓存

使用已有 DiffSynth 缓存可以保留原实验的 VAE/T5 输出，避免重编码带来的差异。转换器检查缓存完成标记、源 CSV 的 SHA256、样本索引和 tensor 形状，使用 `weights_only=True` 读取，保留 BF16 tensor，不修改源文件：

```bash
/app/miniconda/bin/python scripts/convert_diffsynth_cache.py \
  --source-dir /data/wushuo/exps/exps_1109/diffsynth_wan22_lora_8/outputs/cache/20702074f6bbf0d3/tensors \
  --metadata /data/wushuo/exps/exps_1109/diffsynth_wan22_lora_8/data/wan22_t2v_480p/metadata.csv \
  --output-dir /data/wushuo/lightx2v_train_plugin/outputs/cache \
  --repeat 100
```

输出目录必须为空。`cache_data.jsonl` 引用 20 个独立 tensor 文件；`train_data.jsonl` 重复引用它们 100 次，共 2000 条记录。这里不依赖主项目缓存数据集尚未支持的 `dataset_repeat`。8 卡、每卡 batch 1、累积 1 时，250 次更新刚好消费这些记录一次。

本机的 `outputs/cache` 已完成转换，20 组 latent/context 均与源缓存逐位一致，可以直接使用。重新转换时选择新的空目录，并通过 `WAN22_CACHE_DIR` 指向它。

`configs/wan22/cache.yaml` 也能交给主项目 `cache_data.py`，从原始视频生成原生缓存。它取开头 49 帧，但重新编码结果未与 DiffSynth 做逐 tensor 校验；对齐已有实验时优先使用上面的转换路径。

### 训练

在仓库根目录分别启动两个任务：

```bash
TRAIN_PYTHON=/app/miniconda/bin/python NPROC_PER_NODE=8 \
  bash scripts/train_wan22.sh

CONFIG="$PWD/configs/wan22/t2v_a14b_low_lora.yaml" \
  TRAIN_PYTHON=/app/miniconda/bin/python NPROC_PER_NODE=8 \
  bash scripts/train_wan22.sh
```

`TRAIN_PYTHON` 指向包含主项目依赖的现有 Python 环境。可通过 `WAN22_MODEL_ROOT`、`WAN22_CACHE_DIR`、`WAN22_OUTPUT_ROOT` 指定权重、转换后的缓存和输出目录。高低专家默认分别输出到 `outputs/high_lora` 和 `outputs/low_lora`。

训练配置使用 BF16 DiT/LoRA 和梯度检查点，模型前向期间把 autograd 保存的 tensor 卸载到 CPU。其保存范围与 DiffSynth 逐 block checkpoint offload 略有不同，通过梯度测试检查正确性。文本 dropout 和训练中生成关闭。`resume.auto_resume` 默认关闭；显式启用后只能恢复同一专家的训练目录。本次基线只提供 LoRA 配置，全量训练需要另行确定优化参数和资源。

### 双专家推理

```bash
TRAIN_PYTHON=/app/miniconda/bin/python NPROC_PER_NODE=8 \
  bash scripts/infer_wan22.sh --dry-run

TRAIN_PYTHON=/app/miniconda/bin/python NPROC_PER_NODE=8 \
  bash scripts/infer_wan22.sh --seeds 0 1
```

脚本依次为每条 prompt、每个 seed 启动独立推理，使四条 prompt 都使用 seeds 0/1，避免主项目默认按 prompt 索引递增 seed。默认配置使用 49 帧、480×832、24fps、50 步、CFG 5、sigma shift 5 和参考 negative prompt。通过 `WAN22_HIGH_LORA`、`WAN22_LOW_LORA` 指定两份权重，`WAN22_INFER_OUTPUT` 指定输出目录。生成基线时将这两个 LoRA 环境变量设为空。

这里复用主项目双专家推理和 FSDP2；参考实验的单卡逐层 CPU offload 与 tiled VAE 没有移植，实际显存和图像解码数值需要单独验证。

运行训练或生成前需使用具备主项目依赖及 GPU 访问的环境。

## Z-Image 图像 SFT / LoRA

通过 `plugins: [lightx2v_train_plugin.z_image]` 注册 `model.name: z_image` 和图像处理器，复用主项目的 flow matching trainer、梯度检查点、FSDP2、LoRA 导出及断点恢复。默认权重是 `/data/models/Z-Image`，使用包含 `ZImagePipeline` / `ZImageTransformer2DModel` 的 Diffusers。训练前离线计算 VAE latent 和文本条件，训练阶段只加载、更新生成用的 DiT，VAE 和 text encoder 不参与优化。

实现遵循官方接口：Qwen3 chat template、倒数第二层文本特征和去 padding；latent 编码为 `(z - shift_factor) * scaling_factor`，解码为 `z / scaling_factor + shift_factor`；DiT 输入 `t = 1 - sigma`，将其输出取负后与通用目标 `noise - latent` 比较。图像按面积缩放并中心裁剪，宽高对齐 16，保留横竖方向。缓存使用 VAE 分布的 mode，在线训练使用 sample；噪声、latent 状态和 loss 使用 FP32，DiT 前向使用 BF16。

### 准备图片和缓存

准备 `.jsonl`（每行一张图，图片路径相对 JSONL 所在目录，或使用绝对路径）：

```json
{"target_image":"images/0001.jpg","prompt":"一只坐在窗边的橘猫"}
{"target_image":"images/0002.jpg","prompt":"山间湖泊","target_height":512,"target_width":768}
```

从仓库根目录运行：

```bash
Z_IMAGE_METADATA=/path/to/train.jsonl \
  bash scripts/cache_z_image.sh
```

默认目标面积是 1024×1024；显式 `target_height/target_width` 优先。面积可在 `configs/z_image/cache.yaml` 的 `data.processor.target_area` 调整。缓存默认保存到 `outputs/z_image_base/cache`，脚本以 FP32 保存缓存；可用 `Z_IMAGE_CACHE_DIR` 改变位置。切换到 Z-Image 后重新生成缓存，默认目录与此前 Turbo 实验分开。未传 `--overwrite` 时主框架会复用已有 tensor；修改模型、图片或分辨率后，应使用新的缓存目录或显式传 `--overwrite` 重建。

### 训练和恢复

```bash
# 单卡 LoRA；默认 1000 步、rank/alpha=16/16、学习率 1e-4
bash scripts/train_z_image.sh

# 多卡 FSDP2
NPROC_PER_NODE=8 bash scripts/train_z_image.sh
```

`configs/z_image/lora.yaml` 使用 logit-normal 时间采样和静态 shift=6，与 Z-Image 权重的 scheduler 配置一致。默认 LoRA 目标为 `to_q,to_k,to_v,to_out.0,w1,w2,w3`；每卡物理 batch 为 1，通过 `gradient_accumulation_iters` 增大有效 batch。此适配器不支持 sequence parallel，检测到多个序列并行 rank 时会在加载权重前报错；也未实现 Wan 插件的 CPU checkpoint offload。

默认 `data.train.prompt_dropout_rate: 0.1`：缓存阶段提前编码空 prompt，训练时 10% 的样本使用空条件，不再运行 text encoder。该比例可调整；在线编码的 `image_dataset` 训练也支持 prompt dropout。

可设置 `Z_IMAGE_MODEL_ROOT`、`Z_IMAGE_CACHE_DIR`、`Z_IMAGE_OUTPUT_DIR`、`TRAIN_PYTHON`、`LIGHTX2V_TRAIN_ROOT` 和 `CONFIG`。例如 `CONFIG=/path/to/custom.yaml bash scripts/train_z_image.sh`。需要在线编码时，将训练配置的 `data.train` 改成 `image_dataset` 并指向原始 JSONL；缓存训练的默认配置关闭训练中生成。

默认输出 `outputs/z_image_base/lora/checkpoint-000000100/` 等目录，包含 `pytorch_lora_weights.safetensors` 和训练状态。把配置中的 `resume.auto_resume` 设为 `true` 可继续同一 Z-Image 训练，恢复时保持基础权重、world size 和 LoRA 配置一致；不要从此前 Turbo 的 checkpoint 恢复。全量 DiT SFT 可设置 `training.train_type: full`，需自行确定资源及优化参数；默认提供 LoRA 配方。

### 推理

验证 JSONL 可以只包含 `prompt`，例如 `{"prompt":"An astronaut ChalkBoardDrawing"}`。加载训练后的 LoRA：

```bash
Z_IMAGE_VALIDATION_METADATA=/path/to/val.jsonl \
Z_IMAGE_LORA=/path/to/checkpoint-000001000/pytorch_lora_weights.safetensors \
NPROC_PER_NODE=8 bash scripts/infer_z_image.sh
```

`configs/z_image/infer.yaml` 默认使用 1024×1024、50 步、静态 shift=6、原生 CFG 4，输出到 `outputs/z_image_base/inference`。`Z_IMAGE_INFER_OUTPUT` 可以改变输出目录；`Z_IMAGE_LORA` 留空时生成原模型结果。`seed: 42` 按 prompt 索引递增为 42、43 等；对照时使用同一元数据和 seed。未启用多卡时，脚本默认单卡。

这里的 50 步是主框架的 50 次有效 Euler 更新；当前 Diffusers `ZImagePipeline` 将终点 sigma=0 计入步数，对应其 `num_inference_steps=51`，与官方示例传入 50 步的网格不同。

这里的 `cfg_guidance_scale` 遵循 Z-Image 原生公式 `positive + scale * (positive - negative)`，因此默认原生值 `4.0` 等于标准 CFG 5。`negative_prompt` 默认为空，可在推理配置中设置；可通过 `CONFIG=/path/to/infer.yaml` 选择配置。该入口使用普通 flow matching SFT，不进行蒸馏。

模型和推理参数参见 [Z-Image 官方模型说明](https://huggingface.co/Tongyi-MAI/Z-Image) 和 [Diffusers 官方 Z-Image pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/z_image/pipeline_z_image.py)。

历史 Turbo 实验：2026-11-21 在 8 张 A100 40GB 上对 32 张图片完成 1000 步 LoRA，并生成原模型/LoRA 推理对照。归档位于本机 `/data/wushuo/exps/exps_1121/z_image_turbo_lora_dataset_v1_32_20261121_111037/`，不随代码仓库发布；该实验使用 Z-Image-Turbo 权重，不作为 Z-Image 基础模型的训练或质量验证。

## HunyuanVideo 1.5 T2V SFT

插件模块为 `lightx2v_train_plugin.hunyuan_video15`，模型名为 `hunyuan_video15_t2v`。提供 480p/720p T2V 的 LoRA 和全参数 flow matching SFT，默认加载原生权重目录 `/data/models/HunyuanVideo-1.5/transformer/480p_t2v`；720p 设置 `model.transformer_version: 720p_t2v`，并在缓存配置中调整视频尺寸。当前示例不涵盖 I2V、超分、蒸馏或 MeanFlow 变体。

先从原始视频生成独立的 HunyuanVideo 1.5 VAE、文本条件缓存；Wan 缓存不能混用。元数据支持 CSV、JSON、JSONL，使用 `video`、`prompt` 列，视频相对路径从元数据所在目录解析。例如 CSV：

```csv
video,prompt
videos/example.mp4,A cat walking in a garden
```

图片也可作为单帧 SFT 目标：缓存配置使用 `data.train.name: image_dataset`，JSONL 每行提供 `target_image` 和 `prompt`；`data.processor.height/width` 默认为 512。处理器按比例双线性缩放后中心裁剪，输出一帧 `[3,1,H,W]`。此路径学习画风与空间内容，不提供运动训练信号。LoRA 导出包含 rank/alpha 元数据，重载时保留缩放比例。

在仓库根目录执行：

```bash
HY15_METADATA=/path/to/metadata.csv \
  TRAIN_PYTHON=/app/miniconda/bin/python \
  bash scripts/cache_hunyuan_video15.sh

TRAIN_PYTHON=/app/miniconda/bin/python NPROC_PER_NODE=8 \
  bash scripts/train_hunyuan_video15.sh

CONFIG="$PWD/configs/hunyuan_video15/t2v_full.yaml" \
  TRAIN_PYTHON=/app/miniconda/bin/python NPROC_PER_NODE=8 \
  bash scripts/train_hunyuan_video15.sh
```

`HY15_MODEL_ROOT` 指向包含 `transformer/<version>`、`vae`、`text_encoder` 的原生模型根目录；`HY15_METADATA` 指定原始视频元数据；`HY15_OUTPUT_ROOT` 默认为仓库的 `outputs/hunyuan_video15`；`HY15_CACHE_DIR` 默认为 `${HY15_OUTPUT_ROOT}/cache`。缓存脚本默认单卡，可用 `NPROC_PER_NODE` 调整。缓存命令的额外参数传给主项目 `cache_data.py`，例如 `--overwrite`；训练配置可用 `CONFIG` 替换。

缓存配方为 49 帧、480×832、从视频开头取帧、BF16 存储，关闭文本 dropout。训练直接读取 `cache_data.jsonl`，因此不加载 VAE/文本编码器。默认使用 BF16、梯度检查点、8 卡 FSDP2，不启用 sequence parallel；LoRA rank/alpha 为 32/32，目标层覆盖图像与文本分支注意力的 Q/K/V/输出投影。

时间采样采用 logitnormal，并通过 `scheduler.time_shift_settings.time_shift_mu=3.0` 实施固定 shift；训练目标为 `noise - latent`。这里的 AdamW 配置是训练入口基线：LoRA 学习率 `1e-4`，全参数学习率 `1e-5`，均 warmup 100 步、训练 1000 步、每 100 步保存；不等同于官方 Muon 训练配方。正式训练前应按数据量和显存调整分辨率、步数与优化参数。输出分别位于 `${HY15_OUTPUT_ROOT}/lora`、`${HY15_OUTPUT_ROOT}/full`，默认不自动续训。

`configs/hunyuan_video15/cache.yaml` 只生成缓存；`t2v_lora.yaml`、`t2v_full.yaml` 分别通过 `training.method: hunyuan_video15_sft` 启动 LoRA 和全参数训练。该 trainer 在 checkpoint 中保存已解析的模型配置，恢复时核验模型根目录、`transformer_path` 和 `transformer_version`；切换 base 或 480p/720p 变体需使用新的输出目录。测试与短步验证不代表已经完成生产规模训练或质量评估。

全参数配置保留 FP32 主参数及 AdamW 状态，前向与 FSDP mixed precision 采用 BF16；其显存开销高于 LoRA。默认 480×832、49 帧的真实全参数训练显存与质量尚未验证，应先按资源缩短帧数/降低分辨率做短步验证。

全参数配置在每个 checkpoint 目录额外导出 `diffusion_pytorch_model.safetensors` 和配套 `config.json`，可直接用 `HunyuanVideo15Transformer3DModel.from_pretrained(checkpoint_dir, torch_dtype=torch.bfloat16)` 加载。导出 tensor 名称采用 Diffusers 格式，不是 Tencent 原生权重名称，不要直接覆盖原生权重。若从该导出权重开始新训练，可在 YAML 中设置 `model.transformer_path: /path/to/checkpoint`；`model.pretrained_model_name_or_path` 仍指向原模型根目录，提供 VAE 和文本编码器。分布式 `dist_state/` 用于同一并行规模、同一模型变体的训练恢复；恢复已有优化器状态需在相同输出目录启用 `resume.auto_resume`。LoRA 配置保存独立 LoRA 权重及优化器恢复状态。

默认 FSDP2 配置关闭 gradient_checkpointing_offload；当前 PyTorch/FSDP2 与 save_on_cpu 组合在反向阶段会出错，模型会拒绝同时启用。单卡非 FSDP 可按需启用 CPU 保存 tensor 卸载。末层不进入 latent loss 的文本输出投影和文本 MLP 被固定，避免无梯度分支导致优化器恢复失败。

联合注意力使用 `[B,1,1,N]` 的紧凑 padding mask，并在输出投影前清零无效 query，避免每层显式分配 `[B,1,N,N]` mask；前向和输入、参数梯度已与 Diffusers 原处理器对比。此优化只缩小 mask，未改变注意力的计算复杂度。

本机验证依赖为 `/app/miniconda/bin/python`、PyTorch 2.11.0、Diffusers 0.38.0、Transformers 5.9.0。权重转换在 CPU 内存进行，不改写原始 checkpoint；同时支持 Tencent 原生根目录与 Diffusers 本地目录。原生 Qwen 只加载语言塔，ByT5 加载 Glyph 专用权重与扩展词表。VAE 的 FP32 编码/解码已与原生实现做小尺寸数值对比。

2026-11-17 HunyuanVideo 验证：完整插件回归 42 项通过，其中 HunyuanVideo 新增 15 项；真实 8.3B DiT 单卡两步 LoRA 训练通过；正式训练入口的双卡 FSDP2 首步、DCP 保存、恢复至第二步全部通过，loss 为 0.707083 / 0.587269。小尺寸缓存为 5 帧、32×32，该验证不能推断默认 480×832、49 帧的显存或质量；双卡小模型另覆盖全参数保存恢复和 Diffusers 导出重载。真实双卡日志和摘要位于 work/hy15_smoke/real_fsdp_summary.json 及 real_fsdp_lora_fresh.log / real_fsdp_lora_resume.log。

2026-11-21 复核：完整 CPU 回归 57 项通过，包括紧凑注意力的前向/梯度一致性和恢复前的模型来源检查；三份 YAML 的插件注册、脚本语法及 Ruff 校验通过。原生 480p/720p DiT 与 VAE 的全部权重键名和形状均与转换后架构匹配。当前会话没有可用 CUDA，新增改动未重新进行真实多卡验证；上述 GPU 短步记录来自 11 月 17 日，整网转换前后的逐层数值一致性尚未验证。


## SDXL Base 1.0 UNet LoRA / 全参数训练

`plugins: [lightx2v_train_plugin.sdxl]` 提供 `model.name: sdxl`、`training.method: sdxl_sft` 和 `inference.method: sdxl_infer`。默认使用 `/data/models/stable-diffusion-xl-base-1.0`，训练范围是 UNet：LoRA 训练注意力的 `to_q,to_k,to_v,to_out.0`，全参数训练更新整个 UNet。VAE 和两个 CLIP 文本编码器只在独立缓存任务中加载，训练进程要求 `cache_dataset` 和 `inference.method: none`。

缓存记录未缩放的 VAE posterior mean/logvar，训练每次重新采样 posterior，再乘 `0.13025`。双 CLIP 倒数第二层特征拼为 `[1,77,2048]`，第二个 CLIP 提供 `[1,1280]` pooled 特征，并缓存原图尺寸、实际中心裁剪偏移及目标尺寸组成的六维 time IDs。图像保留横竖方向，默认面积约 1024²、宽高对齐 64。训练采用 1000 个离散均匀 timestep、DDPM 加噪和 epsilon MSE。

从仓库根目录运行，使用同一个 `SDXL_OUTPUT_ROOT`：

```bash
export SDXL_MODEL_ROOT=/data/models/stable-diffusion-xl-base-1.0
export SDXL_OUTPUT_ROOT=/data/wushuo/exps/exps_1121/sdxl
export SDXL_METADATA=/data/wushuo/data/dataset_v1_32/train.jsonl
export SDXL_VALIDATION_METADATA=/data/wushuo/data/dataset_v1_32/val.jsonl

# 独立生成训练 posterior + 双 CLIP 条件；验证集只需 prompt
NPROC_PER_NODE=8 bash scripts/cache_sdxl.sh
NPROC_PER_NODE=8 bash scripts/cache_sdxl.sh --data_split val

# LoRA：rank/alpha=16/16，lr=1e-4，默认 1000 次更新
NPROC_PER_NODE=8 bash scripts/train_sdxl.sh

# 全 UNet：FP32 参数、BF16 前向，lr=1e-5
CONFIG="$PWD/configs/sdxl/full.yaml" NPROC_PER_NODE=8 bash scripts/train_sdxl.sh

# 从缓存的验证条件生成；Euler 30 步，标准 CFG 5，seeds 42–49
SDXL_LORA="$SDXL_OUTPUT_ROOT/lora/checkpoint-000001000/pytorch_lora_weights.safetensors" \
  NPROC_PER_NODE=8 bash scripts/infer_sdxl.sh

# 全参数 checkpoint 导出为 Diffusers unet/，可直接用于推理
SDXL_UNET_ROOT="$SDXL_OUTPUT_ROOT/full/checkpoint-000001000/unet" \
  SDXL_INFER_OUTPUT="$SDXL_OUTPUT_ROOT/inference_full" \
  NPROC_PER_NODE=8 bash scripts/infer_sdxl.sh
```

脚本支持 `CONFIG`、`TRAIN_PYTHON`、`LIGHTX2V_TRAIN_ROOT`、`SDXL_CACHE_DIR`、`SDXL_VAL_CACHE_DIR` 和 `SDXL_INFER_OUTPUT`。LoRA 推理以 `SDXL_LORA` 指定权重，留空生成 base；全参数权重以 `SDXL_UNET_ROOT` 指定导出的 `unet/`。推理也读取缓存，因此不会加载文本编码器；变更验证 prompt、negative prompt 或尺寸需要重建验证缓存。

两种训练配置使用 8 卡 FSDP2 和梯度检查点；单卡时通过 `NPROC_PER_NODE=1` 启动，所需显存取决于训练模式与分辨率。每卡物理 batch 为 1，可用 `gradient_accumulation_iters` 增大有效 batch。LoRA 可训练参数保持 FP32，全参数训练保留 FP32 主参数。全参数 checkpoint 同时保存恢复用的分布式模型/优化器状态和推理用的 Diffusers `unet/`；LoRA 文件保留 rank/alpha 元数据。

将自定义配置中的 `resume.auto_resume` 设为 `true`，在原输出目录继续训练。恢复会检查 world size、训练类型、base/UNet 路径、LoRA 配置、DDPM 调度器及缓存清单身份；缺少完整导出或来源信息的 checkpoint 会被拒绝。SDXL 缓存也校验版本、base 路径与 VAE scaling。原始图片、权重或裁剪配方改变后，请换一个缓存目录或传 `--overwrite`，避免主框架复用旧缓存。当前不支持 sequence parallel。

2026-11-21 使用上述 32 图数据集及真实 SDXL Base 权重，在 8 张 A100 40GB 上分别完成 LoRA 和全 UNet 一轮训练（4 次更新），均成功保存后恢复到第 5 次更新。训练日志确认只加载 UNet；LoRA 导出 1120 个 FP32 tensor，全 UNet 导出 1680 个 FP32 tensor，均为有限值。另使用 8 条验证 prompt、seeds 42–49、Euler 30 步和 CFG 5 完成 base/LoRA/full 共 24 张 1024×1024 图；三组逐 prompt 初始噪声一致。CPU tiny UNet 验证覆盖 posterior 重新采样、epsilon 目标与真实 backward、rank=2/alpha=6 的 LoRA 保存/恢复及完整 UNet 导出重载。上述为训练链路短跑验证，不代表收敛或画风质量评估。配置、脚本、源码、日志、权重及对照图归档于 `/data/wushuo/exps/exps_1121/sdxl_unet_dataset_v1_32_20261121_1933/`。本机验证环境为 PyTorch 2.11.0+cu130、Diffusers 0.38.0、Transformers 5.9.0、PEFT 0.19.1。

同日补充完成 8 卡 1000 步 LoRA（32 图、250 轮，训练阶段约 19 分 43 秒），保留每 100 步的全部 10 份 checkpoint。全部 1120 个 LoRA tensor 为有限 FP32；8 卡重新加载最终 checkpoint 后，模型、AdamW step/moments 和 LR scheduler 均恢复到第 1000 步。固定 32 图、5 个 timestep、160 个配对噪声样本的 epsilon MSE 从 `0.139061` 降至 `0.133155`（约 4.25%）；这是训练集拟合指标。另完成 8 条验证 prompt 的 base/最终 LoRA 共 16 张 1024×1024 对照图，逐 prompt 初始噪声一致。19 项归档检查通过，脚本、配置、源码、日志、10 份 checkpoint、MSE 明细和对照图保存在 `/data/wushuo/exps/exps_1121/sdxl_lora_1000_dataset_v1_32_20261121_114915_utc/`。
