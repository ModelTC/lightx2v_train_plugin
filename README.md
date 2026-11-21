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

## Z-Image-Turbo 图像 SFT / LoRA

通过 `plugins: [lightx2v_train_plugin.z_image]` 注册 `model.name: z_image` 和图像处理器，复用主项目的 flow matching trainer、梯度检查点、FSDP2、LoRA 导出及断点恢复。使用包含 `ZImagePipeline` / `ZImageTransformer2DModel` 的 Diffusers；本机验证环境是 `/app/miniconda/bin/python`、Diffusers 0.38.0，权重已在 `/data/models/Z-Image-Turbo`。

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

默认目标面积是 1024×1024；显式 `target_height/target_width` 优先。面积可在 `configs/z_image/cache.yaml` 的 `data.processor.target_area` 调整。缓存默认保存到 `outputs/z_image/cache`，脚本以 FP32 保存缓存，避免 latent 量化；可用 `Z_IMAGE_CACHE_DIR` 改变位置。未传 `--overwrite` 时主框架会复用已有 tensor；修改模型、图片或分辨率后，应使用新的缓存目录或显式传 `--overwrite` 重建。

### 训练和恢复

```bash
# 单卡 LoRA；默认 1000 步、rank/alpha=16/16、学习率 1e-4
bash scripts/train_z_image.sh

# 多卡 FSDP2
NPROC_PER_NODE=8 bash scripts/train_z_image.sh
```

`configs/z_image/turbo_lora.yaml` 使用 logit-normal 时间采样和静态 shift=3，这是可调整的 SFT 基线配方。默认 LoRA 目标为 `to_q,to_k,to_v,to_out.0,w1,w2,w3`；每卡物理 batch 为 1，通过 `gradient_accumulation_iters` 增大有效 batch。此适配器不支持 sequence parallel，检测到多个序列并行 rank 时会在加载权重前报错；也未实现 Wan 插件的 CPU checkpoint offload。

可设置 `Z_IMAGE_MODEL_ROOT`、`Z_IMAGE_CACHE_DIR`、`Z_IMAGE_OUTPUT_DIR`、`TRAIN_PYTHON`、`LIGHTX2V_TRAIN_ROOT` 和 `CONFIG`。例如 `CONFIG=/path/to/custom.yaml bash scripts/train_z_image.sh`。需要在线编码时，将训练配置的 `data.train` 改成 `image_dataset` 并指向原始 JSONL；缓存训练的默认配置关闭训练中生成。

默认输出 `outputs/z_image/lora/checkpoint-000000100/` 等目录，包含 `pytorch_lora_weights.safetensors` 和训练状态。把配置中的 `resume.auto_resume` 设为 `true` 可继续同一输出目录，恢复时保持 world size 和 LoRA 配置一致。全量 DiT SFT 可设置 `training.train_type: full`，需自行确定资源及优化参数；此处提供和验证的是 LoRA 配方。

### 推理

验证 JSONL 可以只包含 `prompt`，例如 `{"prompt":"An astronaut ChalkBoardDrawing"}`。加载训练后的 LoRA：

```bash
Z_IMAGE_VALIDATION_METADATA=/path/to/val.jsonl \
Z_IMAGE_LORA=/path/to/checkpoint-000001000/pytorch_lora_weights.safetensors \
NPROC_PER_NODE=8 bash scripts/infer_z_image.sh
```

`configs/z_image/turbo_infer.yaml` 默认使用 1024×1024、30 步、静态 shift=3，输出到 `outputs/z_image/inference`。`Z_IMAGE_INFER_OUTPUT` 可以改变输出目录；`Z_IMAGE_LORA` 留空时生成原模型结果。`seed: 42` 按 prompt 索引递增为 42、43 等；对照时使用同一元数据和 seed。未启用多卡时，脚本默认单卡。

这里的 `cfg_guidance_scale` 遵循 Z-Image 原生公式 `positive + scale * (positive - negative)`，所以原生值 `1.0` 等于标准 CFG 2。评估 Turbo 少步效果时，使用自定义配置将 `num_inference_steps` 改成 `8`、`enable_cfg` 改成 `false`；主框架的 8 步对应官方 pipeline `num_inference_steps=9` 的 8 次有效前向。可通过 `CONFIG=/path/to/infer.yaml` 选择配置。

**Turbo 限制：** 这里实现常规 flow matching SFT，没有加载训练辅助 adapter 或进行再蒸馏。普通微调可能损害原模型的 8 步生成能力，不能把训练后的 LoRA 视为已保留 Turbo 加速。评估时可参考 30 步、CFG 2 的普通 SFT 设置；是否满足质量要求需用实际数据验证。需要保持少步能力时，应另外接入 differential LoRA / 蒸馏训练。参见 [DiffSynth 的 Z-Image 训练说明](https://github.com/modelscope/DiffSynth-Studio/blob/main/docs/en/Model_Details/Z-Image.md) 和 [Diffusers 官方 Z-Image pipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/z_image/pipeline_z_image.py)。

2026-11-21 使用真实 Z-Image-Turbo 权重、32 张图片、约 1024×1024 目标面积，在 8 张 A100 40GB 上完成 1000 步 LoRA（250 轮），训练耗时约 29 分钟，保留每 100 步的全部 10 份 checkpoint。最终 476 个 LoRA tensor 均为有限值，训练步数及分布式恢复文件检查通过。另用 8 条验证提示词、seeds 42–49 完成原模型/最终 LoRA × 8 步无 CFG/30 步标准 CFG 2 的 32 张 1024×1024 推理图，逐 prompt 初始噪声哈希一致；图像可读性和 SHA256 检查通过。实验的脚本、配置、源码快照、日志、权重及对照图保存在本机 `/data/wushuo/exps/exps_1121/z_image_turbo_lora_dataset_v1_32_20261121_111037/`，不随代码仓库发布。该结果验证训练和推理链路，不代表广泛的生成质量评估。
