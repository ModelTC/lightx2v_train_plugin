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

### 检查

```bash
/app/miniconda/bin/python -m pytest tests -q
```

测试覆盖固定输入下的时间表、loss/梯度、学习率和缓存转换。通过 CPU 数值测试不等于已经完成真实 A14B 多卡训练；运行训练或生成前需使用具备主项目依赖及 GPU 访问的环境。
