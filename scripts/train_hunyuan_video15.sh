#!/usr/bin/env bash
set -euo pipefail

export LIGHTX2V_PLUGIN_ROOT="${LIGHTX2V_PLUGIN_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export LIGHTX2V_TRAIN_ROOT="${LIGHTX2V_TRAIN_ROOT:-/data/wushuo/LightX2V-Train-Plugin}"
export HY15_MODEL_ROOT="${HY15_MODEL_ROOT:-/data/models/HunyuanVideo-1.5}"
export HY15_OUTPUT_ROOT="${HY15_OUTPUT_ROOT:-${LIGHTX2V_PLUGIN_ROOT}/outputs/hunyuan_video15}"
export HY15_CACHE_DIR="${HY15_CACHE_DIR:-${HY15_OUTPUT_ROOT}/cache}"
export PYTHONPATH="${LIGHTX2V_TRAIN_ROOT}/lightx2v_train:${LIGHTX2V_TRAIN_ROOT}:${LIGHTX2V_PLUGIN_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

exec "${TRAIN_PYTHON:-/app/miniconda/bin/python}" -m torch.distributed.run \
  --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
  "${LIGHTX2V_TRAIN_ROOT}/lightx2v_train/train.py" \
  --config "${CONFIG:-${LIGHTX2V_PLUGIN_ROOT}/configs/hunyuan_video15/t2v_lora.yaml}" "$@"
