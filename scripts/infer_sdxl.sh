#!/usr/bin/env bash
set -euo pipefail

export LIGHTX2V_PLUGIN_ROOT="${LIGHTX2V_PLUGIN_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export LIGHTX2V_TRAIN_ROOT="${LIGHTX2V_TRAIN_ROOT:-/data/wushuo/LightX2V-Train-Plugin}"
export PYTHONPATH="${LIGHTX2V_TRAIN_ROOT}/lightx2v_train:${LIGHTX2V_TRAIN_ROOT}:${LIGHTX2V_PLUGIN_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

exec "${TRAIN_PYTHON:-/app/miniconda/bin/python}" -m torch.distributed.run \
  --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
  "${LIGHTX2V_TRAIN_ROOT}/lightx2v_train/infer.py" \
  --config "${CONFIG:-${LIGHTX2V_PLUGIN_ROOT}/configs/sdxl/infer.yaml}" "$@"
