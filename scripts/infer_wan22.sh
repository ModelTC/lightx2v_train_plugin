#!/usr/bin/env bash
set -euo pipefail

export LIGHTX2V_PLUGIN_ROOT="${LIGHTX2V_PLUGIN_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export LIGHTX2V_TRAIN_ROOT="${LIGHTX2V_TRAIN_ROOT:-/data/wushuo/LightX2V-Train-Plugin}"
export PYTHONPATH="${LIGHTX2V_TRAIN_ROOT}/lightx2v_train:${LIGHTX2V_TRAIN_ROOT}:${LIGHTX2V_PLUGIN_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

exec "${TRAIN_PYTHON:-/app/miniconda/bin/python}" "${LIGHTX2V_PLUGIN_ROOT}/scripts/infer_wan22.py" \
  --config "${CONFIG:-${LIGHTX2V_PLUGIN_ROOT}/configs/wan22/dual_infer.yaml}" "$@"
