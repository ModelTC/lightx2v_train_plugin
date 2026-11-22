#!/usr/bin/env bash
set -euo pipefail

export LIGHTX2V_PLUGIN_ROOT="${LIGHTX2V_PLUGIN_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
export LIGHTX2V_TRAIN_ROOT="${LIGHTX2V_TRAIN_ROOT:-/data/wushuo/LightX2V-Train-Plugin}"
export PYTHONPATH="${LIGHTX2V_TRAIN_ROOT}/lightx2v_train:${LIGHTX2V_TRAIN_ROOT}:${LIGHTX2V_PLUGIN_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

cache_args=("$@")
data_split=train
for ((argument_index=0; argument_index<${#cache_args[@]}; argument_index++)); do
  case "${cache_args[$argument_index]}" in
    --data_split)
      data_split="${cache_args[$((argument_index + 1))]:?--data_split requires train or val}"
      ;;
    --data_split=*)
      data_split="${cache_args[$argument_index]#*=}"
      ;;
  esac
done
case "$data_split" in
  train) cache_output="${SDXL_CACHE_DIR:-${SDXL_OUTPUT_ROOT:-${LIGHTX2V_PLUGIN_ROOT}/outputs/sdxl}/cache}" ;;
  val) cache_output="${SDXL_VAL_CACHE_DIR:-${SDXL_OUTPUT_ROOT:-${LIGHTX2V_PLUGIN_ROOT}/outputs/sdxl}/cache_val}" ;;
  *) echo '--data_split must be train or val.' >&2; exit 2 ;;
esac

exec "${TRAIN_PYTHON:-/app/miniconda/bin/python}" -m torch.distributed.run \
  --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
  "${LIGHTX2V_TRAIN_ROOT}/lightx2v_train/cache_data.py" \
  --config "${CONFIG:-${LIGHTX2V_PLUGIN_ROOT}/configs/sdxl/cache.yaml}" \
  --output_dir "$cache_output" --save_dtype fp32 "$@"
