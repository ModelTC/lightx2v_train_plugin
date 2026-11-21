#!/usr/bin/env python3
"""Run each prompt with the same reference seeds through the main infer entry."""

import argparse
import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    seeds = args.seeds or ([int(os.environ["WAN22_SEED"])] if "WAN22_SEED" in os.environ else [0, 1])
    prompts = []
    for path in config["data"]["val"]["data_path"]:
        path = Path(path)
        if path.suffix not in {".txt", ".list"}:
            raise ValueError("Paired inference requires one prompt per line in a .txt or .list file")
        prompts.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if not prompts:
        raise ValueError("No validation prompts")
    output = Path(config["inference"]["output_dir"]).resolve()
    host = Path(os.environ.get("LIGHTX2V_TRAIN_ROOT", "/data/wushuo/LightX2V-Train-Plugin"))
    with tempfile.TemporaryDirectory(prefix="wan22-infer-") as directory:
        temporary = Path(directory)
        for index, prompt in enumerate(prompts):
            prompt_path = temporary / "prompt.txt"
            prompt_path.write_text(prompt + "\n", encoding="utf-8")
            for seed in seeds:
                selected = copy.deepcopy(config)
                selected["data"]["val"]["data_path"] = [str(prompt_path)]
                selected["inference"]["seed"] = seed
                selected["inference"]["output_dir"] = str(output / f"p{index + 1:02d}_seed{seed}")
                config_path = temporary / "infer.yaml"
                OmegaConf.save(OmegaConf.create(selected), config_path)
                print(f"prompt={index + 1} seed={seed} output={selected['inference']['output_dir']}", flush=True)
                if not args.dry_run:
                    subprocess.run([
                        sys.executable, "-m", "torch.distributed.run", "--standalone",
                        f"--nproc_per_node={os.environ.get('NPROC_PER_NODE', '8')}",
                        str(host / "lightx2v_train/infer.py"), "--config", str(config_path),
                    ], check=True)


if __name__ == "__main__":
    main()
