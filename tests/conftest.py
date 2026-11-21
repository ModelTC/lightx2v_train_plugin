"""Make the source-only plugin and its host available to all test modules."""

import os
import sys
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = Path(os.environ.get("LIGHTX2V_TRAIN_ROOT", PLUGIN_ROOT.parent / "LightX2V-Train-Plugin"))
sys.path[:0] = [str(PLUGIN_ROOT), str(TRAIN_ROOT / "lightx2v_train"), str(TRAIN_ROOT)]
