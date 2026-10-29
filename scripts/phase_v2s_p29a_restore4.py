"""HeroCards 恢复 — 8bf8 帧复制入 evidence（provenance 可追溯）。"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

SRC = Path("data2/8bf82f134651db0add626be0304a2ff1.png")
DST = Path("data/vision2/coinpoker_evidence") / SRC.name
if not DST.exists():
    shutil.copy2(SRC, DST)
    print(f"copied -> {DST} ({DST.stat().st_size} bytes)")
else:
    print(f"already: {DST}")

from poker_decision_ai.vision2.pixel.realclient.gt_loader import (  # noqa: E402
    validate_gt)
gt = __import__("json").loads(Path(
    "data/vision2/coinpoker_1876_gt_herocards.json").read_text(
    encoding="utf-8"))
print(f"validate_gt errors: {validate_gt(gt, 'data/vision2/coinpoker_evidence')}")
