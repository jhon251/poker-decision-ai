"""P27-B.0 — 查看用户提供的空座帧（42b...png，1876×1400）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

OUT = Path("data/vision2/p27b_crops")
OUT.mkdir(parents=True, exist_ok=True)

FP = Path("data2/42b553c9468d1d1e55b9801bebad7b4f.png")
w, h, rgba = decode_png(FP.read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
print(f"frame: {w}x{h}")
(OUT / "z_user_full.png").write_bytes(
    encode_rgba_png(w, h, arr.tobytes()))
print("saved z_user_full.png")
