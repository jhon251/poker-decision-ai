"""P25-B.3 — 裁剪 1876 折叠座牌盒/负例座牌盒 供图像模型目视复核。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p25b3_crops")
OUT.mkdir(parents=True, exist_ok=True)

# 1876 座锚点（1848 锚点映射）
ANCH = {"S1": (255, 310), "S2": (918, 210), "S3": (1578, 320),
        "S4": (218, 868), "S5": (1618, 868), "S6": (945, 1160)}

FRAMES = {
    "f01": "BUTTON_CHANGED_167_1787992682749_full.png",
    "f02": "POT_CHANGED_12_1787995290794_full.png",
    "f03": "POT_CHANGED_102_1787992505794_full.png",
    "f04": "POT_CHANGED_13_1787992279195_full.png",
    "f05": "POT_CHANGED_139_1787992592538_full.png",
    "f06": "BUTTON_CHANGED_598_1787993753587_full.png",
}

# 座牌盒区域：锚点上下各 extension
def crop(frame_path, ax, ay, name, ex=230, ey_up=150, ey_dn=200):
    w, h, rgba = decode_png((EV / frame_path).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    x0, x1 = max(0, ax - ex), min(w, ax + ex)
    y0, y1 = max(0, ay - ey_up), min(h, ay + ey_dn)
    sub = arr[y0:y1, x0:x1]
    png = encode_rgba_png(sub.shape[1], sub.shape[0], sub.tobytes())
    (OUT / f"{name}.png").write_bytes(png)
    return f"{name}: crop=({x0},{y0},{x1},{y1}) size={sub.shape[1]}x{sub.shape[0]}"


jobs = [
    ("f01_S5_folded", FRAMES["f01"], *ANCH["S5"]),   # 折叠正例
    ("f01_S6_folded", FRAMES["f01"], *ANCH["S6"]),   # 折叠正例（hero）
    ("f01_S1_active", FRAMES["f01"], *ANCH["S1"]),   # 活跃负例
    ("f01_S3_away", FRAMES["f01"], *ANCH["S3"]),     # away 负例
    ("f03_S1_folded", FRAMES["f03"], *ANCH["S1"]),   # 折叠正例（不同帧）
    ("f03_S3_folded", FRAMES["f03"], *ANCH["S3"]),
    ("f03_S2_allin", FRAMES["f03"], *ANCH["S2"]),    # allin 负例
    ("f05_S2_folded", FRAMES["f05"], *ANCH["S2"]),
    ("f05_S3_disc", FRAMES["f05"], *ANCH["S3"]),     # disconnected 负例
    ("f06_S3_folded", FRAMES["f06"], *ANCH["S3"]),
    ("f06_S6_folded", FRAMES["f06"], *ANCH["S6"]),
]
for name, fp, ax, ay in jobs:
    print(crop(fp, ax, ay, name))
