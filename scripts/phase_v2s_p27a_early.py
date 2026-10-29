"""P27-A — 会话最早帧（开局）vs 6-max 全座检查 + 全帧 panel 分布。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/p27a_crops")

# 按时间戳排序的会话早期帧
early = sorted(EV.glob("*_full.png"))[:400]
names = [p.name for p in early if "1787992" in p.name]
print("session 早期帧:", len(names))


def zoom(fp_name, x0, y0, x1, y1, name, k=1):
    w, h, rgba = decode_png((EV / fp_name).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    sub = arr[y0:y1, x0:x1]
    big = np.repeat(np.repeat(sub, k, axis=0), k, axis=1)
    (OUT / f"{name}.png").write_bytes(
        encode_rgba_png(big.shape[1], big.shape[0], big.tobytes()))
    print(name, big.shape[1], "x", big.shape[0])


# 取最早的 1876 帧
for n in names[:12]:
    w, h, rgba = decode_png((EV / n).read_bytes())
    print(f"  {n[:36]} geom=({w}x{h})")

# 最早 1876 帧全桌
for n in names:
    w, h, rgba = decode_png((EV / n).read_bytes())
    if (w, h) == (1876, 1400):
        print("earliest 1876:", n)
        zoom(n, 0, 0, 1876, 1400, "z_early_1876", 1)
        break
