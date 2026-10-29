"""P27-A — 全 1876 帧扫描：每座"座牌盒"占用对象签名（name 行灰白字）
+ 显示无签名帧/座（潜在空座实例）。"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")

# 每座 name 行 ROI（P26-B stack ROI 上方：name 在 stack 上 ~55px）
# 由 f01 座牌盒实测：S1 name ≈ (183,360,337,410) 之类——用 stack ROI 推导
STACK = {
    "S1": (165, 413, 360, 485), "S2": (845, 298, 1035, 370),
    "S3": (1520, 413, 1715, 485), "S4": (125, 978, 320, 1050),
    "S5": (1560, 978, 1745, 1050), "S6": (838, 1236, 1040, 1308),
}


def name_roi(seat):
    x0, y0, x1, y1 = STACK[seat]
    return (x0, max(0, y0 - 55), x1, max(0, y0 - 5))


def name_ink(arr, seat):
    x0, y0, x1, y1 = name_roi(seat)
    if x1 > arr.shape[1] or y1 > arr.shape[0]:
        return -1
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    # 灰白字：亮 + 近中性
    return int(((l > 110) & (np.abs(r - g) < 28) & (np.abs(g - b) < 28)
                ).sum())


# 采样全部 1876 帧（非 GT 的也要看——找空座实例）
p_1876 = []
for p in sorted(EV.glob("*_full.png")):
    # 只取几何已知的：先解码 IHDR 快速
    pass

# 直接用已知 1876 帧清单（P26 阶段列出的 1074 个；这里重扫耗时——
# 改用：先扫最近 200 帧快速，再全扫）
files = sorted(EV.glob("*.png"))
# 全部其实是 *full.png；为省时按序扫 400 帧
hits = Counter()
examples = []
scanned = 0
for fp in files:
    if not fp.name.endswith("_full.png"):
        continue
    try:
        w, h, rgba = decode_png(fp.read_bytes())
    except Exception:
        continue
    if (w, h) != (1876, 1400):
        continue
    scanned += 1
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    row = {}
    for seat in STACK:
        row[seat] = name_ink(arr, seat)
    # 空座签名：name ink < 阈值（30）→ 疑似空或不完整
    empty_like = [s for s, v in row.items() if 0 <= v < 30]
    if empty_like:
        hits.update(empty_like)
        if len(examples) < 8:
            examples.append((fp.name, dict(row)))
    if scanned >= 1074:
        break
print(f"scanned={scanned}")
print(f"empty-like 座分布(全帧): {dict(hits)}")
for name, row in examples:
    print(f"  {name[:30]} {row}")
