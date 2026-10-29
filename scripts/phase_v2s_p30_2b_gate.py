"""P30-2B — 证据门：不同 stake 组合扫描 + header 文本内容分类。
方法：header 区白色文本 → 列投影 token 带 → 每 token 的几何特征
（帮助区分 1K/2K(320) vs 其它格式如 2.5K/5K(500)）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
files = sorted(EV.glob("*.png"))


def header_tokens(fid):
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[0:40, 0:520]
    r = roi[..., 0].astype(np.int16); g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    white = (l > 150) & (np.abs(r - g) < 40) & (np.abs(g - b) < 40)
    cold = white.sum(axis=0)
    runs, run = [], None
    for i, v in enumerate(cold):
        if v >= 2 and run is None:
            run = [i, i]
        elif v >= 2 and run is not None:
            run[1] = i
        elif run is not None:
            runs.append(run); run = None
    if run:
        runs.append(run)
    merged = []
    for rr in runs:
        if merged and rr[0] - merged[-1][1] <= 4:
            merged[-1][1] = rr[1]
        else:
            merged.append(rr)
    # 过滤宽字形（hand_id 是连续数字串 → 大宽 token）
    return [(a, b, b - a + 1) for a, b in merged]


# 抽样 30 帧：token 带分布 → 推断文本长度差异
import random
random.seed(42)
sample = random.sample([f for f in files if f.name.endswith("_full.png")], 60)
profiles = {}
for fp in sample:
    try:
        w, h, rgba = decode_png(fp.read_bytes())
    except Exception:
        continue
    if (w, h) != (1876, 1400):
        continue
    toks = header_tokens(fp.name)
    key = tuple(t[2] for t in toks)
    profiles.setdefault(key, []).append(fp.name)
print(f"样本 token 宽度剖面（60 帧）: {len(profiles)} 种")
for k, v in sorted(profiles.items(), key=lambda x: -len(x[1])):
    print(f"  {k} -> {len(v)} 帧 例={v[0][:40]}")
