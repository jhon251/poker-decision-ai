"""P30-2B — stake 掩码全集提取（x95-195 亮掩码 → 模板 = 完整 stake 位图）。
检测 = 该 ROI 与已知 stake 位图的 NCC 相似度 + 结构性解析（分 token）。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/coinpoker_assets_v1876betting_header")
OUT.mkdir(parents=True, exist_ok=True)

FP = "BUTTON_CHANGED_167_1787992682749_full.png"
w, h, rgba = decode_png((EV / FP).read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
l = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
mask = (l > 120)[0:30, 90:200].astype(np.uint8)
np.save(OUT / "stake_mask_f01.npy", mask)
png = np.zeros((*mask.shape, 4), dtype=np.uint8)
png[..., :3] = 255
png[..., 3] = (mask * 255).astype(np.uint8)
(OUT / "stake_mask_f01.png").write_bytes(encode_rgba_png(
    mask.shape[1], mask.shape[0], png.tobytes()))
print(f"stake 掩码: {mask.shape}, ink={int(mask.sum())}")
# 行剖面
rowd = mask.sum(axis=1)
print("行剖面:", [int(v) for v in rowd])
