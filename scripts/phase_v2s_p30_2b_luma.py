"""P30-2B — header 两段亮度对比（prefix "NLH 1218466 -" vs stake "1K/2K (320)"）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def luma_profile(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[0:40, 0:520]
    l = roi[..., 0] * 0.299 + roi[..., 1] * 0.587 + roi[..., 2] * 0.114
    # x 窗：prefix 15-65 / stake 66-200
    p1 = l[:, 15:65]; p2 = l[:, 66:200]
    print(f"[{fp[:8]}] prefix max={p1.max():.0f} p99={np.percentile(p1, 99):.0f}"
          f" | stake max={p2.max():.0f} p99={np.percentile(p2, 99):.0f}")
    # 水平剖面（逐 5px 最大亮度——看文本分布）
    prof = [int(np.percentile(l[:, i:i+5], 99)) for i in range(0, 520, 5)]
    print("  x剖面(p99):", prof[:40])


for fp in ["BUTTON_CHANGED_167_1787992682749_full.png",
           "8bf82f134651db0add626be0304a2ff1.png"]:
    luma_profile(fp)
