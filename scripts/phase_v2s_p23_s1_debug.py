"""P23 — 调试 S1 stack 识别失败点（分割/匹配/fresh_ok）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    StackRecognizer, fresh_ok)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
rec = StackRecognizer(window="1848")
w, h, rgba = decode_png((EV /
    "BUTTON_CHANGED_167_1787992682749_full.png").read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
x0, y0, x1, y1 = 150, 325, 360, 390
roi = arr[y0:y1, x0:x1]
gold = rec._gold_mask(roi)
ys, xs = np.where(gold > 0)
print(f"gold 像素: {len(ys)}")
if len(ys):
    inner = gold[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    print(f"tight: {inner.shape}")
    comps = rec._split(inner)
    print(f"分割组件: {len(comps)}")
    for i, (cs, ce) in enumerate(comps):
        seg = inner[:, cs:ce]
        rws = np.where(seg.sum(axis=1) > 0)[0]
        if len(rws) < 6:
            continue
        sub = seg[rws[0]:rws[-1] + 1]
        img = rec._pad_tight(__import__("poker_decision_ai.vision2.realtime.image",
                                        fromlist=["ImageData"]).ImageData
                             .from_rgba_bytes(
            sub.shape[1], sub.shape[0],
            np.stack([sub * 255] * 4, axis=-1).astype(np.uint8).tobytes()),
            pad=2)
        from poker_decision_ai.vision2.realtime.image import to_gray
        g = np.asarray(to_gray(img).data, dtype=float) / 255.0
        ch, margin = rec._match_char(g)
        print(f"  comp{i} w={sub.shape[1]} h={sub.shape[0]} -> {ch} "
              f"margin={margin}")
