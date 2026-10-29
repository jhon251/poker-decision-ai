"""P24-B — S5 89,600 直接裁剪识别测试（白字暗底，非胶囊）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_bet import (  # noqa: E402
    gray_luma, detect_bet_capsule, segment_bet_glyphs, classify_bet_glyphs,
    load_bet_templates)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
tpls = load_bet_templates()

w, h, rgba = decode_png((EV /
    "POT_CHANGED_12_1787995290794_full.png").read_bytes())
arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
lum = gray_luma(arr)
roi = (1350, 888, 130, 48)
cap = detect_bet_capsule(lum, roi)
print("capsule:", cap)
if cap:
    bx, by, bw, bh = cap["bbox"]
    print(f"bbox=({bx},{by},{bw},{bh}) dark={cap['dark_ratio']:.2f}")
    glyphs = segment_bet_glyphs(cap["inner"])
    print("glyphs:", None if glyphs is None else len(glyphs))
    if glyphs:
        labels = classify_bet_glyphs(glyphs, tpls)
        print("读:", "".join(labels[0]) if labels else None)
else:
    # 直接看该区白像素结构
    sub = lum[890:935, 1310:1450]
    bright = (sub > 150).astype(np.uint8)
    cols = bright.sum(axis=0)
    runs, run = [], None
    for i, v in enumerate(cols):
        if v >= 2 and run is None:
            run = i
        elif v < 2 and run is not None:
            runs.append((run, i))
            run = None
    if run is not None:
        runs.append((run, len(cols)))
    print(f"白字符段: {[(1310+s, 1310+e, e-s) for s,e in runs]}")
    print("区内白像素:", int(bright.sum()))
