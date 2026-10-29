"""P24-C — bright-path 验证：check_dark=False 读 S5 89,600 / 34,500。"""
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
ROI = (1350, 886, 130, 48)

for fid, gt in [("POT_CHANGED_12_1787995290794_full.png", "89600"),
                ("BUTTON_CHANGED_598_1787993753587_full.png", "34500")]:
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    lum = gray_luma(arr)
    cap = detect_bet_capsule(lum, ROI, check_dark=False)
    if cap is None:
        print(f"{fid[:28]} GT={gt}: check_dark=False 仍 None")
        continue
    bx, by, bw, bh = cap["bbox"]
    glyphs = segment_bet_glyphs(cap["inner"])
    if glyphs is None:
        print(f"{fid[:28]} GT={gt}: capsule({bx},{by},{bw},{bh}) seg失败")
        continue
    labels = classify_bet_glyphs(glyphs, tpls)
    text = "".join(labels[0]) if labels else None
    print(f"{fid[:28]} GT={gt}: capsule({bx},{by},{bw},{bh}) "
          f"tokens={len(glyphs)} 读={text!r} {'HIT' if text == gt else 'MISS'}")
