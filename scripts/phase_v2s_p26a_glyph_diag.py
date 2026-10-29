"""P26-A — 单字形经由冻结 PotRecognizer._match_char 的分数诊断。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.cashgame import (  # noqa: E402
    PotRecognizer)

EV = Path("data/vision2/coinpoker_evidence")
POT = PotRecognizer(window="1876")


def load(fp):
    w, h, rgba = decode_png((EV / fp).read_bytes())
    return np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)


arr = load("BUTTON_CHANGED_167_1787992682749_full.png")
# f01 S1 '1' 第一字符：实测 comp (188,428)-(199,470) 附近（宽11高34）
# 用 3px 余量 crop
for (x0, y0, x1, y1, tag) in [
        (185, 425, 202, 473, "S1 f01 glyph1 '1'"),
        (231, 425, 253, 473, "S1 f01 glyph2 '0'"),
        (203, 425, 231, 473, "S1 f01 glyph3? mid")]:
    crop = arr[y0:y1, x0:x1]
    ch, sc, margin = POT._match_char(crop)
    print(f"[{tag}] -> ch={ch!r} score={sc:.3f} margin={margin}")

# 打印该字符的 luma 分布
crop = arr[425:473, 185:202]
lc = (crop[..., 0].astype(np.int16) * 0.299
      + crop[..., 1].astype(np.int16) * 0.587
      + crop[..., 2].astype(np.int16) * 0.114)
print("glyph1 luma: p10=%.0f p50=%.0f p90=%.0f max=%.0f" %
      (np.percentile(lc, 10), np.percentile(lc, 50), np.percentile(lc, 90),
       lc.max()))
