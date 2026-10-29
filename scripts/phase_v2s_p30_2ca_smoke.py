"""P30-2C-A — to_call detector 冒烟验证（含 check/无面板）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_to_call import (  # noqa: E402
    ToCallRecognizer1876)

EV = Path("data/vision2/coinpoker_evidence")
REC = ToCallRecognizer1876()

for fid, expect in [
        ("POT_CHANGED_114_1787992536082_full.png", 124163),
        ("POT_CHANGED_110_1787992526057_full.png", 2000),
        ("POT_CHANGED_143_1787992603846_full.png", 2000),
        ("POT_CHANGED_177_1787992718375_full.png", 0),   # 过牌
        ("POT_CHANGED_229_1787992854988_full.png", 0),   # 过牌
        ("BUTTON_CHANGED_167_1787992682749_full.png", None),  # 无动作面板
]:
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    res = REC.recognize(arr)
    ok = "OK" if res.get("to_call") == expect else "XX"
    print(f"{fid[:30]}: to_call={res.get('to_call')} expect={expect} "
          f"{ok} ({res.get('reason')}) conf={res.get('confidence')}")
