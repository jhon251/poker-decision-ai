"""P22-A — 诊断：pot 数字精确高度 + 模板匹配分数。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.cashgame import PotRecognizer  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def main() -> int:
    pot = PotRecognizer()
    name = "BUTTON_CHANGED_167_1787992682749_full.png"
    w, h, rgba = decode_png((EV / name).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[488:543, 908:1060]
    runs = pot._seg(roi)
    print(f"字符段: {len(runs)}")
    for (s, e) in runs:
        crop = roi[:, s:e]
        # tight 高度（luma>120）
        l = (crop[..., 0].astype(np.int16) * 0.299 + crop[..., 1].astype(np.int16)
             * 0.587 + crop[..., 2].astype(np.int16) * 0.114)
        rows = np.where((l > 120).sum(axis=1) > 0)[0]
        hh = rows.max() - rows.min() + 1 if len(rows) else 0
        ch, sc, margin = pot._match_char(crop)
        print(f"  段 x={908+s} w={e-s} h={hh} -> {ch} score={sc} margin={margin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
