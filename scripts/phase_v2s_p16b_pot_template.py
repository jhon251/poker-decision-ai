"""P16-B — Pot 1876 模板兼容实测（READ-ONLY）。

直接对 1876 pot 数字区跑 PotRecognizer._seg + _match_char，判定既有模板
（28-34px）能否匹配 1876 pot 数字（~73px）。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.cashgame import PotRecognizer  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def main() -> int:
    pot = PotRecognizer()
    # 打印模板尺度带
    heights = []
    for ch, insts in pot._prep.items():
        for (t, t0, tn) in insts:
            heights.append((ch, t.shape[0], t.shape[1]))
    from collections import Counter
    print("模板尺度统计:", Counter((h, w) for _, h, w in heights))

    names = [
        "BUTTON_CHANGED_167_1787992682749_full.png",
        "POT_CHANGED_102_1787992505794_full.png",
        "POT_CHANGED_13_1787992279195_full.png",
    ]
    for name in names:
        p = EV / name
        w, h, rgba = decode_png(p.read_bytes())
        if (w, h) != (1876, 1400):
            continue
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        print(f"\n===== {name} =====")
        # pot 数字区（多个 ROI 尝试）
        for (x0, y0, x1, y1, label) in [
            (840, 80, 1040, 180, "数字区840-1040,80-180"),
            (840, 95, 1040, 165, "数字区840-1040,95-165"),
        ]:
            roi = arr[y0:y1, x0:x1]
            runs = pot._seg(roi)
            print(f"[{label}] _seg 字符段: {len(runs)}")
            if not runs:
                continue
            chars = []
            for (s, e) in runs:
                crop = roi[:, s:e]
                ch, sc, margin = pot._match_char(crop)
                hh = e - s
                chars.append(f"{ch}(m={margin:.2f} w={hh})")
                if ch is not None and sc is not None:
                    print(f"    段(x={x0+s},w={hh}) -> {ch} score={sc:.2f} margin={margin:.2f}")
            print("    组装:", " ".join(chars))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
