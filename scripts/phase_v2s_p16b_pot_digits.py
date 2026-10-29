"""P16-B — Pot 1876 数字区精细测量（READ-ONLY）。

聚焦 pot 数字区（x 840-1040, y 80-180）做 column projection + 逐字符分割
+ 颜色 + 字高 + 既有模板 NCC，判定根因。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.cashgame import PotRecognizer  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    StackRecognizer, _pad_tight)
from poker_decision_ai.vision2.realtime.image import ImageData, to_gray  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def main() -> int:
    pot = PotRecognizer()
    stack = StackRecognizer(window="1848")
    names = [
        "BUTTON_CHANGED_167_1787992682749_full.png",
        "POT_CHANGED_12_1787995290794_full.png",
        "POT_CHANGED_102_1787992505794_full.png",
        "POT_CHANGED_13_1787992279195_full.png",
        "BUTTON_CHANGED_598_1787993753587_full.png",
    ]
    for name in names:
        p = EV / name
        if not p.exists():
            continue
        w, h, rgba = decode_png(p.read_bytes())
        if (w, h) != (1876, 1400):
            continue
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        print(f"\n===== {name} =====")
        # pot 数字区
        roi = arr[80:180, 840:1040]
        r = roi[..., 0].astype(np.int16)
        g = roi[..., 1].astype(np.int16)
        b = roi[..., 2].astype(np.int16)
        luma = r * 0.299 + g * 0.587 + b * 0.114
        # 三个掩码：白 / 亮 / 金
        white = luma > 200
        bright = luma > 150
        gold = (r > 120) & (g > 85) & (r >= g) & (g >= b - 30) & (luma > 90)
        print(f"  白(>200)={int(white.sum())} 亮(>150)={int(bright.sum())} "
              f"金={int(gold.sum())}")
        # 逐字符（column projection，luma>150）
        cols = (luma > 150).sum(axis=0)
        active = cols >= 2
        runs, run = [], None
        for i, a in enumerate(active):
            if a and run is None:
                run = i
            elif not a and run is not None:
                runs.append((run, i))
                run = None
        if run is not None:
            runs.append((run, len(active)))
        print(f"  亮字符段(luma>150): {len(runs)} 个")
        for (cs, ce) in runs:
            seg = roi[:, cs:ce]
            # 每段内行投影 tight
            srows = (seg[..., 0].astype(np.int16) * 0.299
                     + seg[..., 1].astype(np.int16) * 0.587
                     + seg[..., 2].astype(np.int16) * 0.114) > 150
            ys = np.where(srows.sum(axis=1) > 0)[0]
            if len(ys) < 10:
                continue
            hh = ys.max() - ys.min() + 1
            ww = ce - cs
            flat = seg.reshape(-1, 4)
            mask = flat[:, 0].astype(np.int16) > 110
            ink = flat[mask]
            col = (f"R={ink[:,0].mean():.0f} G={ink[:,1].mean():.0f} "
                   f"B={ink[:,2].mean():.0f}") if len(ink) else "无ink"
            print(f"    段 abs=({840+cs},{80+ys.min()},{ww},{hh}) {col}")
            # 既有模板 NCC（Stack 模板）
            if 5 <= hh <= 60:
                sub = (bright[ys.min():ys.max() + 1, cs:ce]).astype(np.uint8)
                img = _pad_tight(ImageData.from_rgba_bytes(
                    ww, hh, np.stack([sub * 255] * 4, axis=-1)
                    .astype(np.uint8).tobytes()), pad=2)
                gg = np.asarray(to_gray(img).data, dtype=float) / 255.0
                ch, margin = stack._match_char(gg)
                print(f"        -> Stack模板: {ch} margin={margin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
