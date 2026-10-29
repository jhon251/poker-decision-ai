"""P16-B — Pot 1876 READ-ONLY 根因分析（Step B1）。

逐项测量 Pot 真实几何 / 像素 / 中文字段 / 模板兼容，不修改生产代码。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.cashgame import PotRecognizer  # noqa: E402
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    StackRecognizer)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _roi_stats(arr, x0, y0, x1, y1):
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    luma = r * 0.299 + g * 0.587 + b * 0.114
    gold = (r > 120) & (g > 85) & (r >= g) & (g >= b - 30) & (luma > 90)
    white = luma > 200
    bright = luma > 150
    return {
        "gold": int(gold.sum()), "white": int(white.sum()),
        "bright": int(bright.sum()),
        "r_mean": float(r.mean()), "g_mean": float(g.mean()),
        "b_mean": float(b.mean()), "luma_mean": float(luma.mean()),
    }


def _tight_glyphs(mask, min_h=14):
    ys, xs = np.where(mask)
    if len(ys) < 20:
        return []
    cols = mask.sum(axis=0)
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
    out = []
    for (cs, ce) in runs:
        seg = mask[:, cs:ce]
        rys = np.where(seg.sum(axis=1) > 0)[0]
        if len(rys) < 14:
            continue
        out.append((cs, rys.min(), ce - cs, rys.max() - rys.min() + 1))
    return out


def main() -> int:
    names = [
        "BUTTON_CHANGED_167_1787992682749_full.png",
        "POT_CHANGED_12_1787995290794_full.png",
        "POT_CHANGED_102_1787992505794_full.png",
        "BUTTON_CHANGED_598_1787993753587_full.png",
        "POT_CHANGED_13_1787992279195_full.png",
    ]
    for name in names:
        p = EV / name
        if not p.exists():
            print(f"缺帧: {name}")
            continue
        w, h, rgba = decode_png(p.read_bytes())
        if (w, h) != (1876, 1400):
            print(f"非1876: {name}")
            continue
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        print(f"\n===== {name} =====")
        # ① 顶中区 (700-1200, 50-280) 全像素统计
        st = _roi_stats(arr, 700, 50, 1200, 280)
        print(f"[顶中区 700-1200,50-280] gold={st['gold']} white={st['white']} "
              f"bright={st['bright']} R={st['r_mean']:.0f} G={st['g_mean']:.0f} "
              f"B={st['b_mean']:.0f} luma={st['luma_mean']:.0f}")
        # ② 金掩码 tight glyphs（顶中区）
        r = arr[50:280, 700:1200, 0].astype(np.int16)
        g = arr[50:280, 700:1200, 1].astype(np.int16)
        b = arr[50:280, 700:1200, 2].astype(np.int16)
        luma = r * 0.299 + g * 0.587 + b * 0.114
        gold = ((r > 120) & (g > 85) & (r >= g) & (g >= b - 30) & (luma > 90))
        glyphs = _tight_glyphs(gold)
        print(f"[金掩码 glyphs] {len(glyphs)} 个（min_h=14）")
        for (x, y, gw, gh) in glyphs[:20]:
            # 颜色
            px = arr[50 + y:50 + y + gh, 700 + x:700 + x + gw].reshape(-1, 4)
            ink = px[(px[..., 0] > 110)]
            if len(ink):
                print(f"   glyph abs=({700+x},{50+y},{gw},{gh}) "
                      f"R={ink[...,0].mean():.0f} G={ink[...,1].mean():.0f} "
                      f"B={ink[...,2].mean():.0f} "
                      f"luma={(ink[...,0]*0.299+ink[...,1]*0.587+ink[...,2]*0.114).mean():.0f}")
            else:
                print(f"   glyph abs=({700+x},{50+y},{gw},{gh}) (无ink)")
        # ③ 白掩码 tight glyphs（对比，看数字是否白）
        white = luma > 150
        wglyphs = _tight_glyphs(white)
        print(f"[白掩码(luma>150) glyphs] {len(wglyphs)} 个")
        for (x, y, gw, gh) in wglyphs[:15]:
            print(f"   glyph abs=({700+x},{50+y},{gw},{gh})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
