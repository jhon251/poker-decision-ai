"""P16-A — Stack 1876 数字精确尺度 + scale-adapt 实测（只读）。

精确测量底部带金文本的 tight bbox / 字高 / column projection，并用既有
1419 模板做 NCC 匹配实测，判定 scale-adapt 可行性。
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    SEAT_ANCHORS_1848, StackRecognizer)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")


def _gold(frame):
    r = frame[..., 0].astype(np.int16)
    g = frame[..., 1].astype(np.int16)
    b = frame[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
    return ((r > 120) & (g > 85) & (r >= g) & (g >= b - 30) & (l > 90))


def _tight_glyphs(mask, min_h=14):
    """mask → 每个连通字符段的 tight (x0,y0,w,h)。列投影 + 行投影。"""
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
        rys, rxs = np.where(seg)
        if len(rys) < 20:
            continue
        h = rys.max() - rys.min() + 1
        w = ce - cs
        if h < min_h:
            continue
        out.append((cs, rys.min(), w, h))
    return out


def main() -> int:
    rec = StackRecognizer(window="1848")
    for name in ["POT_CHANGED_102_1787992505794_full.png",
                 "POT_CHANGED_12_1787995290794_full.png",
                 "BUTTON_CHANGED_167_1787992682749_full.png"]:
        p = EV / name
        w, h, rgba = decode_png(p.read_bytes())
        if (w, h) != (1876, 1400):
            continue
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        print(f"\n=== {name} ===")
        # 底带 y1150-1400 金掩码 tight 字符段
        m = _gold(arr)[1150:1400, :]
        glyphs = _tight_glyphs(m)
        print(f"底带金文本字符段: {len(glyphs)} 个")
        for (x, y, gw, gh) in glyphs:
            print(f"   glyph x={x} y={1150+y} w={gw} h={gh}")
        # 对每个字符段跑既有模板 NCC（scale-adapt 实测）
        if glyphs:
            print("  NCC 匹配实测（前 8 段）:")
            from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (
                _pad_tight, _bright_tight)
            from poker_decision_ai.vision2.realtime.image import ImageData, to_gray
            for (x, y, gw, gh) in glyphs[:8]:
                sub = m[y:y + gh, x:x + gw]
                if sub.sum() < 60:
                    continue
                # 转 4 通道灰度图，复用 _match_char
                img = _pad_tight(ImageData.from_rgba_bytes(
                    gw, gh, np.stack([sub * 255] * 4, axis=-1)
                    .astype(np.uint8).tobytes()), pad=2)
                g = np.asarray(to_gray(img).data, dtype=float) / 255.0
                top_h = max(1, int(gh * 0.4))
                top_ink = float(sub[:top_h].sum()) / max(1.0, float(sub.sum()))
                ch, margin = rec._match_char(g, top_ratio=top_ink)
                print(f"   glyph(x={x},h={gh}) -> char={ch} margin={margin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
