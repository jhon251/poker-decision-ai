"""P24-B — S5 亮底 variant 识别测试（白段分割 + v2bet1280 模板）。"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_bet import (  # noqa: E402
    gray_luma, load_bet_templates)
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    _pad_tight, fresh_ok, parse_stack_text)
from poker_decision_ai.vision2.realtime.image import ImageData, to_gray  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
tpls = load_bet_templates()


def _read(rec, arr, roi):
    x0, y0, ww, hh = roi
    sub = arr[y0:y0 + hh, x0:x0 + ww]
    l = sub[..., 0].astype(np.int16) * 0.299 + sub[..., 1].astype(np.int16) \
        * 0.587 + sub[..., 2].astype(np.int16) * 0.114
    bright = l > 150
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
    chars = []
    margins = []
    for (s, e) in runs:
        seg = bright[:, s:e]
        rys = np.where(seg.sum(axis=1) > 0)[0]
        if len(rys) < 6:
            continue
        sub2 = seg[rys.min():rys.max() + 1, s:e]
        gh, gw = sub2.shape
        if gh < 4 or gw > 24 and gh < 10 or gw > 30:
            pass
        if gh < 10 and gw <= 8:
            chars.append(",")
            margins.append(0.08)
        elif gh >= 4 and gw <= 30:
            img = _pad_tight(ImageData.from_rgba_bytes(
                gw, gh, np.stack([sub2 * 255] * 4, axis=-1)
                .astype(np.uint8).tobytes()), pad=2)
            g = np.asarray(to_gray(img).data, dtype=float) / 255.0
            ch, margin = rec._match_char(g)
            if ch is None or margin is None:
                chars.append("?")
            else:
                chars.append(ch)
                margins.append(margin)
    raw = "".join(chars)
    ok = fresh_ok(raw, margins) if margins else False
    return raw, ok


def main() -> int:
    rec = None
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (
        StackRecognizer)
    rec = StackRecognizer(window="1848")
    cases = [
        ("POT_CHANGED_12_1787995290794_full.png", (1340, 886, 145, 50),
         "89600"),
        ("BUTTON_CHANGED_598_1787993753587_full.png", (1340, 886, 145, 50),
         "34500"),
    ]
    for fid, roi, gt in cases:
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        raw, ok = _read(rec, arr, roi)
        print(f"{fid[:28]} GT={gt} 读={raw!r} fresh_ok={ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
