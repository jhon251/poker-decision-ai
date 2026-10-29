"""P19-C — Pot 1876 PRE-AUDIT：6 GT + 顶中 ROI 读取实测。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.cashgame import PotRecognizer  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))
#: 顶中 pot ROI（P16-B 实测 880-1040, 85-158；放大一点）
POT_ROI = (820, 75, 1080, 175)


def main() -> int:
    pot = PotRecognizer()
    ok = 0
    total = 0
    for ann in GT["annotations"]:
        pv = (ann.get("fields") or {}).get("pot")
        if pv is None:
            continue
        total += 1
        fid = ann["frame_id"]
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        x0, y0, x1, y1 = POT_ROI
        roi = arr[y0:y1, x0:x1]
        # 金/白双态分割尝试：亮(luma>150) + 金(g>=b+25)
        r = roi[..., 0].astype(np.int16)
        g = roi[..., 1].astype(np.int16)
        b = roi[..., 2].astype(np.int16)
        luma = r * 0.299 + g * 0.587 + b * 0.114
        bright = luma > 150
        gold = (r > 110) & (g > 80) & (r >= g) & (g >= b + 25)
        use = bright | gold
        # 用 pot._seg 逻辑（luma>150）在 bright 上跑
        runs = pot._seg(np.stack([bright * 255] * 4, axis=-1).astype(np.uint8))
        chars = []
        for (s, e) in runs:
            crop = roi[:, s:e]
            ch, sc, margin = pot._match_char(crop)
            if ch is None or sc is None or sc < pot.min_score \
                    or margin < pot.min_margin:
                chars.append("?")
            else:
                chars.append(ch)
        text = "".join(chars)
        print(f"{fid} GT={pv} 读={' '.join(chars) or '(无)'}")
        if text == str(pv):
            ok += 1
    print(f"\n总: {ok}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
