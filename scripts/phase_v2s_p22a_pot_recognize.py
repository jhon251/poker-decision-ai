"""P22-A — Pot 1876 正确 ROI + 既有模板识别实测（6 GT 对照）。"""
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
#: 正确 ROI（P21+实测）：数字区（排除"底池"标签 x<908）
POT_ROI = (908, 488, 1060, 543)


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
        runs = pot._seg(roi)
        if not runs:
            print(f"{fid} GT={pv} 无字符段")
            continue
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
        import re
        fmt_ok = re.match(r"^\d{1,3}(,\d{3})*$", text) is not None
        hit = (text == str(pv).replace(",", "")) or (
            text.replace(",", "") == str(pv))
        # 也接受带逗号匹配
        hit2 = text.replace(",", "") == str(pv)
        if fmt_ok and hit2:
            ok += 1
        print(f"{fid} GT={pv} 读={text!r} 格式={'OK' if fmt_ok else 'FAIL'} "
              f"{'OK' if (fmt_ok and hit2) else 'FAIL'}")
    print(f"\n总: {ok}/{total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
