"""P22-A — 从 6 个真实 GT 帧提取 1876 pot 数字模板（GT provenance）。

每个 glyph：帧 + GT 值 + 位置 → label（GT 值序）。逗号=几何 token。
输出 data/vision2/coinpoker_assets_v1876pot/（manifest + png）。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.cashgame import PotRecognizer  # noqa: E402
from poker_decision_ai.vision2_platform import encode_rgba_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))
POT_ROI = (908, 488, 1060, 543)
OUT = Path("data/vision2/coinpoker_assets_v1876pot")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pot = PotRecognizer()
    assets = []
    idx = 0
    for ann in GT["annotations"]:
        pv = (ann.get("fields") or {}).get("pot")
        if pv is None:
            continue
        fid = ann["frame_id"]
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        x0, y0, x1, y1 = POT_ROI
        roi = arr[y0:y1, x0:x1]
        runs = pot._seg(roi)
        # GT 值字符串（去逗号后与段配对）
        gt_str = str(pv)
        gt_digits = [c for c in gt_str]  # 纯数字
        di = 0
        for (s, e) in runs:
            crop = roi[:, s:e]
            l = (crop[..., 0].astype(np.int16) * 0.299
                 + crop[..., 1].astype(np.int16) * 0.587
                 + crop[..., 2].astype(np.int16) * 0.114)
            rows = np.where((l > 120).sum(axis=1) > 0)[0]
            if len(rows) < 6:
                continue
            hh = rows.max() - rows.min() + 1
            ww = e - s
            band_h = 30.0
            if hh <= 18 and ww <= 10:
                label = ","
            elif di < len(gt_digits):
                label = gt_digits[di]
                di += 1
            else:
                continue
            tight = crop[rows.min():rows.max() + 1,
                         s:e - s + 1] if False else crop[rows.min():rows.max() + 1]
            # 保存 tight glyph
            gly = crop[rows.min():rows.max() + 1]
            gh, gw = gly.shape[0], gly.shape[1]
            fname = f"pot_{label}_{idx}.png"
            # 用确定二值化：luma>120 白字
            gl = (l[rows.min():rows.max() + 1] > 120).astype(np.uint8) * 255
            rgba4 = np.stack([gl] * 4, axis=-1).astype(np.uint8)
            (OUT / fname).write_bytes(encode_rgba_png(gw, gh, rgba4.tobytes()))
            assets.append({
                "label": label, "kind": "pot_digit_1876",
                "crop_ref": f"data\\vision2\\coinpoker_assets_v1876pot\\{fname}",
                "bbox": [int(908 + s), int(488 + rows.min()), int(gw), int(gh)],
                "source_frame_id": fid,
                "gt_pot_value": int(pv),
                "ground_truth_status": "gt_derived_label",
                "provenance": {
                    "source": "P21/P22-A visual GT",
                    "method": "GT value position labeling",
                    "gt_value": int(pv),
                },
            })
            idx += 1
    mp = {"client": "coinpoker", "version": "v1876pot",
          "coordinate_system": "full_png_1876x1400",
          "ground_truth_status": "gt_derived_label",
          "assets": assets}
    (OUT / "manifest.json").write_text(
        json.dumps(mp, indent=2, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print("提取 glyph 数:", len(assets))
    print("按 label:", dict(Counter(a["label"] for a in assets)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
