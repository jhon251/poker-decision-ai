"""P22-A — 提取帧09 的 '.' 与 'M' 模板（K/M 格式：底池 1.61M，GT=1610000）。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2_platform import encode_rgba_png  # noqa: E402
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
OUT = Path("data/vision2/coinpoker_assets_v1876pot")
fid = "POT_CHANGED_139_1787992592538_full.png"


def main() -> int:
    w, h, rgba = decode_png((EV / fid).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    roi = arr[488:543, 908:1060]
    l = (roi[..., 0].astype(np.int16) * 0.299
         + roi[..., 1].astype(np.int16) * 0.587
         + roi[..., 2].astype(np.int16) * 0.114)
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
    # 帧09 显示 "1.61M"：段 = 1 . 6 1 M
    labels = ["1", ".", "6", "1", "M"]
    mp = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for (s, e), lab in zip(runs, labels):
        seg = bright[:, s:e]
        rows = np.where(seg.sum(axis=1) > 0)[0]
        if len(rows) < 4:
            continue
        gly = bright[rows.min():rows.max() + 1, s:e]
        gh, gw = gly.shape
        fname = f"pot_{lab}_{97}.png" if lab in ".M" else None
        if fname is None:
            continue        # 数字已有模板
        rgba4 = np.stack([(gly * 255).astype(np.uint8)] * 4, axis=-1)
        (OUT / fname).write_bytes(encode_rgba_png(gw, gh, rgba4.tobytes()))
        mp["assets"].append({
            "label": lab, "kind": "pot_digit_1876",
            "crop_ref": f"data\\vision2\\coinpoker_assets_v1876pot\\{fname}",
            "bbox": [int(908 + s), int(488 + rows.min()), int(gw), int(gh)],
            "source_frame_id": fid, "gt_pot_value": 1610000,
            "ground_truth_status": "gt_derived_label",
            "provenance": {"source": "P21/P22-A visual GT", "method": "GT K/M display",
                           "gt_value": 1610000, "display": "1.61M"},
        })
        print(f"提取 {lab}: {fname} {gw}x{gh}")
    (OUT / "manifest.json").write_text(
        json.dumps(mp, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
