"""Hero GT 候选帧 → pair-ROI 裁剪拼接图（供视觉确认身份）。

用法: python scripts/p33_hero_gt_crops.py <candidates.json> <outdir> [max_n]
每 6 个裁剪拼一张竖排 contact sheet（顺序即候选顺序）。identity 仍需人工
（image_observed）逐帧确认——本脚本不做任何推断。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (
    HEROCARDS_PAIR_ROI)
from poker_decision_ai.vision2_platform.png_decode import decode_png
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png

EV = Path("data/vision2/coinpoker_evidence")


def main() -> int:
    cands = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    max_n = int(sys.argv[3]) if len(sys.argv) > 3 else len(cands)
    x0, y0, x1, y1 = HEROCARDS_PAIR_ROI
    per_sheet = 6
    done = 0
    for si in range(0, min(len(cands), max_n), per_sheet):
        rows = []
        for c in cands[si:si + per_sheet]:
            w, h, rgba = decode_png((EV / c["frame"]).read_bytes())
            arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
            rows.append(arr[y0:y1, x0:x1].copy())
        if not rows:
            break
        sheet = np.zeros((sum(r.shape[0] + 4 for r in rows), rows[0].shape[1],
                          4), dtype=np.uint8)
        sheet[..., :3] = 30
        sheet[..., 3] = 255
        yy = 0
        for r in rows:
            sheet[yy:yy + r.shape[0], :r.shape[1]] = r
            yy += r.shape[0] + 4
        f = outdir / f"sheet_{si:04d}.png"
        f.write_bytes(encode_rgba_png(sheet.shape[1], sheet.shape[0],
                                      sheet.tobytes()))
        print(f"{f.name}: frames[{si}..{si + len(rows) - 1}] = "
              + ", ".join(c['frame'] for c in cands[si:si + per_sheet]))
        done += len(rows)
    print(f"cropped {done} frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
