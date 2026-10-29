"""P19-A — Bet 1876 精确根因：细粒度滑动搜索 GT 座，报告胶囊 bbox + 放宽带读取。"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from poker_decision_ai.vision2.pixel.realclient.coinpoker_bet import (  # noqa: E402
    gray_luma, detect_bet_capsule, segment_bet_glyphs, classify_bet_glyphs,
    load_bet_templates)
from poker_decision_ai.vision2.pixel.realclient.coinpoker_stack import (  # noqa: E402
    SEAT_ANCHORS_1848)
from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402

EV = Path("data/vision2/coinpoker_evidence")
GT = json.loads(Path("data/vision2/coinpoker_1876_gt.json").read_text(
    encoding="utf-8"))
tpls = load_bet_templates()


def main() -> int:
    for ann in GT["annotations"]:
        bet = (ann.get("fields") or {}).get("bet") or {}
        if not bet:
            continue
        fid = ann["frame_id"]
        w, h, rgba = decode_png((EV / fid).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        lum = gray_luma(arr)
        print(f"\n{fid}  GT bet={bet}")
        for seat, gt_val in bet.items():
            ax, ay = SEAT_ANCHORS_1848[seat]
            print(f"  [{seat}] GT={gt_val} anchor=({ax},{ay})")
            # 细搜索：锚点 ±400px，窗口 220×70，步 40
            best = None
            for y0 in range(max(0, ay - 400), min(h - 70, ay + 400), 40):
                for x0 in range(max(0, ax - 400), min(w - 220, ax + 400), 40):
                    cap = detect_bet_capsule(lum, (x0, y0, 220, 70))
                    if cap is None:
                        continue
                    glyphs = segment_bet_glyphs(cap["inner"])
                    if glyphs is None:
                        continue
                    labels = classify_bet_glyphs(glyphs, tpls)
                    if labels is None:
                        continue
                    raw = "".join(labels[0])
                    # 找最接近 GT 的胶囊
                    bx, by, bw, bh = cap["bbox"]
                    d = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
                    key = (d, raw, bx, by, bw, bh, labels[1])
                    if best is None or key[0] < best[0]:
                        best = key
            if best:
                print(f"    最近胶囊: raw={best[1]} bbox=({best[2]},{best[3]},"
                      f"{best[4]},{best[5]}) margins={best[6]} dist={best[0]:.0f}")
            else:
                print(f"    无胶囊读出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
