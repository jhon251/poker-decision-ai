"""V3E-P2 — 覆盖矩阵 + UNKNOWN 逐卡归因（只读复用冻结识别器内部函数）。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poker_decision_ai.vision2.pixel.realclient import coinpoker_herocards1876 as HC
from poker_decision_ai.vision2_platform.png_decode import decode_png

EV = Path("data/vision2/coinpoker_evidence")
SCAN = Path("logs/v3e_p2_coverage_scan.json")
OUT = Path("logs/v3e_p2_gap_analysis.json")

MIN_SCORE = HC.MIN_SCORE
MIN_MARGIN = HC.MIN_MARGIN


def main() -> int:
    scan = json.loads(SCAN.read_text(encoding="utf-8"))
    details = scan["details"]
    pair_frames = [d for d in details
                   if d["cards"] and d["reason"] != "no_card_pair_present"]
    matrix = {
        "pair_present_frames": len(pair_frames),
        "frame_both_ok": sum(1 for d in pair_frames
                             if all(c["status"] == "OK" for c in d["cards"])),
        "frame_one_of_two": sum(1 for d in pair_frames
                                if any(c["status"] == "OK" for c in d["cards"])
                                and not all(c["status"] == "OK"
                                            for c in d["cards"])),
        "cards_total": sum(len(d["cards"]) for d in pair_frames),
        "cards_ok": scan["n_cards_ok"],
    }
    matrix["frame_none"] = (matrix["pair_present_frames"]
                            - matrix["frame_both_ok"]
                            - matrix["frame_one_of_two"])

    rec = HC.HeroCardsRecognizer1876()
    prep = rec._prep
    rows = []
    fail_stage = Counter()
    best_rank_counter = Counter()   # pos -> 获胜字符（未过门）
    best_suit_counter = Counter()
    for d in pair_frames:
        w, h, rgba = decode_png((EV / d["frame"]).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        for pos_i, key in enumerate(("card1", "card2"), start=1):
            c = d["cards"][pos_i - 1]
            if c["status"] == "OK":
                continue
            r_ink = HC._corner_mask(arr, HC.RANK_WINDOWS[key])
            s_ink = HC._corner_mask(arr, HC.SUIT_WINDOWS[key])
            row = {"frame": d["frame"], "pos": pos_i, "reason": c.get("reason")}
            if r_ink.sum() < 30 or s_ink.sum() < 20:
                row["stage"] = "corner_ink_insufficient"
                fail_stage[row["stage"]] += 1
                rows.append(row)
                continue
            rch, rsc, rmg = HC._match_glyph(r_ink, prep, None)
            sch, ssc, smg = HC._match_glyph(s_ink, prep, None)
            row["rank_best"] = [rch, round(rsc, 3), round(rmg, 3)]
            row["suit_best"] = [sch, round(ssc, 3), round(smg, 3)]
            if rsc < MIN_SCORE:
                row["stage"] = "rank_score_fail"
                best_rank_counter["%s@c%d" % (rch, pos_i)] += 1
            elif rmg < MIN_MARGIN and len(prep.get(rch, ())) > 1:
                row["stage"] = "rank_margin_fail"
                best_rank_counter["%s@c%d" % (rch, pos_i)] += 1
            elif ssc < MIN_SCORE:
                row["stage"] = "suit_score_fail"
                best_suit_counter["%s@c%d" % (sch, pos_i)] += 1
            elif smg < MIN_MARGIN and len(prep.get(sch, ())) > 1:
                row["stage"] = "suit_margin_fail"
                best_suit_counter["%s@c%d" % (sch, pos_i)] += 1
            else:
                # 颜色校验阶段
                rf = HC._red_frac(arr, HC.SUIT_WINDOWS[key])
                row["red_frac"] = round(rf, 3)
                row["stage"] = ("red_suit_color_mismatch"
                                if sch in ("h", "d") and rf < HC.RED_INK_FRAC_MIN
                                else "black_suit_color_mismatch"
                                if sch not in ("h", "d") and rf > 0.6
                                else "unexplained")
                fail_stage[row["stage"]] += 1
                rows.append(row)
                continue
            fail_stage[row["stage"]] += 1
            rows.append(row)
    OUT.write_text(json.dumps({"matrix": matrix, "fail_stage": dict(fail_stage),
                               "best_rank_unknown": dict(best_rank_counter),
                               "best_suit_unknown": dict(best_suit_counter),
                               "rows": rows},
                              ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps(matrix, indent=1))
    print("fail_stage:", dict(fail_stage))
    print("best_rank_unknown:", dict(best_rank_counter))
    print("best_suit_unknown:", dict(best_suit_counter))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
