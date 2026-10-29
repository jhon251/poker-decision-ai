"""V3E-P4 覆盖闭环审计：覆盖率 + rank×suit×position 矩阵 + UNKNOWN 分类数据。

只读；基于 V3E-P3 修复后 17,243 帧扫描（logs/v3e_p2_coverage_scan.json）。
输出 logs/v3e_p4_closure.json。
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ASSETS = Path("data/vision2/coinpoker_assets_v1876herocards/manifest.json")
SCAN = Path("logs/v3e_p2_coverage_scan.json")
GAP = Path("logs/v3e_p2_gap_analysis.json")
OUT = Path("logs/v3e_p4_closure.json")

RANKS = "AKQJT98765432"
SUITS = "shdc"


def main() -> int:
    man = json.loads(ASSETS.read_text(encoding="utf-8"))
    scan = json.loads(SCAN.read_text(encoding="utf-8"))
    gap = json.loads(GAP.read_text(encoding="utf-8"))

    # ── 1) 覆盖率 ────────────────────────────────────────────────
    pair = scan["n_frames_pair_present"]
    cards_total = pair * 2
    cards_ok = scan["n_cards_ok"]
    frames_full_ok = scan["n_frames"] - scan["n_frames_unknown"]
    stats = {
        "frames_total": scan["n_frames"],
        "frames_pair_present": pair,
        "frames_full_pair_ok": frames_full_ok,
        "frame_level_pct": round(100 * frames_full_ok / pair, 1),
        "cards_total": cards_total,
        "cards_ok": cards_ok,
        "card_level_pct": round(100 * cards_ok / cards_total, 1),
    }

    # ── 2) rank × suit × position 识别输出矩阵（17k 实测）────────
    out_matrix = {"c1": defaultdict(Counter), "c2": defaultdict(Counter)}
    card_reason = {"c1": Counter(), "c2": Counter()}
    for r in scan["details"]:
        for idx, c in enumerate(r["cards"]):
            pos = "c1" if idx == 0 else "c2"
            if c.get("status") == "OK":
                out_matrix[pos][c["rank"]][c["suit"]] += 1
            else:
                card_reason[pos][c.get("reason") or "gate"] += 1
    matrix = {}
    for pos in ("c1", "c2"):
        rows = []
        for rk in RANKS:
            rows.append([rk] + [out_matrix[pos][rk].get(s, 0) for s in SUITS]
                        + [sum(out_matrix[pos][rk].values())])
        matrix[pos] = {"suits": list(SUITS), "rows": rows}

    # ── 3) 模板资产矩阵 ─────────────────────────────────────────
    tpl = {"c1": defaultdict(Counter), "c2": defaultdict(Counter)}
    for a in man["assets"]:
        pos = "c%d" % a.get("card", 0)
        kind = "rank" if a["label"].startswith("rank") else "suit"
        ch = a["rank"] if kind == "rank" else a["suit"]
        tpl[pos][kind][ch] += 1      # 含 dim+overlay 实例（均为有效资产）
    tpl_matrix = {
        pos: {"rank": "".join(sorted(tpl[pos]["rank"], key=RANKS.index)),
              "suit": "".join(sorted(tpl[pos]["suit"], key=SUITS.index)),
              "instances": sum(tpl[pos]["rank"].values())
              + sum(tpl[pos]["suit"].values())}
        for pos in ("c1", "c2")
    }

    # ── 4) 剩余 UNKNOWN 卡位归因（修复后扫描）───────────────────
    unknown_cards = {}
    for pos in ("c1", "c2"):
        unknown_cards[pos] = dict(card_reason[pos].most_common())

    # P2 机器失败归因（定位用，已含在 gap.json）——剩余池复核
    stage = Counter(r.get("stage") for r in gap["rows"])
    best_char_left = Counter(
        (r["pos"], r["rank_best"][0]) for r in gap["rows"]
        if r.get("stage") == "rank_score_fail")

    result = {
        "coverage": stats,
        "recognized_matrix": matrix,
        "template_matrix": tpl_matrix,
        "unknown_card_reasons": unknown_cards,
        "p2_gap_stage_totals": dict(stage),
        "p2_gap_rankfail_by_pos_char": {"%s@%s" % (c, p): n
                                        for (p, c), n in
                                        best_char_left.most_common()},
        "collisions": scan["n_collisions"],
        "value_flips_vs_p2_baseline": "0（逐卡对比见 V3E-P3 REPAIR 节）",
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))
    print("template matrix:", json.dumps(tpl_matrix, ensure_ascii=False))
    for pos in ("c1", "c2"):
        print("== %s recognized (rank x s h d c = total)" % pos)
        for row in matrix[pos]["rows"]:
            print("   %s %s" % (row[0], row[1:]))
    print("unknown card reasons:", json.dumps(unknown_cards, ensure_ascii=False))
    print("p2 rankfail by pos/char:", json.dumps(
        result["p2_gap_rankfail_by_pos_char"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
