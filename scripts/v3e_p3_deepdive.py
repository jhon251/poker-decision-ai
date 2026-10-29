"""V3E-P3 深挖：4@c2 簇归属 / T@c1 位置偏移假说 / J@c1 全同帧清单。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3e_p1_hero_audit import EXT_WIN, EV, glyph, ncc_pair
from v3e_p3_gap_audit import G, GT_FRAMES, ts_of, asset_mask
from poker_decision_ai.vision2_platform.png_decode import decode_png

OUT = Path("logs/v3e_p3_deepdive.json")


def main() -> int:
    audit = json.loads(Path("logs/v3e_p3_gap_audit.json").read_text(encoding="utf-8"))
    out = {}

    # ── 1) 4@c2：重建逐对关系，定位 33/47 所在簇 ──────────────────
    rows = json.loads(Path("logs/v3e_p2_gap_analysis.json").read_text(encoding="utf-8"))["rows"]
    g4 = sorted({r["frame"] for r in rows
                 if r["pos"] == 2 and r.get("stage") == "rank_score_fail"
                 and r["rank_best"][0] == "4"}, key=ts_of)
    pairs33 = {}
    m33, _ = G("POT_CHANGED_33_1787995356823_full.png", 2, "rank")
    m47, _ = G("POT_CHANGED_47_1787995398974_full.png", 2, "rank")
    out["4c2_anchor_pair"] = round(ncc_pair(m33, m47), 3)
    near = [f for f in g4 if abs(ts_of(f) - ts_of("POT_CHANGED_33_1787995356823_full.png")) <= 120_000]
    out["4c2_frames_near_33"] = len(near)
    near47 = [f for f in g4 if abs(ts_of(f) - ts_of("POT_CHANGED_47_1787995398974_full.png")) <= 120_000]
    out["4c2_frames_near_47"] = len(near47)
    # 与 33 同簇（≥0.85）与同 33 时段帧的 NCC 分布
    sims = sorted((round(ncc_pair(m33, G(f, 2, "rank")[0]), 3) for f in near),
                  reverse=True)
    out["4c2_ncc_vs_33_top10"] = sims[:10]
    out["4c2_ncc_vs_33_bottom5"] = sims[-5:]
    # 各帧 bbox x 偏移（4@c2 位置漂移）
    shifts = []
    for f in g4[:69]:
        m, b = G(f, 2, "rank")
        shifts.append({"f": f[:40], "bbox": list(b)})
    xs = [s["bbox"][0] for s in shifts]
    out["4c2_bbox_x_range"] = [min(xs), max(xs)]
    ys = [s["bbox"][1] for s in shifts]
    out["4c2_bbox_y_range"] = [min(ys), max(ys)]

    # ── 2) T@c1 簇 B（T-512 全同帧）位置偏移假说 ─────────────────
    t512_frame = "POT_CHANGED_512_1787993546817_full.png"
    _, b512 = G(t512_frame, 1, "rank")
    tj = audit["pools"]["TJ@c1"]["clusters"]
    # 簇 B = TJ@c1 第 2 簇（n=41）：用 vs_T512==1.0 找成员
    rows_tj = [r for r in rows if r["pos"] == 1
               and r.get("stage") == "rank_score_fail"
               and r["rank_best"][0] in ("T", "J")]
    membersB = []
    for r in rows_tj:
        f = r["frame"]
        m, b = G(f, 1, "rank")
        if m is None:
            continue
        if ncc_pair(m, G(t512_frame, 1, "rank")[0]) >= 0.99:
            membersB.append({"f": f[:44], "bbox": list(b),
                             "dx": b[0] - b512[0], "dy": b[1] - b512[1],
                             "win_score": r["rank_best"][1],
                             "win_margin": r["rank_best"][2]})
    out["T_clusterB_n"] = len(membersB)
    out["T_clusterB_dx_range"] = ([min(m["dx"] for m in membersB),
                                   max(m["dx"] for m in membersB)]
                                  if membersB else None)
    out["T_clusterB_dy_range"] = ([min(m["dy"] for m in membersB),
                                   max(m["dy"] for m in membersB)]
                                  if membersB else None)
    out["T_clusterB_win_scores"] = sorted({m["win_score"] for m in membersB})
    out["T_clusterB"] = membersB[:45]

    # ── 3) J@c1：与 J-181 全同帧 + GT 上下文 ────────────────────
    j181_frame = "POT_CHANGED_181_1787992731553_full.png"
    mj181, bj181 = G(j181_frame, 1, "rank")
    j_ident = []
    for r in rows_tj:
        f = r["frame"]
        m, b = G(f, 1, "rank")
        if m is None or f == j181_frame:
            continue
        v = ncc_pair(m, mj181)
        if v >= 0.99:
            ctx = [(a[:40], h, d) for a, h, d in
                   [(gf, hv[0] + hv[1],
                     round((ts_of(f) - ts_of(gf)) / 1000, 1))
                    for gf, hv in GT_FRAMES.items()
                    if abs(ts_of(f) - ts_of(gf)) <= 120_000]]
            j_ident.append({"f": f[:44], "ncc_vs_J181": round(v, 3),
                            "ctx": sorted(ctx, key=lambda x: abs(x[2]))[:3]})
    out["J181_identical_frames"] = j_ident
    out["n_j181_identical"] = len(j_ident)

    # ── 4) 4@c2 会话时间轴 ─────────────────────────────────────
    sessions = []
    for f in g4:
        t = ts_of(f)
        if sessions and t - sessions[-1][-1][1] <= 30_000:
            sessions[-1].append((f[:40], t))
        else:
            sessions.append([(f[:40], t)])
    out["4c2_sessions"] = [{"n": len(s),
                            "span": [s[0][0], s[-1][0]]} for s in sessions]

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("T_clusterB", "4c2_ncc_vs_33_top10")},
                     ensure_ascii=False, indent=1)[:3500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
