"""V3E-P3 — Hero Coverage Gap Audit（只读审计，不修改任何资产/逻辑）。

对 P2 归因出的三大缺口（4@c2 / J@c1+T@c1）：
  1. 候选帧提取（来自 P2 gap_analysis 的失败归因——机器仅定位，不产生 GT）；
  2. 字形聚类（两两 NCC≥0.85，同 P1 修复门控）；
  3. 时间会话聚类 + 用户 GT 锚帧上下文关联（GT provenance = 人工确认帧 + 同手连续性）；
  4. 形近风险实测：T/J、4/6（历史 rank_4 误标为 6）；
  5. 输出候选面板 PNG（logs/v3e_p3_candidates/）供人工确认。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3e_p1_hero_audit import EXT_WIN, EV, glyph, ncc_pair
from poker_decision_ai.vision2_platform.png_decode import decode_png
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png

GAP = Path("logs/v3e_p2_gap_analysis.json")
OUT = Path("logs/v3e_p3_gap_audit.json")
PANELS = Path("logs/v3e_p3_candidates")
ASSETS = Path("data/vision2/coinpoker_assets_v1876herocards")

# 用户人工 GT 锚帧（20 帧，全部经人工终审确认）
GT_FRAMES = {
    "POT_CHANGED_250_1787992907809_full.png": ("As", "3c"),
    "POT_CHANGED_524_1787993575599_full.png": ("Kd", "Jc"),
    "SCENE_CHANGED_429_1787993345715_full.png": ("Qh", "9s"),
    "POT_CHANGED_512_1787993546817_full.png": ("Td", "6h"),
    "POT_CHANGED_606_1787993775320_full.png": ("7c", "2c"),
    "TABLE_CHANGED_18_1787992293441_full.png": ("6c", "3h"),
    "POT_CHANGED_404_1787993286628_full.png": ("As", "5h"),
    "POT_CHANGED_16_1787995303501_full.png": ("Kc", "3h"),
    "POT_CHANGED_208_1787992806955_full.png": ("Ah", "5h"),
    "SCENE_CHANGED_6_1787992258912_full.png": ("8s", "8h"),
    "POT_CHANGED_708_1787994016446_full.png": ("Ks", "6c"),
    "POT_CHANGED_293_1787993018309_full.png": ("As", "Kc"),
    "POT_CHANGED_535_1787993603723_full.png": ("Qd", "Jc"),
    "POT_CHANGED_454_1787993404728_full.png": ("8c", "3d"),
    "POT_CHANGED_33_1787995356823_full.png": ("4s", "4h"),
    "POT_CHANGED_420_1787993325856_full.png": ("Qh", "9s"),
    "POT_CHANGED_353_1787993164702_full.png": ("Th", "5h"),
    "POT_CHANGED_47_1787995398974_full.png": ("4s", "4h"),
    "POT_CHANGED_181_1787992731553_full.png": ("Jd", "9h"),
    "POT_CHANGED_290_1787993011134_full.png": ("As", "Kc"),
}
CONTEXT_WINDOW_MS = 120_000  # 同手连续性窗口 ±120s

_cache = {}


def G(tag_frame, pos, kind):
    key = (tag_frame, pos, kind)
    if key not in _cache:
        w, h, rgba = decode_png((EV / tag_frame).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        m, bbox = glyph(arr, EXT_WIN[pos][kind])
        _cache[key] = (m, bbox)
    return _cache[key]


def ts_of(frame):
    return int(frame.rsplit("_", 2)[1])


def asset_mask(name):
    w, h, rgba = decode_png((ASSETS / name).read_bytes())
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    m = (arr[..., 3] > 128).astype(np.uint8)
    ys, xs = np.where(m > 0)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def cluster(masks, tags, gate=0.85):
    """贪心聚类：与簇内任一成员 ≥gate 即并入；返回簇列表。"""
    clusters = []
    for t, m in zip(tags, masks):
        placed = False
        for cl in clusters:
            if any(ncc_pair(m, mm) >= gate for mm in cl["masks"]):
                cl["tags"].append(t)
                cl["masks"].append(m)
                placed = True
                break
        if not placed:
            clusters.append({"tags": [t], "masks": [m]})
    for cl in clusters:
        n = len(cl["masks"])
        vals = [round(ncc_pair(cl["masks"][i], cl["masks"][j]), 3)
                for i in range(n) for j in range(i + 1, n)]
        cl["n"] = n
        cl.pop("masks")
        cl["min_pair"] = min(vals) if vals else 1.0
        cl["mean_pair"] = round(sum(vals) / len(vals), 3) if vals else 1.0
    return sorted(clusters, key=lambda c: -c["n"])


def gt_context(frame):
    t = ts_of(frame)
    hits = []
    for f, val in GT_FRAMES.items():
        if abs(ts_of(f) - t) <= CONTEXT_WINDOW_MS:
            hits.append({"anchor": f, "hand": val[0] + val[1],
                         "dt_s": round((t - ts_of(f)) / 1000, 1)})
    return hits


def save_panel(name, masks):
    if not masks:
        return
    h = max(m.shape[0] for m in masks)
    w = sum(m.shape[1] for m in masks) + 4 * (len(masks) - 1)
    canvas = np.zeros((h, w), dtype=np.uint8)
    x = 0
    for m in masks:
        canvas[:m.shape[0], x:x + m.shape[1]] = m
        x += m.shape[1] + 4
    PANELS.mkdir(parents=True, exist_ok=True)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[canvas > 0] = (255, 255, 255, 255)
    rgba[..., 3] = np.where(canvas > 0, 255, 40)
    (PANELS / name).write_bytes(encode_rgba_png(w, h, rgba.tobytes()))


def main() -> int:
    gap = json.loads(GAP.read_text(encoding="utf-8"))
    rows = gap["rows"]
    report = {"pools": {}, "risks": {}}

    def build_pool(name, pred):
        frames = [r["frame"] for r in rows if pred(r)]
        frames = sorted(set(frames), key=ts_of)
        masks, tags = [], []
        for f in frames:
            m, bbox = G(f, 2 if name == "4@c2" else 1, "rank")
            if m is not None:
                masks.append(m)
                tags.append(f)
        cls = cluster(masks, tags)
        for cl in cls:
            reps = sorted(cl["tags"], key=ts_of)
            cl["first"] = reps[0]
            cl["last"] = reps[-1]
            ctx = {}
            for t in cl["tags"]:
                for h in gt_context(t):
                    ctx.setdefault(h["hand"], []).append(
                        {"frame": t[:44], "anchor": h["anchor"][:44],
                         "dt_s": h["dt_s"]})
            cl["gt_context_hands"] = {
                k: {"n": len(v), "examples": v[:3]} for k, v in ctx.items()}
        # 时间会话分布
        sessions = []
        for t in tags:
            if sessions and ts_of(t) - ts_of(sessions[-1][-1]) <= 30_000:
                sessions[-1].append(t)
            else:
                sessions.append([t])
        report["pools"][name] = {
            "n_candidates": len(frames),
            "n_extracted": len(tags),
            "clusters": cls,
            "n_time_sessions_30s": len(sessions),
            "session_sizes": [len(s) for s in sessions],
        }
        save_panel("panel_%s.png" % name.replace("@", "_"),
                   [m for m in masks])
        return tags, masks

    build_pool("4@c2", lambda r: r["pos"] == 2
               and r.get("stage") == "rank_score_fail"
               and r["rank_best"][0] == "4")
    tj_tags, tj_masks = build_pool(
        "TJ@c1", lambda r: r["pos"] == 1
        and r.get("stage") == "rank_score_fail"
        and r["rank_best"][0] in ("T", "J"))
    build_pool("T@c1", lambda r: r["pos"] == 1
               and r.get("stage") == "rank_score_fail"
               and r["rank_best"][0] == "T")
    build_pool("J@c1", lambda r: r["pos"] == 1
               and r.get("stage") == "rank_score_fail"
               and r["rank_best"][0] == "J")

    # ── 风险实测 ────────────────────────────────────────────────
    t512 = asset_mask("hrT_c1_836_1028.png")
    j_c2_rot = asset_mask("hrJ_c2_947_1019.png")   # c2 旋转 J 实例
    j181, _ = G("POT_CHANGED_181_1787992731553_full.png", 1, "rank")
    t353, _ = G("POT_CHANGED_353_1787993164702_full.png", 1, "rank")
    six_c2_a = asset_mask("hcr_6_c2_942_1018.png")   # relabel 自 705
    six_c2_b = asset_mask("hr6_c2_942_1022.png")     # 512 实例
    four_c1 = asset_mask("hr4_c1_824_1029.png")
    six_c1 = asset_mask("hcr_6_c1_823_1026.png")

    # 4@c2 候选代表 vs 6@c2 模板 / c1-4 / c1-6
    g4 = report["pools"]["4@c2"]
    four_c2_reps = []
    seen = set()
    for cl in g4["clusters"]:
        rep = cl["first"]
        if rep not in seen:
            seen.add(rep)
            m, _ = G(rep, 2, "rank")
            four_c2_reps.append((rep, m))
    risks4 = []
    for rep, m in four_c2_reps:
        risks4.append({
            "frame": rep[:48],
            "vs_6c2_relabel_705": round(ncc_pair(m, six_c2_a), 3),
            "vs_6c2_512": round(ncc_pair(m, six_c2_b), 3),
            "vs_4c1": round(ncc_pair(m, four_c1), 3),
            "vs_6c1": round(ncc_pair(m, six_c1), 3) if six_c1 is not None else None,
        })
    report["risks"]["4_vs_6"] = risks4

    # TJ 池代表 vs T-512 / 旋转 J@c2 / J-181 / T-353
    risks_tj = []
    seen = set()
    for cl in report["pools"]["TJ@c1"]["clusters"]:
        for rep in (cl["first"], cl["last"]):
            if rep in seen:
                continue
            seen.add(rep)
            m, _ = G(rep, 1, "rank")
            risks_tj.append({
                "frame": rep[:48],
                "vs_T512": round(ncc_pair(m, t512), 3),
                "vs_J181_upright": round(ncc_pair(m, j181), 3),
                "vs_T353": round(ncc_pair(m, t353), 3),
                "vs_Jc2_rotated": round(ncc_pair(m, j_c2_rot), 3),
            })
    report["risks"]["TJ"] = risks_tj
    report["risks"]["template_cross"] = {
        "T512_vs_J181": round(ncc_pair(t512, j181), 3),
        "T512_vs_T353": round(ncc_pair(t512, t353), 3),
        "J181_vs_T353": round(ncc_pair(j181, t353), 3),
        "J181_vs_Jc2_rotated": round(ncc_pair(j181, j_c2_rot), 3),
        "T512_vs_Jc2_rotated": round(ncc_pair(t512, j_c2_rot), 3),
        "T512_vs_6c1": round(ncc_pair(t512, six_c1), 3) if six_c1 is not None else None,
        "T512_vs_8c1": None,
        "4c1_vs_6c1": round(ncc_pair(four_c1, six_c1), 3) if six_c1 is not None else None,
    }

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    for name, p in report["pools"].items():
        print("==", name, "cand=%d extracted=%d sessions=%d clusters=%d"
              % (p["n_candidates"], p["n_extracted"], p["n_time_sessions_30s"],
                 len(p["clusters"])))
        for cl in p["clusters"][:6]:
            print("   cluster n=%d min=%s mean=%s ctx=%s"
                  % (cl["n"], cl["min_pair"], cl["mean_pair"],
                     {k: v["n"] for k, v in cl["gt_context_hands"].items()}))
    print("risks4:")
    for r in risks4[:8]:
        print("  ", r)
    print("risksTJ:")
    for r in risks_tj[:10]:
        print("  ", r)
    print("template_cross:", json.dumps(report["risks"]["template_cross"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
