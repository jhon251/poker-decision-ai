"""V3E-P1 — Hero 覆盖扩充入口审计（只读分析；仅写 logs/）。

不修改 detector/threshold/模板/manifest/GT/vision contract/P30-P35。
输出（logs/v3e_p1_hero_audit.json + stdout 摘要）：
  1) 标注池逐卡 glyph 重提取（rebuild 机制）→ 目标 rank 按 (rank,pos)
     成对 NCC 内聚度；
  2) ♣/♠ 语义裁决证据：标注 s 实例 vs 既有 c/d/h 模板 NCC、s 实例
     成对 NCC（分位置）、c 实例 vs s 候选交叉（加模板风险）、
     逐实例 (s_pair − vs_c) margin 预演（模板可行性核心证据）；
  3) 33k 全池挖掘：rank/suit glyph 增量聚类（NCC≥0.6 连通），
     输出聚类清单 + 代表帧（供人工确认清单——机器只聚类，不标签）。

确定性：无随机数；聚类按 (pos, frame) 稳定顺序增量合并。
"""
from __future__ import annotations

import csv
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (
    HeroCardsRecognizer1876, _red_frac)
from poker_decision_ai.vision2_platform.png_decode import decode_png

EV = Path("data/vision2/coinpoker_evidence")
ASSETS = Path("data/vision2/coinpoker_assets_v1876herocards")
CSV = Path("gt_confirm/all_unique_candidates.csv")
OUT = Path("logs/v3e_p1_hero_audit.json")

# rebuild 版提取窗（与 p33_hero_gt_adjudicate.py / phase_v2s_p29b_rebuild 一致）
EXT_WIN = {1: {"rank": (822, 1022, 852, 1070), "suit": (822, 1076, 864, 1124)},
           2: {"rank": (938, 1014, 970, 1064), "suit": (932, 1070, 978, 1116)}}
SUIT_WIN = {1: (822, 1074, 864, 1124), 2: (936, 1050, 980, 1116)}
DIGITS = set("23456789TJQKA")
SUITS = set("cdhs")
NCC_JOIN = 0.6          # 聚类合并门（与 V3D 一致）


# ---------------------------------------------------------------- glyph 提取
def blocks(arr, x0, y0, x1, y1, min_n=120):
    roi = arr[y0:y1, x0:x1]
    l = (roi[..., 0] * 0.299 + roi[..., 1] * 0.587 + roi[..., 2] * 0.114)
    ink = (l < 120).astype(bool)
    h, w = ink.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for sy in range(h):
        for sx in range(w):
            if not ink[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            cells = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cx, cy))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy or dx:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h and 0 <= nx < w and ink[ny, nx] \
                                    and not seen[ny, nx]:
                                seen[ny, nx] = True
                                stack.append((ny, nx))
            if len(cells) < min_n:
                continue
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            out.append((x0 + min(xs), y0 + min(ys),
                        max(xs) - min(xs) + 1, max(ys) - min(ys) + 1))
    return out


def glyph(arr, win):
    """tight glyph mask（宽度 12..50 / 高度 30..56，rebuild 过滤）。"""
    for (rx, ry, rw, rh) in blocks(arr, *win):
        if 12 <= rw <= 50 and 30 <= rh <= 56:
            roi = arr[ry:ry + rh, rx:rx + rw]
            l = roi[..., 0] * 0.299 + roi[..., 1] * 0.587 + roi[..., 2] * 0.114
            return (l < 120), (rx, ry, rw, rh)
    return None, None


def ncc_pair(a, b):
    """双向最大 NCC（_match_glyph 冻结公式）。"""
    def one(g, t):
        gp = np.pad(g.astype(np.float64), 2)
        t = t.astype(np.float64)
        th, tw = t.shape
        if th >= gp.shape[0] or tw >= gp.shape[1]:
            return 0.0
        t0 = t - t.mean()
        tn = float(np.sqrt((t0 * t0).sum()))
        if tn < 1e-9:
            return 0.0
        wins = np.lib.stride_tricks.sliding_window_view(gp, (th, tw))
        wm = wins.mean(axis=(2, 3), keepdims=True)
        wc = wins - wm
        wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
        return float(np.abs((wc * t0).sum(axis=(2, 3))
                            / np.maximum(wn * tn, 1e-9)).max())
    return max(one(a, b), one(b, a))


def parse_card(tok):
    tok = tok.strip().replace("10", "T")
    if len(tok) == 2 and tok[0].upper() in DIGITS and tok[1].lower() in SUITS:
        return tok[0].upper(), tok[1].lower()
    return None, None


def load_labels():
    """与 V3D 相同的边界规则解析人工标注（只读）。"""
    labels = {}
    boundary = False
    for r in csv.DictReader(open(CSV, encoding="utf-8-sig")):
        val = (r["rec_value"] or "").strip()
        notes = r["rec_reason"] or ""
        if "以下没有了" in val or "以下没有了" in notes:
            boundary = True
        if boundary:
            continue
        m = re.findall(r"'([^']*)'", val)
        lab = {1: None, 2: None}
        if len(m) >= 2:
            for i, tok in enumerate(m[:2]):
                if tok in ("?", "") or "截" in tok:
                    continue
                rk, st = parse_card(tok)
                if rk and st:
                    lab[i + 1] = (rk, st)
        if lab[1] or lab[2]:
            labels[r["frame"]] = lab
    return labels


# ---------------------------------------------------------------- 逐帧作业
def _job(args):
    frame, lab, want_machine = args
    try:
        w, h, rgba = decode_png((EV / frame).read_bytes())
    except Exception as e:  # noqa: BLE001
        return {"frame": frame, "error": str(e)[:80]}
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    out = {"frame": frame, "cards": {}}
    if want_machine:
        try:
            out["machine"] = HeroCardsRecognizer1876().recognize(arr)
        except Exception as e:  # noqa: BLE001
            out["machine"] = {"error": str(e)[:80]}
    for ci in (1, 2):
        d = {}
        if lab and lab.get(ci):
            d["user"] = f"{lab[ci][0]}{lab[ci][1]}"
        for kind, win, key in (("rank", EXT_WIN[ci]["rank"], "r"),
                               ("suit", EXT_WIN[ci]["suit"], "s")):
            m, bbox = glyph(arr, win)
            if m is not None:
                d[key] = {"bbox": list(bbox), "mask": m.astype(np.uint8)}
        d["red_frac"] = round(float(_red_frac(arr, SUIT_WIN[ci])), 3)
        if d.get("r") or d.get("s"):
            out["cards"][ci] = d
    return out


def load_templates_for(ch):
    """资产目录 suit_{ch} 模板（alpha 掩码 → float64 tight）。"""
    out = []
    mp = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    for a in mp.get("assets", []):
        if a.get("label") == f"suit_{ch}":
            p = ASSETS / Path(a["crop_ref"]).name
            if p.exists():
                w, h, rgba = decode_png(p.read_bytes())
                arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
                g = (arr[..., 3] > 128).astype(np.float64)
                ys, xs = np.where(g > 0)
                g = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                out.append(g)
    return out


def pairwise_stats(items):
    """同位置成对 NCC 统计。items: [(frame, pos, mask)]"""
    vals = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][1] != items[j][1]:
                continue
            vals.append(ncc_pair(items[i][2], items[j][2]))
    if not vals:
        return {"pairs": 0}
    return {"pairs": len(vals), "min": round(min(vals), 3),
            "mean": round(sum(vals) / len(vals), 3),
            "max": round(max(vals), 3)}


def incremental_cluster(items, join=NCC_JOIN):
    """确定性增量聚类：按给定顺序逐个并入首个 NCC≥join 的同位置簇。"""
    clusters = []
    for frame, pos, mask, bbox in items:
        placed = False
        for cl in clusters:
            if pos != cl["pos"]:
                continue
            if ncc_pair(mask, cl["rep"]) >= join:
                cl["members"].append((frame, pos, bbox))
                placed = True
                break
        if not placed:
            clusters.append({"pos": pos, "rep": mask,
                             "members": [(frame, pos, bbox)]})
    return clusters


def main() -> int:
    phase = sys.argv[1] if len(sys.argv) > 1 else "1"
    labels = load_labels()
    print(f"annotated frames: {len(labels)}", flush=True)

    if phase == "1":
        # 阶段 1：仅标注池（116 帧）——♣/♠ 裁决 + rank 内聚 + 跨位转移
        ev_frames = sorted(labels.keys())
    else:
        # 阶段 2：33k 池确定性等步长抽样挖掘缺失位置（每 6 帧取 1）
        ev_frames = sorted(p.name for p in EV.glob("*.png"))[::6]
    print(f"frames to scan: {len(ev_frames)}", flush=True)
    labeled_set = set(labels)

    jobs = [(f, labels.get(f), f in labeled_set) for f in ev_frames]
    results = []
    with Pool(8) as pool:
        for i, res in enumerate(pool.imap_unordered(_job, jobs, chunksize=8)):
            results.append(res)
            if (i + 1) % 500 == 0:
                print(f"[{i + 1}/{len(jobs)}] scanned", flush=True)
    ok_results = [r for r in results if not r.get("error")]
    print(f"decoded ok: {len(ok_results)}", flush=True)

    # ── 1) 标注池：目标 rank (rank,pos) 内聚 + ♣/♠ 证据 ────────────────
    tagged = []
    for res in ok_results:
        lab = labels.get(res["frame"])
        if not lab:
            continue
        for ci, d in (res.get("cards") or {}).items():
            rk, st = lab.get(int(ci)) or (None, None)
            tagged.append({"frame": res["frame"], "pos": int(ci),
                           "rank": rk, "suit": st,
                           "r": d.get("r"), "s": d.get("s")})

    def masks_of(pred, key):
        return [(t["frame"], t["pos"],
                 np.array(t[key]["mask"], dtype=np.uint8))
                for t in tagged if pred(t) and t.get(key)]

    audit = {"targets": {}, "cs_adjudication": {}, "mining": {}}

    for rk in "23456789TJQKA":
        grp = masks_of(lambda t, rk=rk: t["rank"] == rk, "r")
        if not grp:
            continue
        by_pos = {}
        for pos in (1, 2):
            sub = [x for x in grp if x[1] == pos]
            if len(sub) >= 2:
                by_pos[f"c{pos}"] = {"n": len(sub), **pairwise_stats(sub)}
            elif len(sub) == 1:
                by_pos[f"c{pos}"] = {"n": 1, "frame": sub[0][0]}
        # 跨位转移证据：同 rank c1 与 c2 glyph 交叉 NCC（若高 → 单模板可服务双位）
        x1 = [x for x in grp if x[1] == 1]
        x2 = [x for x in grp if x[1] == 2]
        if x1 and x2:
            cross_vals = [ncc_pair(a[2], b[2]) for a in x1[:8] for b in x2[:8]]
            by_pos["c1_vs_c2"] = {
                "pairs": len(cross_vals), "min": round(min(cross_vals), 3),
                "mean": round(sum(cross_vals) / len(cross_vals), 3),
                "max": round(max(cross_vals), 3)}
        audit["targets"][rk] = by_pos

    # ♣/♠ 裁决
    s_items = masks_of(lambda t: t["suit"] == "s", "s")
    c_items = masks_of(lambda t: t["suit"] == "c", "s")
    c_t = load_templates_for("c")
    d_t = load_templates_for("d")
    h_t = load_templates_for("h")

    def vs_templates(items, tmpls):
        if not items or not tmpls:
            return {}
        vals = [max(ncc_pair(m, t) for t in tmpls) for (_f, _p, m) in items]
        return {"n": len(vals), "min": round(min(vals), 3),
                "mean": round(sum(vals) / len(vals), 3),
                "max": round(max(vals), 3)}

    cs = audit["cs_adjudication"]
    cs["s_instances"] = len(s_items)
    cs["c_instances"] = len(c_items)
    cs["s_vs_c_template"] = vs_templates(s_items, c_t)
    cs["s_vs_d_template"] = vs_templates(s_items, d_t)
    cs["s_vs_h_template"] = vs_templates(s_items, h_t)
    cs["s_intra_c1"] = pairwise_stats([x for x in s_items if x[1] == 1])
    cs["s_intra_c2"] = pairwise_stats([x for x in s_items if x[1] == 2])
    cs["c_intra"] = pairwise_stats(c_items)
    cross = [max(ncc_pair(cm, sm) for (_f2, _p2, sm) in s_items)
             for (_f1, _p1, cm) in c_items] if s_items else []
    cs["c_vs_s_cross_max"] = round(max(cross), 3) if cross else None
    margins = []
    for (f, p, m) in s_items:
        others = [x for x in s_items if x[1] == p and x[0] != f]
        s_best = max((ncc_pair(m, o[2]) for o in others), default=0.0)
        c_best = max(ncc_pair(m, t) for t in c_t) if c_t else 0.0
        margins.append({"frame": f, "pos": p, "s_pair": round(s_best, 3),
                        "vs_c": round(c_best, 3),
                        "margin": round(s_best - c_best, 3)})
    margins.sort(key=lambda x: (x["margin"], x["frame"]))
    cs["s_margin_preview"] = margins

    # ── 2) 33k 全池挖掘：glyph 增量聚类（人工确认清单用） ──────────────
    for kind, key in (("rank", "r"), ("suit", "s")):
        items = []
        for res in ok_results:
            for ci, d in (res.get("cards") or {}).items():
                g = d.get(key)
                if g is not None:
                    items.append((res["frame"], int(ci), g["mask"],
                                  tuple(g["bbox"])))
        items.sort(key=lambda x: (x[1], x[0]))       # 确定性顺序
        clusters = incremental_cluster(items)
        multi = sorted(
            ({"pos": cl["pos"], "n": len(cl["members"]),
              "frames": [m[0] for m in cl["members"]][:12],
              "bbox_first": list(cl["members"][0][2])} for cl in clusters
             if len(cl["members"]) >= 2),
            key=lambda c: (-c["n"], c["frames"][0]))
        audit["mining"][kind] = {
            "glyphs": len(items),
            "clusters_ge2": len(multi),
            "singletons": sum(1 for cl in clusters
                              if len(cl["members"]) == 1),
            "top": multi[:80]}
        print(f"mining {kind}: glyphs={len(items)} clusters>=2={len(multi)}",
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out_path = OUT.with_name(f"v3e_p1_hero_audit_p{phase}.json")
    out_path.write_text(json.dumps(audit, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
