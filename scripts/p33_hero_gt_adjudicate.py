"""V3D — 用户标注的像素级分诊（只读诊断，不改 detector/模板/阈值）。

对 gt_confirm/all_unique_candidates.csv 中的人工标注逐卡交叉验证：
  1) suit 颜色门（冻结公式 _red_frac：红 suit 需 rf≥0.15，黑 suit 需 rf≤0.6）
  2) rank/suit 与现有模板的 NCC（复用 _match_glyph 冻结公式）：
     - best≥0.7 且 label==用户标注 → CROSS_VALIDATED（像素与目视一致）
     - best≥0.7 且 label!=用户标注 → COLLISION（像素强烈支持别的 label → 排除）
     - best<0.7 → NEW_SHAPE（新字形；身份依据=用户目视，聚类再验）
  3) 同 (kind,ch,card) 新字形跨帧聚类（成对 NCC≥0.6），离群实例剔除
输出：logs/hero_gt_adjudication.json + GT_EXT 草案 logs/hero_gt_ext_draft.json
机器 OK 帧同样诊断 → 用户 vs 机器 分歧表（报告用；不改动冻结 GT）。
"""
from __future__ import annotations

import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 import (
    MIN_MARGIN, MIN_SCORE, RED_INK_FRAC_MIN, RANK_WINDOWS, SUIT_WINDOWS,
    _corner_mask, _load_templates, _red_frac)
from poker_decision_ai.vision2_platform.png_decode import decode_png

EV = Path("data/vision2/coinpoker_evidence")
CSV = Path("gt_confirm/all_unique_candidates.json")
OUTJ = Path("logs/hero_gt_adjudication.json")
DRAFT = Path("logs/hero_gt_ext_draft.json")

RANK_WIN = {1: RANK_WINDOWS["card1"], 2: RANK_WINDOWS["card2"]}
SUIT_WIN = {1: SUIT_WINDOWS["card1"], 2: SUIT_WINDOWS["card2"]}
DIGITS = set("23456789TJQKA")
SUITS = set("cdhs")

# rebuild 版提取窗（与 phase_v2s_p29b_rebuild.py 完全一致）
EXT_WIN = {1: {"rank": (822, 1022, 852, 1070), "suit": (822, 1076, 864, 1124)},
           2: {"rank": (938, 1014, 970, 1064), "suit": (932, 1070, 978, 1116)}}
MIN_N, MIN_W, MAX_W, MIN_H, MAX_H = 120, 12, 50, 30, 56


def blocks(arr, x0, y0, x1, y1, min_n=MIN_N):
    """8 邻接连通域（复刻 rebuild 版）。"""
    roi = arr[y0:y1, x0:x1]
    r = roi[..., 0].astype(np.int16)
    g = roi[..., 1].astype(np.int16)
    b = roi[..., 2].astype(np.int16)
    l = r * 0.299 + g * 0.587 + b * 0.114
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
                        if dy == 0 and dx == 0:
                            continue
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
                        max(xs) - min(xs) + 1, max(ys) - min(ys) + 1,
                        len(cells)))
    return out


def glyph_mask(arr, win):
    """rebuild 机制提取 tight glyph mask（无合格组件 → None）。"""
    for (rx, ry, rw, rh, _n) in blocks(arr, *win):
        if MIN_W <= rw <= MAX_W and MIN_H <= rh <= MAX_H:
            roi = arr[ry:ry + rh, rx:rx + rw]
            l = roi[..., 0] * 0.299 + roi[..., 1] * 0.587 + roi[..., 2] * 0.114
            return (l < 120), (rx, ry, rw, rh)
    return None, None


def full_scores(ink, prep):
    """_match_glyph 冻结公式，但返回全部 label 分数。"""
    gp = np.pad(ink, 2)
    scores = {}
    for ch, insts in prep.items():
        sc = -1.0
        for (t, t0, tn) in insts:
            th, tw = t.shape
            if th >= gp.shape[0] or tw >= gp.shape[1] or tn < 1e-9:
                continue
            wins = np.lib.stride_tricks.sliding_window_view(gp, (th, tw))
            wm = wins.mean(axis=(2, 3), keepdims=True)
            wc = wins - wm
            wn = np.sqrt((wc * wc).sum(axis=(2, 3)))
            s = np.abs((wc * t0).sum(axis=(2, 3)) / np.maximum(wn * tn, 1e-9))
            sc = max(sc, float(s.max()))
        scores[ch] = sc
    return scores


def parse_card(tok):
    tok = tok.strip().replace("10", "T")
    if len(tok) == 2 and tok[0].upper() in DIGITS and tok[1].lower() in SUITS:
        return tok[0].upper(), tok[1].lower()
    return None, None


def _diag(job):
    frame, labels, machine = job  # labels: {1:(r,s)|None, 2:...}
    from poker_decision_ai.vision2.pixel.realclient.coinpoker_herocards1876 \
        import HeroCardsRecognizer1876
    rec = HeroCardsRecognizer1876()
    try:
        w, h, rgba = decode_png((EV / frame).read_bytes())
    except Exception as e:  # noqa: BLE001
        return {"frame": frame, "error": str(e)[:80]}
    arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
    prep = rec._prep
    out = {"frame": frame, "machine": machine, "cards": {}}
    for ci, lab in labels.items():
        if lab is None:
            continue
        rk, st = lab
        r_ink = _corner_mask(arr, RANK_WIN[ci])
        s_ink = _corner_mask(arr, SUIT_WIN[ci])
        d = {"user": f"{rk}{st}",
             "rank_ink": round(float(r_ink.sum()), 1),
             "suit_ink": round(float(s_ink.sum()), 1)}
        if r_ink.sum() < 30 or s_ink.sum() < 20:
            d["verdict"] = "INK_INSUFFICIENT"
            out["cards"][ci] = d
            continue
        rs = full_scores(r_ink, prep)
        ss = full_scores(s_ink, prep)
        rdig = {k: v for k, v in rs.items() if k in DIGITS}
        ssuit = {k: v for k, v in ss.items() if k in SUITS}
        r_bl, r_bs = max(rdig.items(), key=lambda kv: kv[1])
        s_bl, s_bs = max(ssuit.items(), key=lambda kv: kv[1])
        rf = _red_frac(arr, SUIT_WIN[ci])
        d["rank_best"] = [r_bl, round(r_bs, 3)]
        d["suit_best"] = [s_bl, round(s_bs, 3)]
        d["red_frac"] = round(rf, 3)
        # 颜色门（冻结阈值用于用户所标 suit）
        if st in "hd":
            d["color"] = "OK" if rf >= RED_INK_FRAC_MIN else "COLOR_FAIL"
        else:
            d["color"] = "OK" if rf <= 0.6 else "COLOR_FAIL"
        # rank 判定
        if r_bs >= MIN_SCORE:
            d["rank_v"] = ("CROSS_VALIDATED" if r_bl == rk else "COLLISION")
        else:
            d["rank_v"] = ("UNVERIFIED_EXISTING" if rk in rdig
                           else "NEW_SHAPE")
        # suit 判定；♣/♠ 字形 0.93-0.98 相似 → 用户标 s 无法独立确认 →
        # 按用户指示一律 UNSURE（保留，不猜测）
        if st == "s":
            d["suit_v"] = "UNSURE_CS_SIMILAR"
        elif s_bs >= MIN_SCORE:
            d["suit_v"] = ("CROSS_VALIDATED" if s_bl == st else "COLLISION")
        else:
            d["suit_v"] = ("UNVERIFIED_EXISTING" if st in ssuit
                           else "NEW_SHAPE")
        # 提取 glyph（rebuild 机制）供聚类/后续提取
        for kind, win, key in (("rank", EXT_WIN[ci]["rank"], "rank_g"),
                               ("suit", EXT_WIN[ci]["suit"], "suit_g")):
            m, bbox = glyph_mask(arr, win)
            if m is not None:
                d[key] = {"bbox": list(bbox),
                          "mask": m.astype(np.uint8).tolist()}
        out["cards"][ci] = d
    return out


def _ncc_pair(a, b):
    """两个 tight mask 的双向最大 NCC（_match_glyph 公式）。"""
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


def main() -> int:
    import csv
    rows = list(csv.DictReader(open("gt_confirm/all_unique_candidates.csv",
                                    encoding="utf-8-sig")))
    boundary_hit = False
    jobs, stats = [], {"total": len(rows), "complete": 0, "incomplete": 0,
                       "unannotated": 0, "machine_ok": 0}
    for r in rows:
        val = (r["rec_value"] or "").strip()
        notes = r["rec_reason"] or ""
        if "以下没有了" in val or "以下没有了" in notes:
            boundary_hit = True
        if boundary_hit and "以下没有了" not in val:
            stats["unannotated"] += 1
            continue
        if "以下没有了" in val:
            stats["unannotated"] += 1
            continue
        m = re.findall(r"'([^']*)'", val)
        labels = {1: None, 2: None}
        ok2 = False
        if len(m) >= 2:
            for i, tok in enumerate(m[:2]):
                if tok in ("?", "") or "截" in tok:
                    labels[i + 1] = None
                else:
                    rk, st = parse_card(tok)
                    labels[i + 1] = (rk, st) if rk and st else None
            ok2 = labels[1] is not None or labels[2] is not None
        if not ok2 or "没有hero card" in val or "表情包" in notes:
            stats["incomplete"] += 1
            continue
        if r["rec_status"] == "OK":
            stats["machine_ok"] += 1
        else:
            stats["complete"] += 1
        jobs.append((r["frame"], labels, r["rec_status"]))
    print("stats:", stats, flush=True)

    results = []
    with Pool(12) as pool:
        for i, res in enumerate(pool.imap_unordered(_diag, jobs, chunksize=4)):
            results.append(res)
            if (i + 1) % 25 == 0:
                print(f"[{i + 1}/{len(jobs)}]", flush=True)

    # 聚类：新字形按 (kind,ch,ci) 成对 NCC；离群(<0.6)剔除
    groups = {}
    for res in results:
        for ci, d in (res.get("cards") or {}).items():
            for kind, gk, vk in (("rank", "rank_g", "rank_v"),
                                 ("suit", "suit_g", "suit_v")):
                if d.get(vk) == "NEW_SHAPE" and d.get(gk):
                    rk, st = _parse_user(d["user"])
                    ch = rk if kind == "rank" else st
                    groups.setdefault((kind, ch, int(ci)), []).append(
                        {"frame": res["frame"], "mask": np.array(
                            d[gk]["mask"], dtype=np.uint8),
                         "bbox": d[gk]["bbox"]})
    drops = set()
    for key, insts in sorted(groups.items()):
        if len(insts) < 2:
            continue
        # 迭代剔除最低均值 NCC 的实例，直至全部成对 ≥0.6（确定性）
        cur = list(insts)
        while len(cur) >= 2:
            n = len(cur)
            means = []
            for i in range(n):
                vs = [_ncc_pair(cur[i]["mask"], cur[j]["mask"])
                      for j in range(n) if j != i]
                means.append(sum(vs) / len(vs))
            worst = min(range(n), key=lambda i: means[i])
            bad = [_ncc_pair(cur[i]["mask"], cur[j]["mask"])
                   for i in range(n) for j in range(i + 1, n)]
            if min(bad) >= 0.6:
                break
            drops.add((key, cur[worst]["frame"]))
            print(f"cluster-outlier {key} {cur[worst]['frame'][:22]} "
                  f"meanNCC={means[worst]:.2f} minPair={min(bad):.2f}",
                  flush=True)
            cur.pop(worst)
    for key, insts in groups.items():
        kept = [x for x in insts if (key, x["frame"]) not in drops]
        if kept:
            groups[key] = kept

    # GT_EXT 草案（rank-only）：卡入选条件——
    #   rank 部分：NEW_SHAPE 且该 (rank,card位) 聚类一致（剔除离群后 ≥2 帧）
    #              （CROSS_VALIDATED = 已有 label，无新覆盖价值 → 跳过）
    #   suit 部分：必须为已有 label 且 CROSS_VALIDATED（运行时能认出该 suit）
    #              suit='s' → UNSURE_CS_SIMILAR（用户指示保留 unsure）→ 排除
    #   颜色门 OK；无聚类离群
    def _clustered(kind, ch, ci, frame=None):
        key = (kind, ch, ci)
        kept = groups.get(key, [])
        if len(kept) < 2:
            return False
        if frame is not None:
            return any(x["frame"] == frame for x in kept)   # 必须仍在保留组内
        return True
    draft, unsure = [], []
    for res in results:
        if res.get("machine") == "OK":
            continue
        for ci, d in (res.get("cards") or {}).items():
            rk, st = _parse_user(d["user"])
            reasons = []
            if d.get("rank_v") == "CROSS_VALIDATED":
                reasons.append("rank_existing")
            elif d.get("rank_v") == "NEW_SHAPE":
                if not _clustered("rank", rk, ci, res["frame"]):
                    reasons.append("rank_new_singleton")
            else:
                reasons.append("rank_" + str(d.get("rank_v")))
            if d.get("suit_v") == "UNSURE_CS_SIMILAR":
                reasons.append("suit_s_unsure")
            elif d.get("suit_v") != "CROSS_VALIDATED":
                reasons.append("suit_" + str(d.get("suit_v")))
            if d.get("color") != "OK":
                reasons.append("color_fail")
            if d.get("rank_v") == "NEW_SHAPE" and not d.get("rank_g"):
                reasons.append("no_glyph")
            if reasons:
                unsure.append({"frame": res["frame"], "card": ci,
                               "user": d["user"], "exclude": reasons})
                continue
            draft.append({"frame": res["frame"], "card": ci,
                          "rank": rk, "suit": st,
                          "rank_best": d.get("rank_best"),
                          "suit_best": d.get("suit_best")})
    OUTJ.write_text(json.dumps(
        {"stats": stats, "results": _strip_masks(results),
         "clusters": {f"{k[0]}_{k[1]}_c{k[2]}": [x["frame"] for x in v]
                      for k, v in groups.items()}, "drops": sorted(
              f"{k[0]}_{k[1]}_c{k[2]}|{f}" for k, f in drops)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    DRAFT.write_text(json.dumps({"draft": draft, "unsure": unsure},
                                ensure_ascii=False, indent=1),
                     encoding="utf-8")
    print(f"GT_EXT draft cards (rank-only): {len(draft)}", flush=True)
    from collections import Counter
    print("draft by label:", dict(Counter(f"{d['rank']}c{d['card']}"
                                          for d in draft)), flush=True)
    print("unsure/excluded cards:", len(unsure), flush=True)
    print("unsure reasons:", dict(Counter(
        r for u in unsure for r in u["exclude"])), flush=True)
    return 0


def _parse_user(s):
    return s[0], s[1]


def _strip_masks(results):
    out = []
    for res in results:
        r2 = {"frame": res.get("frame"), "machine": res.get("machine"),
              "error": res.get("error"), "cards": {}}
        for ci, d in (res.get("cards") or {}).items():
            d2 = {k: v for k, v in d.items() if not k.endswith("_g")}
            r2["cards"][ci] = d2
        out.append(r2)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
