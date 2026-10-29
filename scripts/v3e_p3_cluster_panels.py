"""V3E-P3 重调查：去重(同 ts 多事件 tag=同帧) + 代表元紧聚类 + 逐簇面板。

供人工逐簇标注形状。输出 logs/v3e_p3_clusters/。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from v3e_p1_hero_audit import EXT_WIN, EV, glyph, ncc_pair
from poker_decision_ai.vision2_platform.png_decode import decode_png
from poker_decision_ai.vision2_platform.png_encode import encode_rgba_png

OUT = Path("logs/v3e_p3_clusters")
GAP = Path("logs/v3e_p2_gap_analysis.json")

_cache = {}


def G(frame, pos, kind):
    key = (frame, pos, kind)
    if key not in _cache:
        w, h, rgba = decode_png((EV / frame).read_bytes())
        arr = np.frombuffer(rgba, dtype=np.uint8).reshape(h, w, 4)
        m, bbox = glyph(arr, EXT_WIN[pos][kind])
        _cache[key] = (m, bbox)
    return _cache[key]


def ts_of(frame):
    return int(frame.rsplit("_", 2)[1])


def save_panel(name, masks):
    h = max(m.shape[0] for m in masks)
    w = sum(m.shape[1] for m in masks) + 6 * (len(masks) - 1)
    canvas = np.zeros((h, w), dtype=np.uint8)
    x = 0
    for m in masks:
        canvas[:m.shape[0], x:x + m.shape[1]] = m
        x += m.shape[1] + 6
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[canvas > 0] = (255, 255, 255, 255)
    rgba[..., 3] = np.where(canvas > 0, 255, 30)
    (OUT / name).write_bytes(encode_rgba_png(w, h, rgba.tobytes()))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads(GAP.read_text(encoding="utf-8"))["rows"]

    def pool(pred):
        frames = sorted({r["frame"] for r in rows if pred(r)}, key=ts_of)
        # 同 ts 去重（POT/SCENE/TABLE 同刻 = 同帧多份）
        uniq, seen = [], set()
        for f in frames:
            if ts_of(f) not in seen:
                seen.add(ts_of(f))
                uniq.append(f)
        return uniq

    pools = {
        "4@c2": pool(lambda r: r["pos"] == 2
                     and r.get("stage") == "rank_score_fail"
                     and r["rank_best"][0] == "4"),
        "T@c1": pool(lambda r: r["pos"] == 1
                     and r.get("stage") == "rank_score_fail"
                     and r["rank_best"][0] == "T"),
        "J@c1": pool(lambda r: r["pos"] == 1
                     and r.get("stage") == "rank_score_fail"
                     and r["rank_best"][0] == "J"),
    }
    report = {}
    idx = 0
    for pname, frames in pools.items():
        pos = 2 if pname == "4@c2" else 1
        masks, tags = [], []
        for f in frames:
            m, b = G(f, pos, "rank")
            if m is not None:
                masks.append(m)
                tags.append(f)
        # 代表元紧聚类
        assigned = []
        for t, m in zip(tags, masks):
            for cl in assigned:
                if ncc_pair(m, cl["rep"]) >= 0.85:
                    cl["members"].append(t)
                    cl["min"] = min(cl["min"], round(ncc_pair(m, cl["rep"]), 3))
                    break
            else:
                assigned.append({"rep": m, "members": [t], "min": 1.0})
        assigned.sort(key=lambda c: -len(c["members"]))
        plist = []
        for cl in assigned:
            idx += 1
            n = len(cl["members"])
            if n < 2:
                continue
            tss = sorted(ts_of(t) for t in cl["members"])
            sessions = 1
            for a, b in zip(tss, tss[1:]):
                if b - a > 30_000:
                    sessions += 1
            cname = "c%02d_n%d.png" % (idx, n)
            save_panel(cname, cl["members"] and
                       [G(t, pos, "rank")[0] for t in cl["members"]])
            plist.append({
                "panel": cname, "n_masks": n, "n_unique_ts": len(set(tss)),
                "time_sessions_30s": sessions,
                "t_range": [tss[0], tss[-1]],
                "min_pair_vs_rep": cl["min"] if n > 1 else 1.0,
                "members": [t[:48] for t in cl["members"][:12]],
            })
        report[pname] = {"n_unique": len(frames), "clusters": plist}
        print("==", pname, "unique:", len(frames))
        for p in plist:
            print("   %s masks=%d uniq_ts=%d sessions=%d min_vs_rep=%.3f"
                  % (p["panel"], p["n_masks"], p["n_unique_ts"],
                     p["time_sessions_30s"], p["min_pair_vs_rep"]))
    Path("logs/v3e_p3_cluster_index.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
