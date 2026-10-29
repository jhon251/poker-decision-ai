"""M4 Phase A3 — 旋钮复扫汇总 + Phase Gate 判决。

输入：docs/bench_r1/m4_knob_{fish,freqbot}_{b005,b030}.{g0,g1}.*（6,000 手/格）。

判决（任务书冻结的闸门条件，两条都必须成立）：
  (a) 结构分叉：同一对手的低/高预算格之间，同手号 records 出现真实分叉
      （hero_actions 逐手比对），且分叉发生在有界偏离通道上；
  (b) 剥削有效性：软对手（fish）上高预算 bb/100 显著高于低预算——
      配对逐手差（同 deal_seed）均值 95% CI 下界 > 0。

任一条不成立 → GATE 不通过（输出 verdict=FAIL + 根因），M4 停在 Phase A。
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "m4_gate" / "knob"      # 段工件首选位置（R6 收尾）
OUT_DIR_FALLBACK = ROOT / "docs" / "bench_r1"


def _artifact(arm: str, seg: str, ext: str) -> pathlib.Path:
    for d in (OUT_DIR, OUT_DIR_FALLBACK):
        p = d / f"{arm}.{seg}{ext}"
        if p.is_file():
            return p
    return d / f"{arm}.{seg}{ext}"
RESULT = ROOT / "docs" / "m4_gate" / "knob_scan_m4.json"

SEGMENTS = ["g0", "g1", "g2"]
PAIRS = [("fish", "m4_knob_fish_b005", "m4_knob_fish_b030"),
         ("freqbot", "m4_knob_freqbot_b005", "m4_knob_freqbot_b030")]
SOFT_OPPONENT = "fish"


def load_records(arm: str) -> list[dict]:
    rows: list[dict] = []
    for seg in SEGMENTS:
        p = _artifact(arm, seg, ".records.jsonl")
        for ln in p.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                rows.append(json.loads(ln))
    return rows


def md5(path: pathlib.Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def paired_diff_ci(lo: list[dict], hi: list[dict]) -> dict:
    """同 deal_seed 配对逐手差（hi − lo，bb/手）+ 95% CI（bb/100 口径）。"""
    by_seed = {r.get("deal_seed"): r for r in lo
               if r.get("deal_seed") is not None and "hero_net_bb" in r}
    diffs = []
    for r in hi:
        s = r.get("deal_seed")
        if s is None or "hero_net_bb" not in r:
            continue
        base = by_seed.get(s)
        if base is None:
            continue
        diffs.append(r["hero_net_bb"] - base["hero_net_bb"])
    n = len(diffs)
    if n < 2:
        return {"n_paired": n, "ok": False}
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1)
    se = math.sqrt(var / n)
    lo95 = mean - 1.96 * se
    return {
        "n_paired": n,
        "mean_diff_bb_per_hand": round(mean, 5),
        "mean_diff_bb_per_100": round(mean * 100, 3),
        "ci95_bb_per_100": [round((mean - 1.96 * se) * 100, 3),
                            round((mean + 1.96 * se) * 100, 3)],
        "ci_lower_bb_per_100": round(lo95 * 100, 3),
        "positive_diff_hands": sum(1 for d in diffs if d > 0),
        "ok": bool(lo95 > 0),
    }


def action_divergence(lo: list[dict], hi: list[dict]) -> dict:
    """同手（deal_seed 配对）hero_actions 分叉统计（结构分叉证据）。

    R6 修复：以 deal_seed（含绝对手索引的确定性派生）为配对键。
    旧版用 hand_index 配对是错的——records 的 hand_index 是 worker 内
    相对索引（0..chunk-1），跨 worker/跨段大量碰撞 → 假分叉。
    """
    by_seed = {r.get("deal_seed"): r for r in hi
               if r.get("deal_seed") is not None}
    diverged = []
    n_compared = 0
    for r in lo:
        other = by_seed.get(r.get("deal_seed"))
        if other is None or "hero_actions" not in r:
            continue
        n_compared += 1
        a = [(x.get("a"), x.get("size_bb")) for x in r["hero_actions"]]
        b = [(x.get("a"), x.get("size_bb"))
             for x in other.get("hero_actions", [])]
        if a != b:
            diverged.append((r.get("deal_seed"), r.get("hand_index")))
    return {"n_compared": n_compared, "n_diverged": len(diverged),
            "diverged_hand_indices": [d[1] for d in diverged[:200]]}


def segment_layout(arm: str, seg: str) -> int | None:
    """从段 json 的 per_chunk 推断 worker 数（跨布局比较无效性的守卫：
    bot rng 用 worker 内相对 hand_index → 8w 与 4w 的同段不可直接配对）。"""
    p = _artifact(arm, seg, ".json")
    if not p.is_file():
        return None
    try:
        agg = json.loads(p.read_text(encoding="utf-8"))
        return len(agg.get("per_chunk") or {})
    except Exception:                                 # noqa: BLE001
        return None


def main() -> int:
    arms_available = all(
        _artifact(arm, seg, ".json").is_file()
        and _artifact(arm, seg, ".records.jsonl").is_file()
        for _name, lo, hi in PAIRS for arm in (lo, hi)
        for seg in SEGMENTS)
    if not arms_available:
        print("segments missing; run scripts/run_m4_knob_scan.py first",
              file=sys.stderr)
        return 2

    out: dict = {"pairs": {}, "gate": {}}
    all_ok = True
    for name, lo_arm, hi_arm in PAIRS:
        lo_rec, hi_rec = load_records(lo_arm), load_records(hi_arm)
        lo_md5 = {seg: md5(_artifact(lo_arm, seg, ".records.jsonl"))
                  for seg in SEGMENTS}
        hi_md5 = {seg: md5(_artifact(hi_arm, seg, ".records.jsonl"))
                  for seg in SEGMENTS}
        md5_differ = {seg: lo_md5[seg] != hi_md5[seg] for seg in SEGMENTS}
        div = action_divergence(lo_rec, hi_rec)
        # 分段细目 + 布局守卫（bot rng 用 worker 内相对索引：跨布局段不可比）
        div_per_seg: dict = {}
        layout = {}
        for seg in SEGMENTS:
            layout[seg] = {"lo_workers": segment_layout(lo_arm, seg),
                           "hi_workers": segment_layout(hi_arm, seg)}
            lo_s = [json.loads(ln) for ln in
                    _artifact(lo_arm, seg, ".records.jsonl")
                    .read_text(encoding="utf-8").splitlines() if ln.strip()]
            hi_s = [json.loads(ln) for ln in
                    _artifact(hi_arm, seg, ".records.jsonl")
                    .read_text(encoding="utf-8").splitlines() if ln.strip()]
            div_per_seg[seg] = action_divergence(lo_s, hi_s)["n_diverged"]
        pair_summary = {
            "lo_arm": lo_arm, "hi_arm": hi_arm,
            "n_hands_lo": len(lo_rec), "n_hands_hi": len(hi_rec),
            "records_md5": {"lo": lo_md5, "hi": hi_md5,
                            "differ": md5_differ},
            "action_divergence": div,
            "action_divergence_per_segment": div_per_seg,
            "segment_worker_layout": layout,
        }
        # bb/100 汇总（从段 json 合并重算）
        for tag, arm, recs in (("lo", lo_arm, lo_rec),
                               ("hi", hi_arm, hi_rec)):
            nets = [r["hero_net_bb"] for r in recs if "hero_net_bb" in r]
            n = len(nets)
            if n:
                tot = sum(nets)
                mean = tot / n
                var = sum((x - mean) ** 2 for x in nets) / (n - 1) if n > 1 else 0
                se = math.sqrt(var / n)
                pair_summary[f"bb_per_100_{tag}"] = round(100 * tot / n, 3)
                pair_summary[f"ci95_{tag}"] = [
                    round(100 * (mean - 1.96 * math.sqrt(var / (n - 1) if n > 1 else 0) / math.sqrt(n)), 3)
                    if n > 1 else None,
                    round(100 * (mean + 1.96 * math.sqrt(var / (n - 1) if n > 1 else 0) / math.sqrt(n)), 3)
                    if n > 1 else None]
                pair_summary[f"n_hands_{tag}"] = n
        pair_summary["paired_diff"] = paired_diff_ci(lo_rec, hi_rec)
        out["pairs"][name] = pair_summary

    fish = out["pairs"][SOFT_OPPONENT]
    cond_a = (all(fish["records_md5"]["differ"].values())
              and fish["action_divergence"]["n_diverged"] > 0)
    cond_b = bool(fish.get("paired_diff", {}).get("ok"))
    # freqbot 结构分叉也要求（接线非对手特异性）
    fb = out["pairs"]["freqbot"]
    cond_a2 = (all(fb["records_md5"]["differ"].values())
               and fb["action_divergence"]["n_diverged"] > 0)
    fb_hi_gt_lo = ((fb.get("bb_per_100_hi") is not None)
                   and (fb.get("bb_per_100_lo") is not None)
                   and fb["bb_per_100_hi"] > fb["bb_per_100_lo"])
    out["gate"] = {
        "condition_a_records_diverge": {
            "fish": cond_a, "freqbot": cond_a2,
            "detail": {
                "fish_diverged": fish["action_divergence"]["n_diverged"],
                "freqbot_diverged": fb["action_divergence"]["n_diverged"]}},
        "condition_b_soft_higher_significant": {
            "ok": cond_b,
            "detail": fish.get("paired_diff")},
        "freqbot_paired_diff": fb.get("paired_diff"),
        "freqbot_hi_gt_lo_point_estimate": fb_hi_gt_lo,
        # R6 协调方操作判据（偏离分叉为主、bb/100 差值为辅；fish 零偏离
        # 不单独否决，但作为结构性发现入报告）：
        "coordinator_criterion": {
            "freqbot_diverged_and_hi_better": bool(cond_a2 and fb_hi_gt_lo),
            "fish_zero_divergence_structural_note":
                (fish["action_divergence"]["n_diverged"] == 0)},
        "verdict_frozen_taskbook": ("PASS" if (cond_a and cond_a2 and cond_b)
                                    else "FAIL"),
        "note": ("闸门（任务书冻结）=结构分叉（两对手都要）+ 软对手配对差 "
                 "95% CI 下界>0；R6 协调方判据=偏离分叉为主、bb/100 为辅，"
                 "freqbot 分叉存在且 0.30 格点估更高即过（两者并列输出，"
                 "最终判决在 M4_GATE_REPORT.md 说明）"),
    }
    all_ok = bool(cond_a and cond_a2 and cond_b)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    print(json.dumps(out["gate"], ensure_ascii=False, indent=1))
    for name in ("fish", "freqbot"):
        p = out["pairs"][name]
        print(f"[{name}] lo={p.get('bb_per_100_lo')} hi={p.get('bb_per_100_hi')} "
              f"diverged={p['action_divergence']['n_diverged']} "
              f"per_seg={p['action_divergence_per_segment']}")
    print("verdict_frozen_taskbook:", out["gate"]["verdict_frozen_taskbook"])
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
