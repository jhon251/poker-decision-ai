"""R10 KNOB — 旋钮再复扫汇总 + 可调性判决。

输入：docs/m4_gate/knob_r10/r10_knob_{fish,freqbot}_{b005,b030}.{g0,g1,g2}.*
（6,000 手/格；docs/bench_r1/ 为首选源，knob_r10/ 为镜像）。

R10 判据（任务书冻结，取代 M4 时代双条件闸门）：
  * 主判据 = 行为分叉：预算格（b005 vs b030）之间同 deal_seed records 出现
    真实分叉（hero_actions 逐手比对，分叉手数 > 0）；
  * 辅助 = 双预算 bb/100 对照（点估计 + 配对逐手差 95% CI）；
  * fish 零分叉**不再是否决项**（M3 已证结构性：fish 下注率 ~5%，偏离条件
    不触发）；freqbot 分叉存在 = "旋钮活了"证据。
  结论二值：knob_tunability = active（freqbot 分叉 > 0）/ inert（= 0）。

附加（诚实标注）：old-vs-new 对照（同 deal_seed 对 M4 复扫工件）——
  M4 工件 worker 布局混杂（fish g0/g1=8w、g2=4w；freqbot 全 4w），且 bot rng
  消费 worker 内相对 hand_index，跨布局手不可比；布局匹配段（freqbot 全部、
  fish g2）的差异才可归因于 EXE-1 facing 修复。对照结果只作证据，不进判决。
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
NEW_DIR = ROOT / "docs" / "m4_gate" / "knob_r10"
NEW_FALLBACK = ROOT / "docs" / "bench_r1"
OLD_DIR = ROOT / "docs" / "m4_gate" / "knob"
RESULT = ROOT / "docs" / "m4_gate" / "knob_scan_r10.json"

SEGMENTS = ["g0", "g1", "g2"]
#: R10 分段布局（任务书：每格 ≥5,000 手，分段 ≤2,000）。fish_b005 三段为
#: 2,000 手段（g0/g1/g2，offset 0/2000/4000）；其余格因本机外杀窗口不稳
#: （实测 ~23-48 min 波动）改 1,000 手段 × 6（k0..k5，offset 0..5000）。
#: deal_seed 按绝对手索引派生，分段大小不影响逐手配对。
SEG_MAP = {
    "r10_knob_fish_b005": ["k0", "k1", "k2", "k3", "k4", "k5"],
    "r10_knob_fish_b030": ["k0", "k1", "k2", "k3", "k4", "k5"],
    "r10_knob_freqbot_b005": ["k0", "k1", "k2", "k3", "k4", "k5"],
    "r10_knob_freqbot_b030": ["k0", "k1", "k2", "k3", "k4", "k5"],
}
#: fish_b005 的 g* 工件（2000 手段）保留，仅用于旧-vs-新同布局对照
#: （old g2 与 new g2 同为 4 worker × 500 手/worker → 可归因）。
LEGACY_NEW_SEGMENTS = {"r10_knob_fish_b005": ["g0", "g1", "g2"]}
OLD_SEG_MAP = {
    "r10_knob_fish_b005": ["k0", "k1", "k2", "k3", "k4", "k5"],
    "r10_knob_fish_b030": ["g0", "g1", "g2"],
    "r10_knob_freqbot_b005": ["g0", "g1", "g2"],
    "r10_knob_freqbot_b030": ["g0", "g1", "g2"],
}
PAIRS = [("fish", "r10_knob_fish_b005", "r10_knob_fish_b030"),
         ("freqbot", "r10_knob_freqbot_b005", "r10_knob_freqbot_b030")]
OLD_ARM_PREFIX = {"fish": "m4_knob_fish_", "freqbot": "m4_knob_freqbot_"}
# M4 复扫盘面实测布局（per_chunk 推断）：fish g0/g1 为 8 worker（250 手/片）、
# fish g2 与 freqbot 全部为 4 worker（500 手/片）。bot rng 用 worker 内相对
# hand_index → old-vs-new 逐手对照仅在两端同布局时有归因力。
OLD_LAYOUT_8W = {("fish", "g0"), ("fish", "g1")}
NEW_WORKERS = 4


def _read(d: pathlib.Path, arm: str, seg: str, ext: str):
    p = d / f"{arm}.{seg}{ext}"
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8")
    if ext == ".json":
        return json.loads(text)          # 单文档聚合
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def _read_new(arm: str, seg: str, ext: str) -> list:
    for d in (NEW_FALLBACK, NEW_DIR):
        rows = _read(d, arm, seg, ext)
        if rows is not None:
            return rows
    raise SystemExit(f"missing segment artifact: {arm}.{seg}{ext}")


def load_records(arm: str) -> list[dict]:
    rows: list[dict] = []
    for seg in SEG_MAP[arm]:
        rows += _read_new(arm, seg, ".records.jsonl")
    return rows


def action_divergence(lo: list[dict], hi: list[dict]) -> dict:
    """同 deal_seed 配对 hero_actions 分叉统计（结构分叉证据）。"""
    by_seed = {r.get("deal_seed"): r for r in hi
               if r.get("deal_seed") is not None}
    n_compared = 0
    diverged_seeds = []
    for r in lo:
        other = by_seed.get(r.get("deal_seed"))
        if other is None or "hero_actions" not in r:
            continue
        n_compared += 1
        a = [(x.get("a"), x.get("size_bb")) for x in r["hero_actions"]]
        b = [(x.get("a"), x.get("size_bb"))
             for x in other.get("hero_actions", [])]
        if a != b:
            diverged_seeds.append(r.get("deal_seed"))
    return {"n_compared": n_compared, "n_diverged": len(diverged_seeds),
            "diverged_deal_seeds": diverged_seeds[:200]}


def paired_diff_ci(lo: list[dict], hi: list[dict]) -> dict:
    """同 deal_seed 配对逐手差（hi − lo）+ 95% CI（bb/100 口径）。"""
    by_seed = {r.get("deal_seed"): r for r in lo
               if r.get("deal_seed") is not None and "hero_net_bb" in r}
    diffs = []
    for r in hi:
        s = r.get("deal_seed")
        if s is None or "hero_net_bb" not in r:
            continue
        base = by_seed.get(s)
        if base is not None:
            diffs.append(r["hero_net_bb"] - base["hero_net_bb"])
    n = len(diffs)
    if n < 2:
        return {"n_paired": n, "ok": False}
    mean = sum(diffs) / n
    var = sum((x - mean) ** 2 for x in diffs) / (n - 1)
    se = math.sqrt(var / n)
    return {
        "n_paired": n,
        "mean_diff_bb_per_100": round(mean * 100, 3),
        "ci95_bb_per_100": [round((mean - 1.96 * se) * 100, 3),
                            round((mean + 1.96 * se) * 100, 3)],
        "positive_diff_hands": sum(1 for d in diffs if d > 0),
        "ok": bool(mean - 1.96 * se > 0),
    }


def bb_per_100(recs: list[dict]) -> dict | None:
    nets = [r["hero_net_bb"] for r in recs if "hero_net_bb" in r]
    n = len(nets)
    if not n:
        return None
    tot = sum(nets)
    mean = tot / n
    var = sum((x - mean) ** 2 for x in nets) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return {"bb_per_100": round(100 * tot / n, 3),
            "ci95": [round(100 * (mean - 1.96 * se), 3),
                     round(100 * (mean + 1.96 * se), 3)],
            "n_hands": n}


def old_vs_new(name: str) -> dict:
    """同 deal_seed 新-vs-旧（M4 复扫工件）逐段对照 + 布局混杂标注。"""
    out: dict = {"per_segment": {}, "layout_note": (
        "M4 工件：fish g0/g1=8w、fish g2 与 freqbot 全部=4w；新扫描全 4w。"
        "bot rng 消费 worker 内相对 hand_index → layout_matched=false 段的"
        "逐手差异不可归因（混杂），只列数不下结论。新扫描 b030/freqbot 格为"
        " 1,000 手段（k0..k5），按 deal_seed 与旧 g 段求交仍然有效。")}
    for tag in ("b005", "b030"):
        old_arm = OLD_ARM_PREFIX[name] + tag
        new_arm_full = f"r10_knob_{name}_{tag}"
        new_all: list[dict] = []
        for nseg in SEG_MAP[new_arm_full]:
            new_all += _read_new(new_arm_full, nseg, ".records.jsonl")
        new_by_seed = {r.get("deal_seed"): r for r in new_all
                       if r.get("deal_seed") is not None}
        for seg in OLD_SEG_MAP[new_arm_full]:
            old = _read(OLD_DIR, old_arm, seg, ".records.jsonl")
            key = f"{tag}.{seg}"
            if old is None:
                out["per_segment"][key] = {"old_present": False}
                continue
            common = [r for r in old
                      if r.get("deal_seed") in new_by_seed]
            div = action_divergence(
                common, [new_by_seed[r.get("deal_seed")] for r in common])
            out["per_segment"][key] = {
                "old_present": True, "layout_matched": False,
                "n_compared": div["n_compared"],
                "n_diverged_vs_m4": div["n_diverged"]}
        # 同布局旧-vs-新（可归因对）：new g*（2000 手段 × 4w，500 手/worker）
        # vs old g*（fish g2 同布局；g0/g1 旧为 8w 仍混杂）。
        for seg in LEGACY_NEW_SEGMENTS.get(new_arm_full, []):
            old = _read(OLD_DIR, old_arm, seg, ".records.jsonl")
            new = _read(NEW_FALLBACK, new_arm_full, seg,
                        ".records.jsonl")  # bench_r1 直读（g* 不进镜像）
            key = f"{tag}.{seg}.same_layout"
            if old is None or new is None:
                out["per_segment"][key] = {"present": False}
                continue
            matched = (name, seg) not in OLD_LAYOUT_8W and NEW_WORKERS == 4
            div = action_divergence(old, new)
            out["per_segment"][key] = {
                "present": True, "layout_matched": matched,
                "n_compared": div["n_compared"],
                "n_diverged_vs_m4": div["n_diverged"]}
    return out


def main() -> int:
    for _name, lo, hi in PAIRS:
        for arm in (lo, hi):
            for seg in SEG_MAP[arm]:
                _read_new(arm, seg, ".json")     # 存在性检查（缺失即退出）
    out: dict = {"pairs": {}, "verdict": {}}
    for name, lo_arm, hi_arm in PAIRS:
        lo_rec, hi_rec = load_records(lo_arm), load_records(hi_arm)
        div = action_divergence(lo_rec, hi_rec)
        div_per_seg = {}
        for seg in SEG_MAP[lo_arm]:
            lo_s = _read_new(lo_arm, seg, ".records.jsonl")
            hi_s = _read_new(hi_arm, seg, ".records.jsonl")
            d = action_divergence(lo_s, hi_s)
            div_per_seg[seg] = {"n_compared": d["n_compared"],
                                "n_diverged": d["n_diverged"]}
        pair = {
            "lo_arm": lo_arm, "hi_arm": hi_arm,
            "bb_lo": bb_per_100(lo_rec), "bb_hi": bb_per_100(hi_rec),
            "action_divergence": {k: v for k, v in div.items()
                                  if k != "diverged_deal_seeds"},
            "diverged_deal_seeds_sample": div["diverged_deal_seeds"][:20],
            "action_divergence_per_segment": div_per_seg,
            "paired_diff": paired_diff_ci(lo_rec, hi_rec),
            "old_vs_new": old_vs_new(name),
        }
        out["pairs"][name] = pair

    fb = out["pairs"]["freqbot"]
    fish = out["pairs"]["fish"]
    freqbot_diverged = fb["action_divergence"]["n_diverged"] > 0
    out["verdict"] = {
        "criterion": ("R10：行为分叉为主判据（freqbot 预算格分叉>0 = 旋钮活跃）；"
                      "fish 零分叉为结构性备注非否决项；bb/100 差为辅"),
        "knob_tunability": "active" if freqbot_diverged else "inert",
        "freqbot_divergence": (
            fb["action_divergence"]["n_diverged"],
            "of", fb["action_divergence"]["n_compared"]),
        "fish_divergence": (
            fish["action_divergence"]["n_diverged"],
            "of", fish["action_divergence"]["n_compared"]),
        "fish_zero_divergence_structural_note":
            fish["action_divergence"]["n_diverged"] == 0,
        "freqbot_bb_hi_minus_lo":
            (round(fb["bb_hi"]["bb_per_100"] - fb["bb_lo"]["bb_per_100"], 3)
             if fb["bb_hi"] and fb["bb_lo"] else None),
        "fish_bb_hi_minus_lo":
            (round(fish["bb_hi"]["bb_per_100"] - fish["bb_lo"]["bb_per_100"], 3)
             if fish["bb_hi"] and fish["bb_lo"] else None),
        "freqbot_paired_diff": fb["paired_diff"],
        "fish_paired_diff": fish["paired_diff"],
        "context": ("facing-bet 路径修复（EXE-1 _lookup_facing range-set 键"
                    "索引回归）后首次复扫；M4 复扫（修复前）为 fish 1/6000、"
                    "freqbot 0/6000 零分叉"),
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    print(json.dumps(out["verdict"], ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
