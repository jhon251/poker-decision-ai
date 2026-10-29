"""Phase 6 Step 6B.3：Tier A/B/C 与 reliability 自动判定（代码规则，不人工标记）。

Tier 规则顺序（确定性，纯证据驱动）:
  1. category == outcome          → C   （won_hand）
  2. MVP core (decision=core)     → A   （vpip/pfr/three_bet/position_sensitivity）
  3. MVP exploratory 全部门通过    → A   （cbet: icc_pass & r_pass & ta_coverage_pass
       (icc_pass & r_pass & TaskA 覆盖 & TaskB HB 优于 naive)     & tb_hb_better）
  4. 全局成功率 < MIN_RATE        → C   （four_bet: 0.0004）
  5. 每玩家成功数中位 < 1         → C   （防御性：零事件主导）
  6. alias_of                     → 跟随目标 tier，consumable=false（ep_open→rfi_ep）
  7. conditional 且机会中位 < 15  → C   （btn_steal）
  8. 机会中位 >= 50 且 icc1_k >= 0.95 → A
  9. 其余                         → B

reliability（samples + shrinkage + CI 宽度共同决定）:
  reliable   = samples >= 30 AND shrinkage <= 0.5 AND ci_width <= 0.30
  marginal   = samples >= 15 AND shrinkage <= 0.7
  unreliable = 其余

输出 data/uoft/models/phase6/tier_reliability_rules.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from scripts.phase5b_common import ensure_dir  # noqa: E402

MIN_RATE = 0.002
CONDITIONAL_MIN_MEDIAN_OPP = 15
A_MIN_MEDIAN_OPP = 50
A_MIN_ICC = 0.95

REL_SAMPLES = 30
REL_SHRINK = 0.5
REL_CI_WIDTH = 0.30
MAR_SAMPLES = 15
MAR_SHRINK = 0.7

_TIER_CACHE: dict[str, str] = {}


def load_schema() -> list:
    return json.load(open("data/uoft/models/phase6/feature_schema.json",
                          encoding="utf-8"))["features"]


def load_mvp_evidence() -> tuple[set, set]:
    """返回 (mvp_core, mvp_exploratory_all_pass)。"""
    sel = json.load(open("data/uoft/models/mvp/probe_selection.json",
                         encoding="utf-8"))["probes"]
    ta = json.load(open("data/uoft/models/mvp/task_a.json",
                        encoding="utf-8"))["probes"]
    tb = json.load(open("data/uoft/models/mvp/task_b.json",
                        encoding="utf-8"))["probes"]
    mvp_core = {k for k, v in sel.items() if v.get("decision") == "core"}
    all_pass = set()
    for k, v in sel.items():
        if v.get("decision") != "exploratory":
            continue
        gates = [v.get("icc_pass"), v.get("r_pass"),
                 ta.get(k, {}).get("coverage_pass"),
                 tb.get(k, {}).get("hb_better_than_naive")]
        if all(g is True for g in gates):
            all_pass.add(k)
    return mvp_core, all_pass


MIN_RELIABLE_HANDS = 50   # tier 证据只在 n_hands>=50 玩家上计算（特征级属性，防长尾稀释）


def opp_stats(stats_path: str) -> dict:
    """每特征的全量机会分布（median_opp/median_success/全局率/零机会玩家）。

    只在 n_hands >= MIN_RELIABLE_HANDS 的玩家上计算——tier 是特征级属性，
    用可靠子集证据判定，避免 n<50 长尾稀释 median_opp。
    """
    import collections
    per_n = collections.defaultdict(list)
    per_s = collections.defaultdict(list)
    tot_s = collections.defaultdict(int)
    tot_n = collections.defaultdict(int)
    for line in open(stats_path, encoding="utf-8"):
        rec = json.loads(line)
        if rec["n_hands"] < MIN_RELIABLE_HANDS:
            continue
        for k, (S, n) in rec["features"].items():
            per_n[k].append(n)
            per_s[k].append(S)
            tot_s[k] += S
            tot_n[k] += n
    out = {}
    for k in per_n:
        ns = sorted(per_n[k])
        ss = sorted(per_s[k])
        out[k] = {
            "median_opp": ns[len(ns) // 2],
            "p10_opp": ns[len(ns) // 10],
            "median_success": ss[len(ss) // 2],
            "rate_global": tot_s[k] / tot_n[k] if tot_n[k] else 0.0,
            "zero_players": sum(1 for v in ns if v == 0),
            "n_players": len(ns),
        }
    return out


def compute_icc1k(stats_path: str) -> dict:
    from phase5b_step1_icc import icc_from_stats
    import collections
    records = collections.defaultdict(list)
    for line in open(stats_path, encoding="utf-8"):
        rec = json.loads(line)
        if rec["n_hands"] < MIN_RELIABLE_HANDS:
            continue
        for k, (S, n) in rec["features"].items():
            if n > 0:
                records[k].append((n, S))
    out = {}
    for k, recs in records.items():
        r = icc_from_stats(recs)
        out[k] = {"icc1": r.get("icc1"), "icc1_k": r.get("icc1_k"),
                  "n_players": r.get("k")}
    return out


def tier_rule(feat: dict, ev: dict, icc: dict, mvp_core: set,
              mvp_all_pass: set) -> tuple[str, str]:
    name = feat["name"]
    if feat.get("category") == "outcome":
        return "C", "category=outcome"
    if name in mvp_core:
        return "A", "mvp_core_validated"
    if name in mvp_all_pass:
        return "A", "mvp_exploratory_all_gates"
    if ev.get("rate_global", 1.0) < MIN_RATE:
        return "C", f"rate_global<{MIN_RATE}"
    if feat.get("alias_of"):
        return _tier_of(feat["alias_of"]), "alias_of"
    if feat.get("conditional") and ev.get("median_opp", 0) < CONDITIONAL_MIN_MEDIAN_OPP:
        return "C", f"conditional_median_opp<{CONDITIONAL_MIN_MEDIAN_OPP}"
    if (ev.get("median_opp", 0) >= A_MIN_MEDIAN_OPP
            and (icc.get(name, {}).get("icc1_k") or 0) >= A_MIN_ICC):
        return "A", f"median_opp>={A_MIN_MEDIAN_OPP} & icc1k>={A_MIN_ICC}"
    return "B", "default"


def _tier_of(name: str) -> str:
    return _TIER_CACHE.get(name, "B")


def reliability_rule(samples: int, shrinkage: float, ci_width: float) -> str:
    if samples >= REL_SAMPLES and shrinkage <= REL_SHRINK and ci_width <= REL_CI_WIDTH:
        return "reliable"
    if samples >= MAR_SAMPLES and shrinkage <= MAR_SHRINK:
        return "marginal"
    return "unreliable"


def main() -> int:
    stats_path = "data/uoft/stats/full_pool_suff_stats.jsonl"
    schema = load_schema()
    mvp_core, mvp_all_pass = load_mvp_evidence()
    ev = opp_stats(stats_path)
    icc = compute_icc1k(stats_path)

    tiers = {}
    for feat in schema:
        name = feat["name"]
        tier, path = tier_rule(feat, ev.get(name, {}), icc, mvp_core, mvp_all_pass)
        _TIER_CACHE[name] = tier
        tiers[name] = {
            "tier": tier, "rule_path": path,
            "evidence": ev.get(name, {}),
            "icc": icc.get(name, {}),
            "consumable": tier in ("A", "B"),
        }
    # 第二遍：alias 特征跟随目标（ep_open → rfi_ep 的 tier），consumable=false
    for feat in schema:
        if feat.get("alias_of"):
            name = feat["name"]
            target = _TIER_CACHE[feat["alias_of"]]
            tiers[name]["tier"] = target
            tiers[name]["rule_path"] = f"alias_of({feat['alias_of']})"
            tiers[name]["consumable"] = False

    out = {
        "reliability_thresholds": {
            "reliable": {"samples": REL_SAMPLES, "shrinkage": REL_SHRINK,
                         "ci_width": REL_CI_WIDTH},
            "marginal": {"samples": MAR_SAMPLES, "shrinkage": MAR_SHRINK},
            "note": "reliability 由 samples + shrinkage + CI 宽度共同决定",
        },
        "tier_rules": {
            "min_rate": MIN_RATE,
            "conditional_min_median_opp": CONDITIONAL_MIN_MEDIAN_OPP,
            "a_min_median_opp": A_MIN_MEDIAN_OPP, "a_min_icc": A_MIN_ICC,
            "mvp_core": sorted(mvp_core),
            "mvp_exploratory_all_pass": sorted(mvp_all_pass),
        },
        "features": tiers,
    }
    ensure_dir("data/uoft/models/phase6")
    (Path("data/uoft/models/phase6") / "tier_reliability_rules.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    for name in sorted(tiers):
        t = tiers[name]
        print(f"  {name:20s} -> {t['tier']} ({t['rule_path']}) "
              f"consumable={t['consumable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
