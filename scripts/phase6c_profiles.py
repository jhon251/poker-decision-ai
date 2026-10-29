"""Phase 6 Step 6C：分层 Individual Profiling（T1-T4）。

只消费 6A 已验证的 sufficient statistics + 6B 先验/tier/rules，不重新解释事件。

分层（按 n_hands 总手数）:
  T1: n>=200   → 全 17 维个体画像
  T2: 50<=n<200 → 核心 6 维（vpip/pfr/three_bet/cbet/position_sensitivity/fold_to_3bet）
  T3: 10<=n<50 → 仅 3 维（vpip/pfr/three_bet），强收缩标注
  T4: n<10     → 无个体行（仅 population prior，Profile 接口按需服务）

每特征（独立 Beta-Binomial EB 后验）:
  - 先验选择: venue×stakes_bucket 分层先验；不存在/fallback → global（显式记录）
  - posterior: Beta(a+S, b+n-S) → mean / 95% CI（beta 分位数）
  - shrinkage = (a+b)/(a+b+n)
  - reliability: samples + shrinkage + CI 宽度共同决定（6B.3 规则）
  - consumable 硬规则:
      Tier C → False；alias（ep_open）→ False；
      Tier A/B 且 reliability=reliable 且 (Tier A 或 shrinkage<0.5) → True；否则 False
  - 边界显式处理:
      samples=0 → value=prior mean, CI=prior 区间, shrinkage=1, reliability=unreliable,
                  flag zero_opportunity=true（不把"未观察到"解释成"不会该行为"）
      数值保护 → 先验保护标志透传进画像 degradation 字段

输出 data/uoft/models/phase6/profiles.jsonl（每玩家一行）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from scripts.phase5b_common import ensure_dir  # noqa: E402
from scripts.phase6b_tiers import reliability_rule  # noqa: E402

T2_FEATURES = ("vpip", "pfr", "three_bet", "cbet", "position_sensitivity",
               "fold_to_3bet")
T3_FEATURES = ("vpip", "pfr", "three_bet")
PS_COMPONENTS = ("rfi_ep", "rfi_lp")


def tier_of(n_hands: int) -> str:
    if n_hands >= 200:
        return "T1"
    if n_hands >= 50:
        return "T2"
    if n_hands >= 10:
        return "T3"
    return "T4"


def features_for(tier: str, schema: list) -> list:
    names = [f["name"] for f in schema]
    if tier == "T1":
        return schema
    if tier == "T2":
        return [f for f in schema if f["name"] in T2_FEATURES]
    return [f for f in schema if f["name"] in T3_FEATURES]


def pick_prior(feat_name: str, venue: str | None, stakes_bucket: str,
               priors: dict) -> tuple[dict, str, bool]:
    """返回 (prior_fit, prior_source, fallback)。"""
    key = f"{venue}|{stakes_bucket}"
    strata = priors.get("strata", {}).get(key)
    if strata and feat_name in strata and not strata[feat_name].get(
            "fallback_to_global"):
        return strata[feat_name], key, False
    if strata and feat_name in strata:
        return strata[feat_name], key, True      # 显式回退
    g = priors["global"][feat_name]
    return g, "global", True


def _beta_post_ci(a: float, b: float) -> tuple[float, float, float]:
    """Beta(a,b) 后验的 Normal 近似 95% CI（±1.96·sd，截断 [0,1]）。

    注：与 Phase 5B Task A 的精确分位数口径不同——本层为 24 万玩家 × 17 维
    的批量画像，Normal 近似可解释且速度快；精确分位数口径保留在 5B 验证层。
    """
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1))
    sd = var ** 0.5
    lo = max(0.0, mean - 1.96 * sd)
    hi = min(1.0, mean + 1.96 * sd)
    return round(mean, 6), round(lo, 6), round(hi, 6)


def posterior_feature(S: int, n: int, prior: dict) -> dict:
    a, b = prior["a"], prior["b"]
    if n == 0:
        # 零机会：后验 = 先验；显式标记，不解释为"玩家不会该行为"
        mean, lo, hi = _beta_post_ci(a, b)
        return {
            "value": mean, "ci95": [lo, hi],
            "samples": 0, "shrinkage": 1.0,
            "zero_opportunity": True,
            "posterior_a": round(a, 4), "posterior_b": round(b, 4),
            "ci_width": round(hi - lo, 6),
        }
    ap, bp = a + S, b + n - S
    mean, lo, hi = _beta_post_ci(ap, bp)
    return {
        "value": mean,
        "ci95": [lo, hi],
        "samples": n,
        "shrinkage": round((a + b) / (a + b + n), 6),
        "zero_opportunity": False,
        "posterior_a": round(ap, 4), "posterior_b": round(bp, 4),
        "ci_width": round(hi - lo, 6),
    }


def consumable_for(feat: dict, tier: str, reliability: str, shrinkage: float
                   ) -> bool:
    """consumable 硬规则。alias/Tier C 永远 False。"""
    if feat.get("alias_of") or feat.get("category") in ("outcome", "alias"):
        return False
    if tier not in ("A", "B"):
        return False
    if reliability != "reliable":
        return False
    if tier == "B" and shrinkage >= 0.5:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="data/uoft/stats/full_pool_suff_stats.jsonl")
    ap.add_argument("--priors", default="data/uoft/models/phase6/population_priors.json")
    ap.add_argument("--rules", default="data/uoft/models/phase6/tier_reliability_rules.json")
    ap.add_argument("--schema", default="data/uoft/models/phase6/feature_schema.json")
    ap.add_argument("--out", default="data/uoft/models/phase6")
    args = ap.parse_args()

    schema = json.load(open(args.schema, encoding="utf-8"))["features"]
    priors = json.load(open(args.priors, encoding="utf-8"))
    rules = json.load(open(args.rules, encoding="utf-8"))
    feat_tier = {name: info["tier"] for name, info in rules["features"].items()}
    by_name = {f["name"]: f for f in schema}

    out_dir = ensure_dir(args.out)
    out_path = out_dir / "profiles.jsonl"

    n_by_tier = {"T1": 0, "T2": 0, "T3": 0, "T4": 0}
    n_zero_opp = 0
    n_fallback = 0
    n_protected = 0
    consumable_true = 0
    consumable_false = 0
    reli_dist = {}

    with open(out_path, "w", encoding="utf-8") as w:
        for line in open(args.stats, encoding="utf-8"):
            rec = json.loads(line)
            tier = tier_of(rec["n_hands"])
            n_by_tier[tier] += 1
            if tier == "T4":
                continue
            feats = features_for(tier, schema)
            profile = {
                "player_id": rec["player_id"], "tier": tier,
                "n_hands": rec["n_hands"],
                "n_ring_hands": rec["n_ring_hands"],
                "n_hu_hands": rec["n_hu_hands"],
                "venue": rec["venue"], "stakes_bucket": rec["stakes_bucket"],
                "first_seen": rec["first_seen"], "last_seen": rec["last_seen"],
                "profile_type": "static_full_history",
                "ci_method": "normal_approx",
                "features": {}, "degradation": [],
            }
            # 先计算全部基础特征后验（ps 依赖 rfi_ep/rfi_lp）
            comp = {}
            for feat in feats:
                name = feat["name"]
                sk = feat.get("suff_key")
                if sk is None:                     # position_sensitivity 派生
                    continue
                S, n = rec["features"][sk]
                prior, src, fb = pick_prior(name, rec["venue"],
                                            rec["stakes_bucket"], priors)
                post = posterior_feature(S, n, prior)
                if fb:
                    n_fallback += 1
                if prior.get("protection_triggered"):
                    n_protected += 1
                    profile["degradation"].append({
                        "component": name, "path": "prior_numerical_protection",
                        "reason": prior.get("protection_reason"),
                    })
                ft = feat_tier.get(name, "B")
                rel = reliability_rule(post["samples"], post["shrinkage"],
                                       post["ci_width"])
                reli_dist[rel] = reli_dist.get(rel, 0) + 1
                cons = consumable_for(feat, ft, rel, post["shrinkage"])
                if cons:
                    consumable_true += 1
                else:
                    consumable_false += 1
                if post["zero_opportunity"]:
                    n_zero_opp += 1
                comp[name] = {
                    "value": post["value"], "ci95": post["ci95"],
                    "samples": post["samples"],
                    "shrinkage": post["shrinkage"],
                    "reliability": rel, "tier": ft, "consumable": cons,
                    "zero_opportunity": post["zero_opportunity"],
                    "prior_source": src,
                }
            # position_sensitivity：rfi_lp 后验 − rfi_ep 后验（若二者在本层）
            if "position_sensitivity" in [f["name"] for f in feats]:
                ep = comp.get("rfi_ep")
                lp = comp.get("rfi_lp")
                # T2 层需要临时计算 rfi 分量（不在 T2_FEATURES）
                for cname in PS_COMPONENTS:
                    if cname not in comp:
                        feat = by_name[cname]
                        S, n = rec["features"][feat["suff_key"]]
                        prior, src, fb = pick_prior(
                            cname, rec["venue"], rec["stakes_bucket"], priors)
                        if fb:
                            n_fallback += 1      # T2 ps 分量 fallback 计入 summary
                        post = posterior_feature(S, n, prior)
                        ft = feat_tier.get(cname, "B")
                        rel = reliability_rule(post["samples"], post["shrinkage"],
                                               post["ci_width"])
                        comp[cname] = {
                            "value": post["value"], "ci95": post["ci95"],
                            "samples": post["samples"],
                            "shrinkage": post["shrinkage"],
                            "reliability": rel, "tier": ft,
                            "consumable": consumable_for(
                                feat, ft, rel, post["shrinkage"]),
                            "zero_opportunity": post["zero_opportunity"],
                            "prior_source": src,
                        }
                if ep is None:
                    ep = comp["rfi_ep"]
                if lp is None:
                    lp = comp["rfi_lp"]
                ps_value = lp["value"] - ep["value"]
                ps_lo = lp["ci95"][0] - ep["ci95"][1]
                ps_hi = lp["ci95"][1] - ep["ci95"][0]
                ps_samples = min(ep["samples"], lp["samples"])
                ps_shrink = max(ep["shrinkage"], lp["shrinkage"])
                ps_width = ps_hi - ps_lo
                ps_rel = reliability_rule(ps_samples, ps_shrink, ps_width)
                ft = feat_tier["position_sensitivity"]
                comp["position_sensitivity"] = {
                    "value": round(ps_value, 6),
                    "ci95": [round(ps_lo, 6), round(ps_hi, 6)],
                    "samples": ps_samples,
                    "shrinkage": ps_shrink,
                    "reliability": ps_rel, "tier": ft,
                    "consumable": consumable_for(
                        by_name["position_sensitivity"], ft, ps_rel, ps_shrink),
                    "zero_opportunity": (ep["zero_opportunity"]
                                         or lp["zero_opportunity"]),
                    "prior_source": "derived",
                }
            profile["features"] = comp
            w.write(json.dumps(profile, ensure_ascii=False) + "\n")

    summary = {
        "n_by_tier": n_by_tier,
        "t4_note": "T4 无个体行，仅 population prior（接口按需服务）",
        "n_zero_opportunity_feature_records": n_zero_opp,
        "n_prior_fallback_records": n_fallback,
        "n_numerical_protection_records": n_protected,
        "consumable": {"true": consumable_true, "false": consumable_false},
        "reliability_dist": reli_dist,
        "ci_method": "normal_approx(±1.96sd)",
    }
    (out_dir / "profiles_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
