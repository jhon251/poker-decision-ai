"""Phase 6 Step 6G：接口层（load / join / filter / validate / interface contract）。

严格限制：不新增特征、模型、GTO、EV、Agent 决策逻辑、时间衰减。
只读所有冻结产物，接口层不写回。

三个接口（契约见 docs/PHASE6G_INTERFACE_AUDIT.md + PHASE6G_INTERFACE.md）：
  get_profile(player_id, as_of) -> Profile
  predict_probe(profile, feature, n_future) -> dict
  deviation_score(profile, feature, observed_rate, observed_n) -> dict

关键语义：
  - as_of >= DATA_END → 返回静态画像；as_of < DATA_END → UnsupportedError
    （static_full_history 不伪造历史；未来 dynamic/as_of-aware 扩展语义保留：
     profile(as_of) = history where started_at < as_of，本阶段不实现）
  - time_sensitive 语义（补充契约）：unstable → time_sensitive=true，但
      · 不改变 feature value
      · 不降低 reliability
      · 不禁止 consumable
      · 不得解释为"玩家已改变打法"（只是"需关注时间漂移"标记）
    insufficient_evidence → time_sensitive=null（不可知 ≠ 不稳定）
  - 6E 时间状态 join：按 feature 名并入 time_status/time_sensitive（不改 6C）
  - cluster 已降级：meta.cluster_probs=null 恒成立，接口无 cluster 参数
  - utility_proven（6F）写入 meta.utility_proven
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from phase5b_math import beta_binomial_interval, wilson_interval  # noqa: E402

ROOT = Path(__file__).parent.parent
PROFILES = ROOT / "data/uoft/models/phase6/profiles.jsonl"
TIME_STABILITY = ROOT / "data/uoft/models/phase6/phase6e/time_stability.json"
UTILITY = ROOT / "data/uoft/models/phase6/phase6f/utility.json"
POP_PRIORS = ROOT / "data/uoft/models/phase6/population_priors.json"
INDEX_PATH = ROOT / "data/uoft/models/phase6/phase6g/profile_index.jsonl"

DATA_WINDOW = ("2009-07-01T00:00:00", "2009-07-26T10:00:35")
DATA_END = DATA_WINDOW[1]
SCHEMA_VERSION = "6.0"


class UnsupportedError(Exception):
    """静态画像不支持的操作（如 as_of 回溯）。"""


def build_profile_index() -> None:
    """player_id → 字节偏移 索引（只读 6C，写入 phase6g/ 新文件）。"""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILES, encoding="utf-8") as f, \
         open(INDEX_PATH, "w", encoding="utf-8") as w:
        off = 0
        for line in f:
            pid = json.loads(line)["player_id"]
            w.write(json.dumps([pid, off]) + "\n")
            off += len(line.encode("utf-8"))


_index_cache: dict[str, int] | None = None
_time_cache: dict | None = None
_pop_prior_cache: dict | None = None
_utility_cache: dict | None = None


def _index() -> dict[str, int]:
    global _index_cache
    if _index_cache is None:
        if not INDEX_PATH.exists():
            build_profile_index()
        _index_cache = {}
        for line in open(INDEX_PATH, encoding="utf-8"):
            pid, off = json.loads(line)
            _index_cache[pid] = off
    return _index_cache


def _time_status() -> dict:
    global _time_cache
    if _time_cache is None:
        _time_cache = json.load(open(TIME_STABILITY, encoding="utf-8"))["features"]
    return _time_cache


def _pop_priors() -> dict:
    global _pop_prior_cache
    if _pop_prior_cache is None:
        _pop_prior_cache = json.load(open(POP_PRIORS, encoding="utf-8"))
    return _pop_prior_cache


def _utility() -> dict:
    global _utility_cache
    if _utility_cache is None:
        _utility_cache = json.load(open(UTILITY, encoding="utf-8"))
    return _utility_cache


def _posterior_ab(feat: dict) -> tuple[float, float] | None:
    """从 (value, samples, shrinkage) 确定性重构后验 Beta(a',b')。

    shrinkage = (a+b)/(a+b+n)；value = (a+S)/(a+b+n) ⇒
      a' = value*n/(1-shrinkage)，b' = (1-value)*n/(1-shrinkage)
    零机会（samples=0, shrinkage=1）无法重构个体后验 → None。
    """
    n = feat.get("samples")
    sh = feat.get("shrinkage")
    v = feat.get("value")
    if not n or sh >= 1.0:
        return None
    ap = v * n / (1.0 - sh)
    bp = (1.0 - v) * n / (1.0 - sh)
    return ap, bp


def _prior_ab(feature: str, prior_source: str | None) -> tuple[float, float]:
    """population prior 的 (a,b)（用于 zero-opportunity 的 prior 区间预测）。"""
    pop = _pop_priors()
    if prior_source and prior_source != "global":
        strata = pop.get("strata", {}).get(prior_source, {})
        fit = strata.get(feature)
        if fit and not fit.get("fallback_to_global"):
            return fit["a"], fit["b"]
    return pop["global"][feature]["a"], pop["global"][feature]["b"]


def get_profile(player_id: str, as_of: str) -> dict:
    """返回 Profile（6C 行 + 接口层 join 6E 时间状态 + meta）。

    as_of 规则（补充契约冻结）：
      - as_of >= DATA_END → 静态画像
      - as_of < DATA_END → UnsupportedError（static_full_history 不伪造历史；
        未来扩展语义 profile(as_of)=history where started_at<as_of 已预留，不实现）
    """
    if as_of is None:
        raise ValueError("as_of 必填（ISO8601）")
    if as_of < DATA_END:
        raise UnsupportedError(
            f"static_full_history profile 不支持 as_of 回溯（as_of={as_of} < "
            f"data_end={DATA_END}）。未来 dynamic profile 扩展语义："
            "profile(as_of) = history where started_at < as_of。")
    idx = _index()
    off = idx.get(player_id)
    if off is None:
        # 未知玩家（含 T4 n<10 无个体行）：population_only 空画像，不抛异常
        return {"player_id": player_id, "tier": "unknown",
                "profile_type": "static_full_history",
                "population_only": True, "features": {},
                "meta": {"data_window": list(DATA_WINDOW), "as_of": DATA_END,
                         "schema_version": SCHEMA_VERSION,
                         "utility_proven": bool(_utility()["utility_proven"]),
                         "cluster_probs": None}}
    with open(PROFILES, encoding="utf-8") as f:
        f.seek(off)
        row = json.loads(f.readline())

    time_status = _time_status()
    for fname, feat in row.get("features", {}).items():
        ts = time_status.get(fname, {})
        status = ts.get("status")
        feat["time_status"] = status
        # 补充契约：unstable → time_sensitive=true（仅标记）；
        # insufficient_evidence → null（不可知 ≠ 不稳定）
        if status == "unstable":
            feat["time_sensitive"] = True
        elif status == "insufficient_evidence":
            feat["time_sensitive"] = None
        else:
            feat["time_sensitive"] = False

    row["meta"] = {
        "data_window": list(DATA_WINDOW),
        "as_of": DATA_END,
        "schema_version": SCHEMA_VERSION,
        "utility_proven": bool(_utility()["utility_proven"]),
        "cluster_probs": None,       # 6D 降级恒 null
    }
    return row


def predict_probe(profile: dict, feature: str, n_future: int) -> dict:
    """后验均值 + Beta-Binomial 预测区间（吸收未来 n_future 手抽样噪声）。

    consumable 过滤：consumable=false 不报错，返回带 consumable:false 的结果。
    zero_opportunity → population_dominated=true，预测 = prior 区间。
    """
    if n_future is None or n_future <= 0:
        raise ValueError("n_future 必须为正整数")
    if n_future > 10000:
        raise ValueError("n_future 过大（>10000 拒绝，避免病态计算）")
    feat = profile.get("features", {}).get(feature)
    if feat is None:
        raise KeyError(f"unknown feature: {feature}")
    consumable = bool(feat.get("consumable"))
    out = {"feature": feature, "consumable": consumable,
           "time_sensitive": feat.get("time_sensitive")}
    ab = _posterior_ab(feat)
    if ab is None:
        # zero_opportunity：无个体证据 → 预测 = population prior 区间
        a, b = _prior_ab(feature, feat.get("prior_source"))
        lo, hi = beta_binomial_interval(a, b, n_future)
        out.update({
            "posterior_mean": feat["value"],        # = prior mean（6C 已存）
            "predictive_interval": [lo, hi],
            "samples": 0, "shrinkage": 1.0,
            "reliability": feat.get("reliability"),
            "population_dominated": True,
            "note": "zero_opportunity：无个体证据，预测为 population prior 区间",
        })
        return out
    a, b = ab
    lo, hi = beta_binomial_interval(a, b, n_future)
    out.update({
        "posterior_mean": feat["value"],
        "predictive_interval": [lo, hi],
        "samples": feat["samples"],
        "shrinkage": feat["shrinkage"],
        "reliability": feat["reliability"],
        "population_dominated": False,
    })
    return out


def deviation_score(profile: dict, feature: str, observed_rate: float,
                    observed_n: int) -> dict:
    """观测率相对画像后验的偏差检验（Wilson 区间不重叠 → significant）。

    仅 consumable=true 的 Tier A/合格 B 特征给出 significant 判定；
    consumable=false / zero_opportunity → significant=null。
    """
    if observed_n is None or observed_n <= 0:
        raise ValueError("observed_n 必须为正整数")
    if not (0.0 <= observed_rate <= 1.0):
        raise ValueError("observed_rate 必须在 [0,1]")
    feat = profile.get("features", {}).get(feature)
    if feat is None:
        raise KeyError(f"unknown feature: {feature}")
    consumable = bool(feat.get("consumable"))
    out = {"feature": feature, "consumable": consumable,
           "observed_rate": observed_rate, "observed_n": observed_n}
    if not consumable or feat.get("zero_opportunity"):
        out.update({"z_score": None, "p_ci": None, "significant": None,
                    "direction": "none",
                    "population_dominated": bool(feat.get("zero_opportunity")),
                    "reason": ("consumable=false" if not consumable
                               else "zero_opportunity")})
        return out
    ab = _posterior_ab(feat)
    if ab is None:
        out.update({"z_score": None, "p_ci": None, "significant": None,
                    "direction": "none", "population_dominated": True})
        return out
    a, b = ab
    mean = feat["value"]
    var_post = a * b / ((a + b) ** 2 * (a + b + 1.0))
    var_obs = observed_rate * (1.0 - observed_rate) / observed_n
    z = (observed_rate - mean) / ((var_post + var_obs) ** 0.5)
    wlo, whi = wilson_interval(round(observed_rate * observed_n), observed_n)
    clo, chi = feat["ci95"]
    # 95% 不重叠判定
    if whi < clo:
        direction = "under"
        significant = True
    elif wlo > chi:
        direction = "over"
        significant = True
    else:
        direction = "none"
        significant = False
    out.update({"z_score": round(z, 4), "p_ci": [round(wlo, 4), round(whi, 4)],
                "significant": significant, "direction": direction,
                "population_dominated": False})
    return out


def validate_contract() -> dict:
    """全量扫描 6C 行，复验 consumable 硬规则（违规应恒为 0）。"""
    rules = json.load(open(ROOT / "data/uoft/models/phase6"
                               "/tier_reliability_rules.json", encoding="utf-8"))
    feat_tier = {k: v["tier"] for k, v in rules["features"].items()}
    alias = {f["name"] for f in json.load(open(
        ROOT / "data/uoft/models/phase6/feature_schema.json",
        encoding="utf-8"))["features"] if f.get("alias_of")}
    viol = {"tier_c_consumable": 0, "alias_consumable": 0,
            "consumable_not_reliable": 0, "b_not_shrink": 0,
            "zero_opp_value_zero": 0, "ci_bad": 0}
    n_rows = 0
    for line in open(PROFILES, encoding="utf-8"):
        row = json.loads(line)
        n_rows += 1
        for fname, f in row.get("features", {}).items():
            t = feat_tier.get(fname, "B")
            if t == "C" and f.get("consumable"):
                viol["tier_c_consumable"] += 1
            if fname in alias and f.get("consumable"):
                viol["alias_consumable"] += 1
            if f.get("consumable") and f.get("reliability") != "reliable":
                viol["consumable_not_reliable"] += 1
            if f.get("consumable") and t == "B" and f.get("shrinkage", 1.0) >= 0.5:
                viol["b_not_shrink"] += 1
            if f.get("zero_opportunity") and f.get("value") == 0.0:
                viol["zero_opp_value_zero"] += 1
            lo, hi = f["ci95"]
            if not (lo <= hi):
                viol["ci_bad"] += 1
    return {"n_rows": n_rows, "violations": viol,
            "all_clean": all(v == 0 for v in viol.values())}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-index", action="store_true")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.build_index:
        build_profile_index()
        print(f"index built: {INDEX_PATH} ({len(_index())} players)")
    if args.validate:
        res = validate_contract()
        out_path = ROOT / "data/uoft/models/phase6/phase6g/contract_check.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False))
