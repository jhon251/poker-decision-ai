"""Phase 7A：Exploit Advisor 决策层（纯函数，顾问模式）。

严格依据 docs/PHASE7_STEP0_EXPLOIT_ARCHITECTURE_AUDIT.md §10（Step 7A 契约）。
管线：Profile Image(history-only) → Statistical Evidence → Actionability Gate
     → Exploit Confidence → Exploit Strength → ExploitDecision。

红线（违规 = AssertionError / ValueError，绝不降级）：
  - strength_applied ≡ 0、adjustment_applied ≡ false、ev_claim ≡ false（顾问模式）
  - 决策特征白名单 = {vpip, pfr, three_bet, cbet, fold_to_3bet}（Audit §5.3.1）
  - prior 必须 train-only（provenance ∈ {"hb_fit","scheme_b"}）；任何测试玩家
    进入 prior fitting 由调用方（7B）硬断言，本层只消费已声明的 train-only 先验
  - spot 上下文只允许决策点之前的状态字段；outcome 字段 → AssertionError
  - config.cluster 必须 "disabled"；config.time_decay 必须 "none"
  - history 事件 ts ≤ as_of 且升序；混入 hu 手 → AssertionError
  - time_sensitive 仅透传标记，不改变 value/reliability/consumable/confidence
  - 不修改任何 Phase ≤6.5 冻结产物；本模块只读 tier 规则与 6E 时间状态

依赖（全部冻结，只 import 不改）：
  - phase5b_math.betainc（正则化不完全 Beta）
  - phase6b_tiers.reliability_rule（reliability 冻结阈值）
  - phase6f_common.feature_suff（5 特征充分统计口径）
  - data/uoft/models/phase6/tier_reliability_rules.json（feature 级 tier）
  - data/uoft/models/phase6/phase6e/time_stability.json（time_status join）

consumable 规则（6C 冻结，本层镜像）：Tier C/alias/outcome → false；
  Tier A/B ∧ reliability=reliable ∧ (Tier A 或 shrinkage<0.5) → true。
  5 个决策特征全部 Tier A → 实际 = reliable 且非 zero_opportunity。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from phase5b_math import betainc  # noqa: E402
from phase6b_tiers import reliability_rule  # noqa: E402
from phase6f_common import feature_suff  # noqa: E402

SCHEMA_VERSION = "7.0"
DECISION_FEATURES = ("vpip", "pfr", "three_bet", "cbet", "fold_to_3bet")

TIER_RULES_PATH = _ROOT / "data/uoft/models/phase6/tier_reliability_rules.json"
TIME_STABILITY_PATH = _ROOT / "data/uoft/models/phase6/phase6e/time_stability.json"

# ---- 预注册参数（Audit §5.8–§5.10，全部来源可追溯）----
P7_EFFECT_SIZE_DEFAULT = 0.02      # 主口径；先例 = 6E EFFECT_SIZE=0.02（漂移口径，
                                   # 对剥削的适用性是待验证假设 H3，7B 敏感性报告）
P7_EFFECT_SIZE_ALLOWED = (0.0, 0.02)   # 预注册集合，其余 → ValueError
STRENGTH_ARMS = ("k=0", "k=0.5", "k=1.0")   # 假设族 g=k·c·|m−μ|·S_max；主口径 k=0
S_MAX_PLACEHOLDER = 1.0    # 中性占位（=无放大）；Phase 8 从 {1.25,1.5}/{0.10,0.20} 校准
CONF_STRONG = 0.95         # 五关全过 ∧ c ≥ 0.95 → strong；否则 weak
CONF_FLOOR = 0.5           # c ≤ 0.5 → baseline（Audit §5.11）
TAIL_SIG = 0.975           # 单侧 95%（项目统一 CI 惯例）
T1_HANDS, T2_HANDS, T3_HANDS = 200, 50, 10   # 6C tier_of 手数口径

# 决策映射假设表（Audit §5.8；全部为 H-* 待验证假设，仅标注用途）
HYPOTHESIS_MAP = {
    ("fold_to_3bet", "over"): {"id": "H-ft3b-over", "type": "frequency_shift",
                               "description": "对手过度弃牌于 3bet：增加 3bet 诈唬、"
                                              "减少薄价值 3bet（假设，未验证）"},
    ("fold_to_3bet", "under"): {"id": "H-ft3b-under", "type": "frequency_shift",
                                "description": "对手 3bet 弃牌不足：价值 3bet 为主、"
                                               "减少诈唬（假设，未验证）"},
    ("vpip", "over"): {"id": "H-vpip-over", "type": "frequency_shift",
                       "description": "对手过度入池：扩大价值加注、减少诈唬（假设，未验证）"},
    ("vpip", "under"): {"id": "H-vpip-under", "type": "frequency_shift",
                        "description": "极紧对手：多偷盲、对其加注多弃（假设，未验证）"},
    ("pfr", "over"): {"id": "H-pfr-over", "type": "frequency_shift",
                      "description": "对手 open 过多：对其 open 更宽 3bet（假设，未验证）"},
    ("pfr", "under"): {"id": "H-pfr-under", "type": "frequency_shift",
                       "description": "对手 open 过少：更多跟注、少 3bet 诈唬（假设，未验证）"},
    ("three_bet", "over"): {"id": "H-3b-over", "type": "frequency_shift",
                            "description": "对手 3bet 过多：对其 3bet 收紧/4bet 调整"
                                           "（假设，未验证）"},
    ("three_bet", "under"): {"id": "H-3b-under", "type": "frequency_shift",
                             "description": "对手 3bet 过少：对其加注更宽（假设，未验证）"},
    ("cbet", "over"): {"id": "H-cbet-over", "type": "frequency_shift",
                       "description": "对手 cbet 过多：check-raise/跟注更宽（假设，未验证）"},
    ("cbet", "under"): {"id": "H-cbet-under", "type": "frequency_shift",
                        "description": "对手 cbet 过少：对其 check 多诈唬（假设，未验证）"},
}

# spot 上下文白名单（决策点之前的状态/情境字段）与黑名单（outcome/回应字段）
# —— Audit §8-5 / §8-10：outcome 字段进入决策输入 = 未来泄漏，硬失败。
ALLOWED_CONTEXT_KEYS = frozenset({
    "position", "table_type", "stakes_bucket", "venue", "game_mode",
    "faced_3bet", "faced_cbet", "cbet_opportunity", "rfi_opp", "saw_flop",
    "street", "pot", "bet_to_call", "effective_stack", "n_players", "bb", "sb",
})
FORBIDDEN_CONTEXT_KEYS = frozenset({
    "vpip", "pfr", "three_bet", "four_bet", "cbet", "folded_to_3bet",
    "folded_to_cbet", "won_at_showdown", "won_hand", "net_bb", "bets",
    "calls", "folds", "rfi_did", "went_to_showdown", "wtsd", "wsd",
    "cold_call", "btn_steal", "ep_open", "agg_freq",
})

_tier_cache: dict | None = None
_time_cache: dict | None = None


def _tier_of(feature: str) -> str:
    """feature 级 tier（只读冻结 6B rules；5 个决策特征均为 A）。"""
    global _tier_cache
    if _tier_cache is None:
        _tier_cache = {k: v["tier"] for k, v in json.load(
            open(TIER_RULES_PATH, encoding="utf-8"))["features"].items()}
    return _tier_cache[feature]


def _time_status() -> dict:
    """6E 特征级时间状态（只读冻结 time_stability.json）。"""
    global _time_cache
    if _time_cache is None:
        _time_cache = json.load(open(TIME_STABILITY_PATH,
                                     encoding="utf-8"))["features"]
    return _time_cache


def _time_sensitive_of(status: str | None) -> bool | None:
    """6G 补充契约映射：unstable→true；insufficient_evidence→null；其余→false。"""
    if status == "unstable":
        return True
    if status == "insufficient_evidence":
        return None
    return False


def _consumable(tier: str, reliability: str, shrinkage: float) -> bool:
    """6C 冻结 consumable 硬规则（镜像，不改冻结文件）。"""
    if tier not in ("A", "B"):
        return False
    if reliability != "reliable":
        return False
    if tier == "B" and shrinkage >= 0.5:
        return False
    return True


def _validate_config(cfg: dict) -> tuple[str, float]:
    """预注册 config 校验：cluster/time_decay 红线 = AssertionError。"""
    if cfg.get("cluster") != "disabled":
        raise AssertionError("cluster 必须保持 disabled（6D 降级恒成立，禁止重启）")
    if cfg.get("time_decay") != "none":
        raise AssertionError("time_decay 必须为 none（6E 未校准，禁止引入半衰期）")
    arm = cfg.get("strength_arm", "k=0")
    if arm not in STRENGTH_ARMS:
        raise ValueError(f"strength_arm={arm!r} 不在预注册臂 {STRENGTH_ARMS}")
    delta = cfg.get("p7_effect_size", P7_EFFECT_SIZE_DEFAULT)
    if delta not in P7_EFFECT_SIZE_ALLOWED:
        raise ValueError(f"p7_effect_size={delta!r} 不在预注册集合 "
                         f"{P7_EFFECT_SIZE_ALLOWED}")
    return arm, float(delta)


def _check_prior(feature: str, priors: dict) -> dict:
    """train-only prior 校验（Audit §8-2/§8-3）：provenance/mu 一致性硬断言。"""
    p = priors.get(feature)
    if not isinstance(p, dict):
        raise AssertionError(f"prior 缺失: {feature}（决策层必须消费 train-only "
                             f"prior：hb_fit / scheme_b）")
    prov = p.get("provenance")
    if prov not in ("hb_fit", "scheme_b"):
        raise AssertionError(f"{feature} prior provenance={prov!r} 非 train-only "
                             f"（只允许 hb_fit / scheme_b；任何全 pool 先验禁止）")
    a_raw, b_raw = p.get("a"), p.get("b")
    if (not isinstance(a_raw, (int, float)) or not isinstance(b_raw, (int, float))
            or not (a_raw > 0 and b_raw > 0)):
        raise AssertionError(f"{feature} prior 参数非法: a={a_raw!r}, b={b_raw!r}")
    a, b = float(a_raw), float(b_raw)
    mu = p.get("mu")
    if mu is None:
        mu = a / (a + b)
    if abs(float(mu) - a / (a + b)) > 1e-9:
        raise AssertionError(f"{feature} prior mu={mu} 与 a/(a+b)={a / (a + b)} "
                             f"不一致（防静默错先验）")
    return {"a": a, "b": b, "mu": float(mu), "provenance": prov}


def _suff(history: list, feature: str) -> tuple[int, int]:
    """(S, n) 充分统计（冻结 6F 口径）；cbet 套用 6.5 冻结 n_eff=max(S,n)。"""
    S, n = feature_suff(history, feature)
    if feature == "cbet" and S > n:
        n = S   # 6.5 冻结口径：S 事件必然有对应机会，n_eff 保守修正（config 记录）
    return S, n


def _normal_ci(a: float, b: float) -> tuple[float, float, float]:
    """6C 口径：Beta 后验 Normal 近似 95% CI（±1.96sd，截断 [0,1]）。"""
    mean = a / (a + b)
    var = a * b / ((a + b) ** 2 * (a + b + 1.0))
    sd = var ** 0.5
    lo = max(0.0, mean - 1.96 * sd)
    hi = min(1.0, mean + 1.96 * sd)
    return round(mean, 6), round(lo, 6), round(hi, 6)


def _posterior(prior: dict, S: int, n: int) -> dict:
    """EB 后验（冻结口径）。n=0 → zero_opportunity：值=先验均值，"未观察到≠不会"。"""
    a, b = prior["a"], prior["b"]
    if n == 0:
        mean, lo, hi = _normal_ci(a, b)
        return {"value": mean, "ci95": [lo, hi], "samples": 0,
                "shrinkage": 1.0, "ci_width": round(hi - lo, 6),
                "zero_opportunity": True, "posterior_a": a, "posterior_b": b}
    ap, bp = a + S, b + n - S
    mean, lo, hi = _normal_ci(ap, bp)
    return {"value": mean, "ci95": [lo, hi], "samples": n,
            "shrinkage": round((a + b) / (a + b + n), 6),
            "ci_width": round(hi - lo, 6), "zero_opportunity": False,
            "posterior_a": ap, "posterior_b": bp}


def _meta(arm: str, delta: float) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "ev_claim": False,                 # 恒 false（无 EV/收益主张，Audit F2）
        "advisor_mode": True,              # 恒 true（strength_applied ≡ 0）
        "cluster_probs": None,             # 6D 降级恒 null
        "time_sensitive_policy": "marker_only",     # 仅标记，不改值/置信/强度
        "strength_policy": "uncalibrated_hypothesis_family_k0",
        "decision_features": list(DECISION_FEATURES),
        "p7_effect_size": delta,
        "strength_arm": arm,
        "train_only_prior_required": True,
    }


def _population_only_output(opp: str, as_of: str, baseline: dict, n_hist: int,
                            arm: str, delta: float) -> dict:
    return {
        "opponent_id": opp, "as_of": as_of,
        "tier": "unknown" if n_hist == 0 else "T4",
        "profile_image": {"provenance": "history_only", "as_of": as_of,
                          "features": {}},
        "exploit_status": "none",
        "fallback_level": "population_prior",
        "candidates": [],
        "baseline_decision": dict(baseline),
        "adjustment_applied": False,
        "degradation": [{
            "component": "player_evidence", "path": "population_only",
            "reason": "history ring 手数 < 10（T4/未知玩家）：无个体证据，"
                      "population prior 口径（不把 population_only 伪装成"
                      "individual evidence）",
            "evidence": {"n_history_hands": n_hist}}],
        "warnings": [],
        "meta": _meta(arm, delta),
    }


def decide(inp: dict) -> dict:
    """Exploit Advisor 决策层主入口（纯函数，确定性）。

    输入契约见 Audit §10.2，输出契约见 §10.3。任何泄漏/红线违规 → 硬失败。
    """
    # ---- 基础校验 ----
    opp = inp.get("opponent_id")
    if not isinstance(opp, str) or not opp:
        raise ValueError("opponent_id 必填（非空 str）")
    as_of = inp.get("as_of")
    if not isinstance(as_of, str) or not as_of:
        raise ValueError("as_of 必填（ISO8601 str）")
    history = inp.get("history")
    if not isinstance(history, list):
        raise ValueError("history 必填（list of ring 事件行）")
    priors = inp.get("priors") or {}
    pop_mu = inp.get("population_mu") or {}
    spot = inp.get("spot") or {}
    baseline = inp.get("baseline_decision")
    if not isinstance(baseline, dict):
        raise ValueError("baseline_decision 必填（dict，layer 永不修改，仅透传）")
    arm, delta = _validate_config(inp.get("config") or {})
    k = {"k=0": 0.0, "k=0.5": 0.5, "k=1.0": 1.0}[arm]
    warnings: list = []

    # ---- 泄漏硬断言：history 边界（Audit §8-4/§8-12）----
    prev_ts: str | None = None
    for ev in history:
        if not isinstance(ev, dict):
            raise AssertionError("history 元素必须为 dict")
        ts = ev.get("ts")
        if not ts:
            raise AssertionError("history 事件缺 ts（时间边界不可校验）")
        if ts > as_of:
            raise AssertionError(f"history 事件 ts={ts} 晚于 as_of={as_of} "
                                 f"（future 进入 history，泄漏）")
        if prev_ts is not None and prev_ts > ts:
            raise AssertionError("history 未按 ts 升序")
        prev_ts = ts
        if ev.get("table_type") == "hu":
            raise AssertionError("ring 口径：history 混入 hu 手")

    # ---- train-only prior provenance（全部 5 特征，Audit §8-2/§8-3）----
    prior_ab = {f: _check_prior(f, priors) for f in DECISION_FEATURES}
    for f in DECISION_FEATURES:
        pm = pop_mu.get(f)
        if pm is None:
            raise AssertionError(f"population_mu 缺失: {f}")
        if abs(float(pm) - prior_ab[f]["mu"]) > 1e-9:
            raise AssertionError(f"{f} population_mu={pm} 与 train-only prior "
                                 f"mu={prior_ab[f]['mu']} 不一致")

    # ---- spot 校验（白名单/黑名单，Audit §8-5/§8-10）----
    feat = spot.get("feature")
    if feat not in DECISION_FEATURES:
        raise ValueError(f"feature={feat!r} 不在 Phase 7 决策特征白名单 "
                         f"{list(DECISION_FEATURES)}（Audit §5.3.1：rfi_ep/rfi_lp/"
                         f"agg_freq/cold_call/fold_to_cbet/wtsd/wsd 等未通过独立"
                         f"验证或先验不可用；position_sensitivity 无直接行动映射；"
                         f"Tier C/alias/outcome 恒禁止）")
    trigger = bool(spot.get("trigger"))
    context = spot.get("context") or {}
    if not isinstance(context, dict):
        raise ValueError("spot.context 必填 dict（决策点之前的状态字段白名单）")
    for key in context:
        if key in FORBIDDEN_CONTEXT_KEYS:
            raise AssertionError(f"spot.context 含 outcome 字段 {key!r}："
                                 f"outcome 进入决策输入 = 未来泄漏（Audit §8-5）")
        if key not in ALLOWED_CONTEXT_KEYS:
            raise AssertionError(f"spot.context 含白名单外字段 {key!r}")

    n_hist = len(history)
    if n_hist < T3_HANDS:
        return _population_only_output(opp, as_of, baseline, n_hist, arm, delta)
    tier = "T1" if n_hist >= T1_HANDS else ("T2" if n_hist >= T2_HANDS else "T3")

    # ---- L0 Profile Image（history-only，6G Profile 同 shape）----
    time_status = _time_status()
    profile_features = {}
    for f in DECISION_FEATURES:
        S, n = _suff(history, f)
        post = _posterior(prior_ab[f], S, n)
        ft = _tier_of(f)
        rel = reliability_rule(post["samples"], post["shrinkage"],
                               post["ci_width"])
        cons = _consumable(ft, rel, post["shrinkage"])
        ts = time_status.get(f, {})
        status = ts.get("status")
        profile_features[f] = {
            "value": post["value"], "ci95": post["ci95"],
            "samples": post["samples"], "shrinkage": post["shrinkage"],
            "reliability": rel, "tier": ft, "consumable": cons,
            "zero_opportunity": post["zero_opportunity"],
            "time_status": status,
            "time_sensitive": _time_sensitive_of(status),
        }
    post = None
    if not profile_features[feat]["zero_opportunity"]:
        S, n = _suff(history, feat)
        post = _posterior(prior_ab[feat], S, n)

    degradation: list = []
    fstat = profile_features[feat]
    mu = prior_ab[feat]["mu"]

    # ---- L1 Statistical Evidence（横向偏差：个体后验 vs population 先验）----
    if post is not None:
        ap, bp = post["posterior_a"], post["posterior_b"]
        m = fstat["value"]
        if m > mu:
            direction, tail_sig = "over", 1.0 - betainc(ap, bp, mu)
        elif m < mu:
            direction, tail_sig = "under", betainc(ap, bp, mu)
        else:
            direction, tail_sig = "none", 0.5
    else:
        direction, tail_sig = "none", None

    # ---- L2 Actionability Gate（五关合取，Audit §5.9）+ L3/L4 ----
    gate = {
        "consumable": fstat["consumable"],
        "zero_opportunity": not fstat["zero_opportunity"],
        "direction_tail": (tail_sig is not None and tail_sig >= TAIL_SIG),
        "effect_magnitude": (post is not None
                             and abs(fstat["value"] - mu) >= delta),
        "spot_trigger": trigger,
        "hypothesis_map": (trigger and direction in ("over", "under")
                           and (feat, direction) in HYPOTHESIS_MAP),
    }
    confidence: float | None = None
    hyp = HYPOTHESIS_MAP.get((feat, direction)) if trigger else None
    if post is not None and gate["direction_tail"] and direction in ("over", "under"):
        if direction == "over":
            confidence = 1.0 - betainc(post["posterior_a"], post["posterior_b"],
                                       mu + delta)
        else:
            confidence = betainc(post["posterior_a"], post["posterior_b"],
                                 mu - delta)
    gate["confidence_floor"] = (confidence is not None and confidence > CONF_FLOOR)

    failed = [p for p, ok in gate.items() if not ok]
    _fmt = lambda v: f"{v:.4f}" if v is not None else "N/A"
    reasons = {
        "consumable": f"consumable=false（reliability={fstat['reliability']}, "
                      f"tier={fstat['tier']}, shrinkage={fstat['shrinkage']}，"
                      f"samples={fstat['samples']}）",
        "zero_opportunity": "zero_opportunity：无个体证据，横向偏差不计算"
                            "（未观察到 ≠ 不会），population prior 口径",
        "direction_tail": f"方向尾概率 {_fmt(tail_sig)} < {TAIL_SIG}（个体后验未显著"
                          f"偏离 population 先验 μ={mu:.4f}）",
        "effect_magnitude": f"|后验均值−先验均值|={_fmt(abs(fstat['value'] - mu) if post is not None else None)} < "
                            f"p7_effect_size={delta:.2f}（统计偏差未达实践幅度）",
        "spot_trigger": "spot.trigger=false：该决策点不存在对应情境，不产生建议",
        "hypothesis_map": f"无 (feature={feat}, direction={direction}) 的预注册"
                          f"决策映射假设",
        "confidence_floor": f"confidence={_fmt(confidence)} ≤ {CONF_FLOOR}",
    }
    for p in failed:
        degradation.append({
            "component": "actionability_gate", "path": p,
            "reason": reasons[p],
            "evidence": {"feature": feat, "direction": direction,
                         "samples": fstat["samples"],
                         "deviation_magnitude": round(fstat["value"] - mu, 6)
                         if post is not None else None,
                         "tail_probability": (round(tail_sig, 6)
                                              if tail_sig is not None else None),
                         "confidence": (round(confidence, 6)
                                        if confidence is not None else None)}})

    # ---- L4 Exploit Strength（假设族；strength_applied 恒 0）----
    status = "none"
    fallback = ("population_prior" if fstat["zero_opportunity"] else "baseline")
    candidate: dict | None = None
    if trigger:
        candidate = {
            "feature": feat, "spot": f"{feat}_spot",
            "direction": direction,
            "deviation_magnitude": (round(fstat["value"] - mu, 6)
                                    if post is not None else None),
            "tail_probability": (round(tail_sig, 6)
                                 if tail_sig is not None else None),
            "effect_gate_pass": gate["effect_magnitude"],
            "confidence": (round(confidence, 6)
                           if (not failed and confidence is not None) else None),
            "hypothesis_id": hyp["id"] if hyp else None,
            "recommended_adjustment": {"type": hyp["type"], "value": None}
            if hyp else {"type": None, "value": None},
            "strength_recommended": 0.0,
            "strength_applied": 0.0,      # 硬不变量：顾问模式恒 0
        }
        if not failed and hyp is not None and confidence is not None:
            # strength 假设族 g = k·c·|m−μ|·S_max（S_max=1.0 中性占位；Phase 8 校准）
            s_rec = k * confidence * abs(fstat["value"] - mu) * S_MAX_PLACEHOLDER
            candidate["strength_recommended"] = round(s_rec, 6)
            status = "strong" if confidence >= CONF_STRONG else "weak"
            fallback = None
            if arm != "k=0":
                warnings.append({
                    "code": "STRENGTH_HYPOTHESIS_ARM",
                    "message": f"strength_arm={arm} 仅产生推荐强度假设值"
                               f"（strength_recommended={s_rec:.4f}），"
                               f"不是真实下注频率调整；strength_applied 恒 0，"
                               f"Phase 8 EV 级校准前不得应用"})
    if delta == 0.0:
        warnings.append({"code": "EFFECT_SIZE_SENSITIVITY",
                         "message": "p7_effect_size=0.0 为敏感性口径；"
                                    "主口径 = 0.02"})

    return {
        "opponent_id": opp, "as_of": as_of, "tier": tier,
        "profile_image": {"provenance": "history_only", "as_of": as_of,
                          "features": profile_features},
        "exploit_status": status,
        "fallback_level": fallback,
        "candidates": [candidate] if candidate is not None else [],
        "baseline_decision": dict(baseline),   # 拷贝透传，绝不修改
        "adjustment_applied": False,
        "degradation": degradation,
        "warnings": warnings,
        "meta": _meta(arm, delta),
    }


def _demo() -> None:
    """合成自检（顾问模式输出示例）。不读真实数据，不改任何文件。"""
    priors = {
        "vpip": {"a": 5.0, "b": 15.0, "mu": 0.25, "provenance": "hb_fit"},
        "pfr": {"a": 3.0, "b": 17.0, "mu": 0.15, "provenance": "hb_fit"},
        "three_bet": {"a": 1.0, "b": 19.0, "mu": 0.05, "provenance": "hb_fit"},
        "cbet": {"a": 13.0, "b": 7.0, "mu": 0.65, "provenance": "hb_fit"},
        "fold_to_3bet": {"a": 6.0, "b": 4.0, "mu": 0.60, "provenance": "scheme_b"},
    }
    pop_mu = {f: p["mu"] for f, p in priors.items()}
    history = []
    for i in range(800):
        history.append({"ts": f"2009-07-05T00:{i // 60:02d}:{i % 60:02d}",
                        "faced_3bet": True, "folded_to_3bet": i < 680})
    inp = {
        "opponent_id": "synthetic-demo",
        "as_of": "2009-07-06T00:00:00",
        "history": history,
        "priors": priors, "population_mu": pop_mu,
        "spot": {"feature": "fold_to_3bet", "trigger": True,
                 "context": {"faced_3bet": True, "position": "LP"}},
        "baseline_decision": {"action": "check", "size": None},
        "config": {"p7_effect_size": 0.02, "strength_arm": "k=0",
                   "cluster": "disabled", "time_decay": "none"},
    }
    out = decide(inp)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _demo()
