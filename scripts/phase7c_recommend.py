"""Phase 7C：Recommendation Boundary Layer（证据型推荐封装层）。

严格依据 docs/PHASE7C_STEP0_RECOMMENDATION_BOUNDARY_AUDIT.md（已通过）。

管线：Profile(history-only) → Statistical Evidence（7A decide）→ 7B Eligibility
     → Recommendation。

本层**不是** Poker Strategy Engine / EV Engine / GTO Engine：
  - 只封装"已通过 7B 验证且 7A 五关全过"的证据为推荐；
  - action 恒为 null；strength.applied 恒为 0；无任何 confidence→行动幅度公式；
  - 7B persistence/calibration 只作为特征级 evidence 元数据（Audit §E-4），
    绝不对 per-decision confidence 做二次校准（= 用 test 结果校准 = 泄漏）；
  - 特征级硬状态：three_bet/cbet 在 7B 为 insufficient_evidence → eligible 恒
    false → recommendation 恒 null（禁止单次高置信绕过 / fallback 剥削 /
    跨特征推断）；
  - 只消费 7B validated 特征：vpip / pfr / fold_to_3bet；
  - Phase 8 数据缺口只作为 meta.phase8_dependencies 依赖声明，不在本层补造。

红线（违规 = AssertionError，绝不自动降级）：
  - cluster 必须 disabled；time_decay 必须 none（7A 同款硬断言）
  - strength_arm 必须 k=0（Phase 7 主口径；k=0.5/1.0 是 Phase 8 假设槽，
    推荐层强制拒绝）
  - 只读 7B validation.json 且与冻结特征状态做一致性硬断言
  - 不导入 6G/静态画像；不读取 6C 画像文件；不新增特征/模型/EV/ROI
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_SCRIPT_DIR))

from phase7a_decision import decide  # noqa: E402

VALIDATION_JSON = _ROOT / "data/uoft/models/phase7/phase7b/validation.json"
SCHEMA_VERSION = "7.0"

# 冻结的 7B 特征状态（来自 validation.json main_judgment；一致性硬断言）
VALIDATED_FEATURES = ("vpip", "pfr", "fold_to_3bet")
INSUFFICIENT_FEATURES = ("three_bet", "cbet")

# Phase 8 数据缺口依赖声明（仅 metadata；7C 不补造任何数据）
PHASE8_DEPENDENCIES = {
    "declaration": "Phase 8 EV 验证依赖声明（仅 metadata，7C 内禁止补造数据）",
    "required_new_data": [
        "hero 视角决策记录（UofT 主数据缺失；仅 Pluribus 10,011+11 手可回放子集）",
        "equity/outcome 真值标签（主数据 96%+ 决策点无手牌）",
        "counter-action 生成器与预注册 H-* 策略（7C action 恒 null）",
        "strength 臂校准数据（k/S_max；train/validation 校准、test 判定）",
        "stack/position 分层数据（board texture 分层需新特征，属 Phase 9）",
    ],
    "absent_datasets": [
        "tournament / MTT ICM 数据（UofT 全为 cash）",
        "PKO bounty 数据",
    ],
    "candidate_engines": [
        "Pluribus 子集 replay",
        "合成 hero 环境",
        "GTO baseline（可选对照，solver 依赖）",
    ],
}

_status_cache: dict | None = None


def _7b_status() -> dict:
    """只读 7B 冻结状态 + 一致性硬断言（冻结产物被改动 → 立即失败，不静默适配）。"""
    global _status_cache
    if _status_cache is None:
        v = json.load(open(VALIDATION_JSON, encoding="utf-8"))
        mj = v["main_judgment"]
        assert mj["verdict"] == "advisor_evidence_validated", \
            "7B verdict 与冻结不符（7C 契约建立在 7B validated 之上）"
        assert sorted(mj["features_passed"]) == sorted(VALIDATED_FEATURES), \
            f"7B features_passed 与冻结清单不符: {mj['features_passed']}"
        assert sorted(mj["insufficient_features"]) == sorted(INSUFFICIENT_FEATURES), \
            f"7B insufficient_features 与冻结清单不符: {mj['insufficient_features']}"
        pf = v["per_horizon"]["200"]["per_feature"]
        _status_cache = {
            "features_passed": list(mj["features_passed"]),
            "insufficient_features": list(mj["insufficient_features"]),
            "persistence": {
                f: {"p15_feature_status": "validated",
                    "persistence_rate": pf[f]["persistence_rate"],
                    "persistence_ci95": pf[f]["persistence_ci95"],
                    "n_full_pass": pf[f]["n_full_pass"],
                    "horizon": 200}
                for f in mj["features_passed"]},
        }
    return _status_cache


def _meta(out: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "ev_claim": False,                 # 恒 false（P2 未验证）
        "advisor_mode": True,              # 恒 true（无行动应用）
        "cluster_probs": None,             # 6D 降级恒 null
        "time_sensitive_policy": "marker_only",   # 仅状态标记，不影响 eligibility
        "eligibility_source": "7B validation.json main_judgment（冻结）",
        "no_confidence_recalibration": ("7B persistence/calibration 仅作特征级 "
                                        "evidence 元数据，绝不进入 per-decision "
                                        "confidence 计算（禁止用 test 结果校准）"),
        "phase8_dependencies": PHASE8_DEPENDENCIES,
    }


def _null_result(out: dict, feature: str, state: str, reason: str) -> dict:
    return {
        "opponent_id": out["opponent_id"], "as_of": out["as_of"],
        "feature": feature,
        "recommendation": None,
        "state": state, "reason": reason,
        "meta": _meta(out),
    }


def recommend(inp: dict) -> dict:
    """7C 主入口：7A decide + 7B eligibility → Recommendation（纯函数，确定性）。

    输入 = 7A decide 的输入契约（opponent_id/as_of/history/priors/population_mu/
    spot/baseline_decision/config）。泄漏/边界/prior/黑白名单全部由 7A 硬断言。
    """
    cfg = inp.get("config") or {}
    if cfg.get("cluster") != "disabled":
        raise AssertionError("cluster 必须 disabled（6D 降级恒成立，禁止重启）")
    if cfg.get("time_decay") != "none":
        raise AssertionError("time_decay 必须 none（6E 未校准，禁止半衰期）")
    if cfg.get("strength_arm", "k=0") != "k=0":
        raise AssertionError("7C 强制 strength_arm=k=0（Phase 7 主口径；k=0.5/1.0 "
                             "是 Phase 8 假设槽，推荐层禁止携带非零强度）")

    out = decide(inp)
    feature = inp["spot"]["feature"]
    st = _7b_status()

    # ---- 特征级硬状态（冻结判定顺序）----
    if feature in st["insufficient_features"]:
        return _null_result(
            out, feature, "feature_insufficient_evidence",
            f"{feature} 在 7B 为 insufficient_evidence（冻结），eligible 恒 false；"
            f"禁止因单次高置信绕过、禁止 fallback 剥削、禁止跨特征推断")
    if out["tier"] in ("unknown", "T4"):
        return _null_result(
            out, feature, "population_only",
            "无个体证据（T4/未知玩家）：population prior 口径，不产生推荐，"
            "population prior 不构成剥削通道")

    cands = out["candidates"]
    if not cands:
        return _null_result(out, feature, "gate_failed",
                            "无候选（spot 未触发等），不产生推荐")
    cand = cands[0]
    fstat = out["profile_image"]["features"].get(feature, {})
    if fstat.get("zero_opportunity") is True:
        return _null_result(out, feature, "zero_opportunity",
                            "zero_opportunity：未观察到 ≠ 不会；不消费、不产生推荐")
    if fstat.get("consumable") is False:
        return _null_result(
            out, feature, "not_consumable",
            f"consumable=false（reliability={fstat.get('reliability')}, "
            f"samples={fstat.get('samples')}, shrinkage={fstat.get('shrinkage')}）")
    if out["exploit_status"] not in ("strong", "weak"):
        return _null_result(out, feature, "gate_failed",
                            "7A 五关未全过（exploit_status=none）："
                            "不产生推荐，绝不降级为弱推荐")
    if cand["confidence"] is None:
        return _null_result(out, feature, "gate_failed",
                            "confidence=null（gate 失败），不产生推荐")

    # ---- eligible：构造 Recommendation（无行动、无强度、无二次校准）----
    per = st["persistence"][feature]
    rec = {
        "direction": cand["direction"],
        "confidence": cand["confidence"],      # 7A 原值，绝不重校准
        "evidence": {
            "p15_feature_status": per["p15_feature_status"],
            "persistence_rate": per["persistence_rate"],   # 特征级元数据
            "persistence_ci95": per["persistence_ci95"],
            "n_full_pass": per["n_full_pass"],
            "horizon": per["horizon"],
        },
        "sample_size": fstat["samples"],
        "reliability": fstat["reliability"],
        "action": None,                        # Phase 7 无 action mapping（恒 null）
        "hypothesis_id": cand["hypothesis_id"],  # 仅注释（H-* 不产生实际 action）
        "strength": {"recommended": 0.0, "applied": 0.0},
        "eligibility_checks": {
            "p15_validated": True,
            "five_gates": True,
            "confidence_not_null": True,
            "no_time_leakage_violation": True,
        },
    }
    return {
        "opponent_id": out["opponent_id"], "as_of": out["as_of"],
        "feature": feature,
        "recommendation": rec,
        "state": "eligible", "reason": "",
        "meta": _meta(out),
    }


# ---------------- 契约自检（写 contract_check.json，不读任何真实对手数据） ----------------
def _mk_priors() -> dict:
    specs = {"vpip": 0.25, "pfr": 0.15, "three_bet": 0.05, "cbet": 0.65}
    out = {}
    for f, mu in specs.items():
        out[f] = {"a": mu * 20.0, "b": (1.0 - mu) * 20.0,
                  "mu": mu, "provenance": "hb_fit"}
    out["fold_to_3bet"] = {"a": 6.0, "b": 4.0, "mu": 0.60,
                           "provenance": "scheme_b"}
    return out


def _ts(i: int) -> str:
    return f"2009-07-05T00:00:{i:06d}"


def _base_input(feature: str, history: list, **over) -> dict:
    priors = _mk_priors()
    ctx = ({"faced_3bet": True, "position": "LP"} if feature == "fold_to_3bet"
           else {"position": "LP"})
    inp = {
        "opponent_id": "selfcheck-opp",
        "as_of": "2009-07-06T00:00:00",
        "history": history,
        "priors": priors,
        "population_mu": {f: p["mu"] for f, p in priors.items()},
        "spot": {"feature": feature, "trigger": True, "context": ctx},
        "baseline_decision": {"action": "check", "size": None},
        "config": {"p7_effect_size": 0.02, "strength_arm": "k=0",
                   "cluster": "disabled", "time_decay": "none"},
    }
    inp.update(over)
    return inp


def _ft3b_events(S: int, n: int, total: int | None = None) -> list:
    total = n if total is None else total
    return [{"ts": _ts(i), "faced_3bet": i < n,
             "folded_to_3bet": i < S} for i in range(total)]


def _full_events(feature: str, S: int, n: int) -> list:
    return [{"ts": _ts(i), feature: i < S} for i in range(n)]


def _cbet_events(S: int, n: int, total: int) -> list:
    return [{"ts": _ts(i), "cbet_opportunity": i < n,
             "cbet": i < n and i < S} for i in range(total)]


def _run_selfcheck() -> dict:
    cases = []

    def record(name, fn, expect_state=None, expect_hard_fail=False):
        try:
            r = fn()
            if expect_hard_fail:
                cases.append({"case": name, "pass": False,
                              "detail": "预期硬失败但未触发"})
                return
            if expect_state is not None:
                ok = r["state"] == expect_state
                cases.append({"case": name, "pass": ok,
                              "detail": {"state": r["state"],
                                         "recommendation_null":
                                             r["recommendation"] is None}})
            else:
                ok = r["state"] == "eligible" and r["recommendation"] is not None
                cases.append({"case": name, "pass": ok,
                              "detail": {"state": r["state"]}})
        except AssertionError:
            cases.append({"case": name, "pass": expect_hard_fail,
                          "detail": "AssertionError 硬失败（预期）"
                          if expect_hard_fail else "意外 AssertionError"})

    # eligible：fold_to_3bet 强偏差
    record("validated_feature_eligible",
           lambda: recommend(_base_input("fold_to_3bet",
                                         _ft3b_events(S=680, n=800))))
    # insufficient 特征：three_bet 即使强偏差也 null
    record("three_bet_hard_insufficient",
           lambda: recommend(_base_input("three_bet",
                                         _full_events("three_bet", S=120, n=400))),
           expect_state="feature_insufficient_evidence")
    record("cbet_hard_insufficient",
           lambda: recommend(_base_input("cbet",
                                         _cbet_events(S=120, n=200, total=800))),
           expect_state="feature_insufficient_evidence")
    # gate 失败 / 样本不足 / 零机会 / 未知 / T4
    record("gate_fail_tail",
           lambda: recommend(_base_input("fold_to_3bet",
                                         _ft3b_events(S=54, n=100))),
           expect_state="gate_failed")
    record("not_consumable_small_sample",
           lambda: recommend(_base_input("fold_to_3bet",
                                         _ft3b_events(S=8, n=10))),
           expect_state="not_consumable")
    record("zero_opportunity",
           lambda: recommend(_base_input("fold_to_3bet",
                                         _full_events("vpip", S=30, n=100))),
           expect_state="zero_opportunity")
    record("unknown_player",
           lambda: recommend(_base_input("fold_to_3bet", [])),
           expect_state="population_only")
    record("t4_player",
           lambda: recommend(_base_input("vpip", _full_events("vpip", S=3, n=5))),
           expect_state="population_only")
    # 硬失败（完整 config + 单个违规键，逐项触发对应断言）
    def bad_config(**bad):
        cfg = {"p7_effect_size": 0.02, "strength_arm": "k=0",
               "cluster": "disabled", "time_decay": "none"}
        cfg.update(bad)
        return cfg

    record("cluster_violation",
           lambda: recommend(_base_input("fold_to_3bet", _ft3b_events(S=680, n=800),
                                         **{"config": bad_config(cluster="enabled")})),
           expect_hard_fail=True)
    record("time_decay_violation",
           lambda: recommend(_base_input("fold_to_3bet", _ft3b_events(S=680, n=800),
                                         **{"config": bad_config(time_decay="x")})),
           expect_hard_fail=True)
    record("strength_arm_nonzero",
           lambda: recommend(_base_input("fold_to_3bet", _ft3b_events(S=680, n=800),
                                         **{"config": bad_config(strength_arm="k=0.5")})),
           expect_hard_fail=True)
    events = _ft3b_events(S=680, n=800)
    events[5]["ts"] = "2009-07-07T00:00:00"
    record("future_in_history",
           lambda: recommend(_base_input("fold_to_3bet", events)),
           expect_hard_fail=True)

    return {"cases": cases,
            "n_cases": len(cases),
            "all_clean": all(c["pass"] for c in cases)}


def main() -> int:
    out_dir = _ROOT / "data/uoft/models/phase7/phase7c"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = _run_selfcheck()
    (out_dir / "contract_check.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for c in res["cases"]:
        print(f"  [{('PASS' if c['pass'] else 'FAIL')}] {c['case']}: {c['detail']}")
    print(f"contract_check: n={res['n_cases']} all_clean={res['all_clean']}")
    return 0 if res["all_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
