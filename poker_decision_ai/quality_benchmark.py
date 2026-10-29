"""R1 — Poker AI 策略质量验证基线（冻结 / 可重复 / 可审计）。

性质：**纯验证工具**（domain-level；不是 Agent）。本模块：
  - 不新增策略能力、不改任何策略数学（`FROZEN_STRATEGY_FILES` SHA256 断言）；
  - 不为 agreement/Brier/regret 调参；不扩大/收缩 UNKNOWN 语义；
  - 不伪造 MTT/PKO 数据；不把 synthetic/fixture 结果标成 REAL；
  - 不把 Solver 作为必需条件；不宣称任何盈利能力。

四层分离（本模块输出必须保持该分离）：
  工程完成 ≠ 策略能力完成 ≠ 策略质量得到验证 ≠ 真实世界盈利验证。
  本模块只服务于第三层（策略质量验证），且对其结论给出证据等级与置信度。

三模式可验证程度（R1 骨架结论）：
  CASH  —— decision-grade 真实语料可用（UofT CPRG normalized + Pluribus
           human seats）→ 质量**可验证**（fold 校准 / 决策对照 / legality）。
  MTT   —— 契约允许（decision_grade.mode_validation_scope）但**无真实锦标赛
           语料** → `NOT_VALIDATABLE_WITH_CURRENT_DATA`；仅机制/隔离/确定性/边界。
  PKO/KO/Mystery —— 同上（无真实 bounty 语料）。
  即：MTT/PKO 的缺口是**数据缺口**，不是契约拒绝，也不是策略缺陷。
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from .artifact_guard import ArtifactGuard

ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_VERSION = "r1-benchmark-v1"

# ── 证据等级词汇表（唯一口径；不允许自造同义标签） ───────────────────────
EVIDENCE_GRADES: dict[str, str] = {
    "REAL_OBSERVED": "真实观测行为（人类座位真实动作）——唯一 ground truth",
    "HISTORICAL_REPLAY": "真实历史手牌重放：同手牌状态重建 → 模型决策",
    "COUNTERFACTUAL": "反事实臂：同一状态换范围/证据条件（未真实发生）",
    "SYNTHETIC": "合成/fixture 数据（确定性生成，非真实历史）",
    "MECHANISM_ONLY": "仅机制/隔离/确定性/边界验证——不构成任何质量结论",
    "NOT_VALIDATABLE": "当前数据不可验证（不得以任何指标替代）",
}

# ── 置信度分级（与样本量/独立性绑定，不与环境无关地宣称） ───────────────
CONFIDENCE_LEVELS: dict[str, str] = {
    "HIGH": "真实语料 + 无泄漏分割 + n 大 + 有对照臂 + 可复现",
    "MEDIUM": "真实语料但存在结构性限制（如单街近似、研究对局）",
    "LOW": "机制级或极小样本；仅作方向性参考",
    "NONE": "无证据（数据缺失）",
}

# ── 指标定义（**冻结引用**；R1 不得改定义） ─────────────────────────────
FROZEN_METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "fold_brier": {
        "definition": "mean((p_fold_pred − y_observed_fold)²)，逐 eval fold 决策点",
        "predicted": "模型 p_fold（BASELINE=population 条件率 / REAL=player 条件估计）",
        "observed": "人类座位真实 fold 行为",
        "source": "poker_decision_ai/pluribus_validation.py::_metrics",
        "must_not_redefine": True,
    },
    "fold_agreement": {
        "definition": "share((p_fold_pred ≥ 0.5) == bool(y_observed_fold))",
        "source": "poker_decision_ai/pluribus_validation.py::_metrics",
        "must_not_redefine": True,
    },
    "observed_action_agreement": {
        "definition": ("share(arm_best_action == observed_human_action)，"
                       "分母=有 best 的决策点；臂=BASELINE / REAL"),
        "source": "poker_decision_ai/pluribus_validation.py::validate::decision_ev",
        "must_not_redefine": True,
    },
    "action_change_rate": {
        "definition": "share(REAL 臂 best action ≠ BASELINE 臂 best action)",
        "source": "poker_decision_ai/pluribus_validation.py::validate::decision_ev",
        "must_not_redefine": True,
    },
    "regret_proxy": {
        "definition": ("model_best_utility − observed_action_utility（同模型族内；"
                       "构造性 ≤0）；zero_share=观测动作即模型最优的份额"),
        "source": "poker_decision_ai/pluribus_validation.py::_decision_ev_row",
        "must_not_redefine": True,
    },
    "legality": {
        "definition": "决策点推荐动作是否落在该状态合法动作集内（legal/illegal 计数）",
        "source": "poker_decision_ai/pluribus_validation.py::validate::legality",
        "must_not_redefine": True,
    },
}

# ── 冻结文件（策略数学 + 验证口径）：R1 期间必须逐字节不变 ────────────────
FROZEN_STRATEGY_FILES: tuple[str, ...] = (
    "poker_decision_ai/domain/poker/strategy_core.py",
    "poker_decision_ai/domain/poker/bounty.py",
    "poker_decision_ai/strategy_intel.py",
    "poker_decision_ai/strategy_fusion.py",
    "poker_decision_ai/mtt_context.py",
    "poker_decision_ai/pko_context.py",
    "poker_decision_ai/exploit_strategy.py",
    "poker_decision_ai/domain/poker/conditional_model.py",
    "poker_decision_ai/pluribus_validation.py",
    "poker_decision_ai/pluribus_decision_adapter.py",
    "poker_decision_ai/decision_grade.py",
)


def frozen_hashes(paths: tuple[str, ...] | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in (paths or FROZEN_STRATEGY_FILES):
        p = ROOT / rel
        out[rel] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() \
            else "MISSING"
    return out


def verify_frozen(expected: dict[str, str]) -> dict:
    now = frozen_hashes(tuple(expected.keys()) or None)
    mism = sorted(k for k, v in expected.items() if now.get(k) != v)
    return {"match": not mism, "mismatches": mism, "checked": len(expected)}


# ── 冻结 benchmark manifest ────────────────────────────────────────────
PLURIBUS_PARQUET_DIR = Path("C:/PokerDataset/UofT_CPRG/data")
UOFT_NORMALIZED_DIR = ROOT / "data" / "uoft" / "normalized"
UOFT_MANIFEST = ROOT / "data" / "uoft" / "manifest.json"

#: B7-2/B7-8：运行期工件登记清单（fail closed；禁止 fallback）
ARTIFACT_MANIFEST = ROOT / "docs" / "V3F_ARTIFACT_MANIFEST.json"
_GUARD: ArtifactGuard | None = None


def _guard() -> ArtifactGuard:
    """本模块全部历史工件读取的唯一入口守卫（惰性单例）。

    单例只缓存**守卫对象**；`require()` 每次重算工件真实 SHA-256，且每次从
    manifest 重新载入登记集 —— 不存在陈旧授权/陈旧哈希。
    """
    global _GUARD
    if _GUARD is None:
        _GUARD = ArtifactGuard(ARTIFACT_MANIFEST, root=ROOT)
    return _GUARD


def build_manifest(*, decision_hand_limit: int = 400,
                   equity_iterations: int = 1500) -> dict:
    """统一冻结 manifest：数据身份 / 切分 / 随机种子 / 限制 / 指标定义 / 哈希。"""
    from .pluribus_validation import (EVIDENCE_FRAC, EQUITY_ITERATIONS,
                                      MIN_SAMPLES, NEUTRAL_FOLD)
    n_parquet = len(list(PLURIBUS_PARQUET_DIR.glob("*.parquet"))) \
        if PLURIBUS_PARQUET_DIR.is_dir() else 0
    # UofT normalized manifest：读取前 fail closed 校验（未登记/缺失/stale → 拒绝）
    _guard().require("data/uoft/manifest.json")
    uoft = json.loads(UOFT_MANIFEST.read_text(encoding="utf-8"))
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "purpose": ("策略质量基线：冻结、可重复、可审计；不新增能力、不改数学、"
                    "不为指标调参"),
        "datasets": {
            "cash_pluribus_human_seats": {
                "kind": "decision_grade_real",
                "evidence_grade": "REAL_OBSERVED + HISTORICAL_REPLAY",
                "raw_parquet_dir": str(PLURIBUS_PARQUET_DIR),
                "parquet_files": n_parquet,
                "dataset": "UofT CPRG Poker Hand Histories（Pluribus 研究对局）",
                "license": "cc-by-4.0",
                "source": "https://zenodo.org/records/13997158",
                "ai_seat_excluded": "Pluribus 座位 = solver_label（全程排除）",
                "caveats": ["研究对局条件（非自然线上行为）",
                            "单一盲注级 50/100、100bb 起始栈",
                            "人类座位为匿名职业玩家"],
            },
            "cash_uoft_normalized": {
                "kind": "corpus",
                "evidence_grade": "HISTORICAL_REPLAY（行为/范围证据）",
                "dir": str(UOFT_NORMALIZED_DIR),
                "dataset_version": uoft.get("dataset_version"),
                "hand_count": uoft.get("hand_count"),
                "event_count": uoft.get("event_count"),
                "player_count": uoft.get("player_count"),
                "license": "cc-by-4.0",
                "known_issues": (uoft.get("quality") or {}).get("known_issues"),
            },
            "mtt_corpus": {
                "kind": "absent",
                "evidence_grade": "NOT_VALIDATABLE",
                "reason": ("无真实锦标赛语料：UofT CPRG 为 cash-only"
                           "（venue 全现金桌）"),
            },
            "pko_corpus": {
                "kind": "absent",
                "evidence_grade": "NOT_VALIDATABLE",
                "reason": ("无真实 bounty/淘汰赛语料；mystery 真实信封数据源"
                           "为接口占位"),
            },
            "synthetic_fixtures": {
                "kind": "synthetic",
                "evidence_grade": "SYNTHETIC",
                "locations": ["data/hand_history/synthetic_cash_100（seed=42）",
                              "tests/** fixtures"],
                "usage": "仅管线/边界/契约验证；**不得**用于质量结论",
            },
        },
        "split": {
            "rule": "hand_id 字典序确定性排序；前 70% = evidence，后 30% = evaluation",
            "evidence_frac": EVIDENCE_FRAC,
            "no_overlap": "证据/评估 hand_id 集合不相交（本模块泄漏审计断言）",
            "ai_seat": "Pluribus 动作不进入任何证据/评估（solver_label）",
            "leakage_free_required": True,
        },
        "seeds_and_limits": {
            "equity_iterations": equity_iterations,
            "module_default_equity_iterations": EQUITY_ITERATIONS,
            "decision_hand_limit": decision_hand_limit,
            "min_samples": MIN_SAMPLES,
            "neutral_fold": NEUTRAL_FOLD,
            "uoft_corpus_sampling_seed": (uoft.get("sampling") or {}).get("seed"),
            "determinism": "全部管线确定性（无 wall-clock/随机源）",
        },
        "metric_definitions": FROZEN_METRIC_DEFINITIONS,
        "metric_discipline": {
            "no_single_metric_strength_claim": ("agreement / Brier / regret proxy "
                                               "任一单独指标均不构成强度证明"),
            "profitability": "不宣称任何盈利能力；本基线不含实盘结果",
            "no_tuning": "任何指标不得作为调参目标（R1 禁止）",
        },
        "frozen_files": frozen_hashes(),
        "code_freeze_statement": ("R1 期间 strategy_core / fusion / range / fold / "
                                  "exploit / MTT / PKO 数学零修改（哈希断言）"),
    }


# ── 泄漏审计 ───────────────────────────────────────────────────────────
def leakage_audit(hands: list[dict]) -> dict:
    """证据/评估分割不相交 + AI 座位隔离 + 无 GT 泄漏。"""
    from .pluribus_validation import AI_SEAT, EVIDENCE_FRAC, split_hands
    train, ev = split_hands(hands)
    tr_ids = {str(h["hand_id"]) for h in train}
    ev_ids = {str(h["hand_id"]) for h in ev}
    overlap = tr_ids & ev_ids
    def _pid(p: Any) -> str:
        return str(p.get("player_id")) if isinstance(p, dict) else str(p)

    ai_hands = sum(1 for h in hands
                   if any(_pid(p) == str(AI_SEAT)
                          for p in (h.get("players") or [])))
    return {
        "n_hands": len(hands),
        "evidence_hands": len(train),
        "evaluation_hands": len(ev),
        "evidence_frac_declared": EVIDENCE_FRAC,
        "hand_id_overlap": sorted(overlap)[:5],
        "leakage_free": not overlap,
        "ai_seat": str(AI_SEAT),
        "hands_containing_ai_seat": ai_hands,
        "ai_seat_policy": "solver_label——不进入证据/评估（由 validate 保证）",
        "ground_truth_types": ["REAL_OBSERVED（人类动作）",
                               "solver_label（AI 座位，排除）",
                               "model_recommendation（模型输出，非 GT）"],
    }


# ── 三模式质量矩阵（本阶段核心结论骨架） ────────────────────────────────
def mode_matrix() -> dict:
    from .decision_grade import mode_validation_scope
    cash_scope = mode_validation_scope("cash")
    mtt_scope = mode_validation_scope("mtt")
    pko_scope = mode_validation_scope("pko")
    return {
        "cash": {
            "contract": cash_scope,
            "data_available": True,
            "validatable_now": True,
            "evidence_grade": "REAL_OBSERVED + HISTORICAL_REPLAY",
            "confidence": "HIGH",
            "metrics_available": ["fold_brier", "fold_agreement",
                                  "observed_action_agreement",
                                  "action_change_rate", "regret_proxy",
                                  "legality", "unknown_propagation"],
            "note": "唯一具备 decision-grade 真实语料与对照臂的模式",
        },
        "mtt": {
            "contract": mtt_scope,
            "data_available": False,
            "validatable_now": False,
            "evidence_grade": "NOT_VALIDATABLE",
            "confidence": "NONE",
            "metrics_available": [],
            "mechanism_only_checks": ["objective=chip_ev+icm 契约",
                                      "ICM 可用性 vs payouts",
                                      "MTT 无 bounty（隔离）",
                                      "determinism", "boundary/UNKNOWN"],
            "note": ("契约允许（requires payout_icm_context 等）但当前无真实"
                     "锦标赛语料 → 数据缺口；不得以机制验证替代质量结论"),
        },
        "pko": {
            "contract": pko_scope,
            "data_available": False,
            "validatable_now": False,
            "evidence_grade": "NOT_VALIDATABLE",
            "confidence": "NONE",
            "metrics_available": [],
            "mechanism_only_checks": ["objective=chip_ev+icm+bounty 契约",
                                      "bounty 通道激活 vs 缺失诚实降级",
                                      "P44 knockout 与 P67 context 分账不双计",
                                      "determinism", "boundary/UNKNOWN"],
            "note": ("契约允许但无真实 bounty 语料；KO/Mystery 同结论。"
                     "不得伪造 bounty 数据或质量指标"),
        },
    }


# ── CASH 质量测量（复用冻结实现，不重定义指标） ─────────────────────────
def run_cash_benchmark(*, decision_hand_limit: int = 400,
                       equity_iterations: int = 1500,
                       hands: list[dict] | None = None) -> dict:
    """在真实 Pluribus human-seat 语料上重放决策级验证（冻结口径）。"""
    from .pluribus_decision_adapter import (iter_pluribus_rows,
                                            parse_hand_row)
    from .pluribus_validation import validate
    if hands is None:
        hands = [p for row in iter_pluribus_rows() if (p := parse_hand_row(row))]
    val = validate(hands, decision_hand_limit=decision_hand_limit,
                   equity_iterations=equity_iterations)
    return {
        "n_hands": len(hands),
        "split": val["split"],
        "labels": val["labels"],
        "unknown": val["unknown"],
        "legality": val["legality"],
        "fold_calibration": val["fold_calibration"],
        "decision_ev": val["decision_ev"],
        "manifest_core": val["manifest"],
        "evidence_grades": {
            "fold_calibration": "REAL_OBSERVED（人类 fold 行为）",
            "decision_ev.observed_action_agreement": ("REAL_OBSERVED 对照 + "
                                                      "模型输出"),
            "decision_ev.action_change_rate": "COUNTERFACTUAL（REAL vs BASELINE 臂）",
            "decision_ev.regret_proxy": "COUNTERFACTUAL（同模型族内代理量）",
            "legality": "REAL_OBSERVED 状态 + 契约判定",
        },
        "leakage_audit": leakage_audit(hands),
    }


def known_ceilings() -> list[dict]:
    """已知结构性天花板与局限（引用既有实证，不新造结论）。"""
    return [
        {"id": "A1_single_street_call_ceiling",
         "statement": ("单街近似（A1）下 call 一致率 ≈0.7%（BASELINE/REAL 双臂）"
                       "——set mining / implied odds / 多街价值不可见"),
         "evidence": "P55 divergence diagnosis（真实 Pluribus，22,947 决策点）",
         "evidence_grade": "REAL_OBSERVED + COUNTERFACTUAL",
         "impact": "observed_action_agreement 的结构性天花板（与证据质量无关）"},
        {"id": "A1_single_street_bet_bias",
         "statement": ("REAL 臂在人类 bet/check 时约 96–97% 推荐 bet——单街 "
                       "fold equity 奖励 + 无未来街价值"),
         "evidence": "P55 §3",
         "evidence_grade": "COUNTERFACTUAL",
         "impact": "check→bet 主导 action change；是否『更优』不可判"},
        {"id": "fold_calibration_vs_agreement",
         "statement": ("fold 校准改善（ΔBrier≈−0.0032）与 agreement 下降"
                       "是同一机制（fold 证据激活）的两个侧面——两个指标度量"
                       "不同对象，不可互相替代"),
         "evidence": "P55 §4",
         "evidence_grade": "REAL_OBSERVED + COUNTERFACTUAL",
         "impact": "禁止以单一指标宣称强度"},
        {"id": "research_play_extrapolation",
         "statement": ("Pluribus 为研究对局（固定盲注/100bb/匿名职业玩家）——"
                       "结论不得外推自然线上玩家池"),
         "evidence": "P54/P55 manifest caveats",
         "evidence_grade": "REAL_OBSERVED（受条件限制）",
         "impact": "外部效度限制"},
        {"id": "cash_only_corpus",
         "statement": ("现有真实语料 cash-only → MTT/PKO 质量不可验证"),
         "evidence": "P51/P53/P54 honesty 条款",
         "evidence_grade": "NOT_VALIDATABLE",
         "impact": "目标覆盖（MTT/PKO）的质量证据缺口"},
        {"id": "no_multistreet_reference",
         "statement": ("无多街/混合策略基准（solver 环境缺失）→ λ 幅度与"
                       "check→bet 转化是否恰当不可判定"),
         "evidence": "P55 §8；P61 solver audit",
         "evidence_grade": "NOT_VALIDATABLE（等待外部环境）",
         "impact": "质量上限判定受限（非本阶段 blocker）"},
    ]


# ── MTT/PKO 机制验证（仅机制，不构成质量结论） ───────────────────────────
def mechanism_checks_mtt_pko() -> dict:
    """复用 P68 跨模式压力 + intel 契约：机制/隔离/确定性/边界。"""
    from .strategy_stress_xmode import (MODES, build_scenarios,
                                        determinism_sample, run_cross_stress)
    cells = build_scenarios()
    stress = run_cross_stress()
    det = determinism_sample(12)
    return {
        "evidence_grade": "MECHANISM_ONLY",
        "statement": ("MTT/PKO 无真实语料 → 以下仅为机制/隔离/确定性/边界，"
                      "**不构成任何质量结论**"),
        "modes": list(MODES),
        "cross_mode_cells": len(cells),
        "violations": stress.get("n_violation_cells"),
        "violation_types": stress.get("violation_types"),
        "determinism": det,
        "contract": {
            "mtt": "objective=chip_ev+icm；无 payouts → 诚实 chip_ev",
            "pko": "objective=chip_ev+icm+bounty；bounty 缺失 → 诚实降级",
            "cash": "objective=chip_ev；ICM/bounty 永不激活",
        },
    }


# ── 快照组装 / 与 P69 baseline 对比 ─────────────────────────────────────
REFERENCE_SNAPSHOTS = {
    "P54_decision_grade": "docs/V3F_P54_DECISION_VALIDATION_SNAPSHOT.json",
    "P55_divergence": "docs/V3F_P55_DIVERGENCE_DIAGNOSIS.json",
    "P56_continuum_v31": "docs/V3F_P56_CONTINUUM_UPGRADE_SNAPSHOT.json",
    "P51_uoft_real": "docs/V3F_P51_REAL_VALIDATION_SNAPSHOT.json",
}


def reference_metrics() -> dict:
    """读取既有快照的冻结指标（只读；不重算、不改定义）。

    B7-2/B7-8：每个快照在读取前逐一经 `ArtifactGuard.require()` fail closed
    校验；未登记 / 缺失 / stale / 哈希不一致 → `ArtifactGuardError`，
    **不回退**到空指标、也不读取未校验文件。
    """
    out: dict[str, Any] = {}
    guard = _guard()
    for name, rel in REFERENCE_SNAPSHOTS.items():
        guard.require(rel)                      # 读取前校验（先校验后读）
        d = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        if name == "P54_decision_grade":
            v = d.get("validation", {})
            out[name] = {"present": True,
                         "fold_calibration": v.get("fold_calibration"),
                         "decision_ev": v.get("decision_ev"),
                         "split": v.get("split"),
                         "legality": v.get("legality")}
        elif name == "P55_divergence":
            bm = d.get("base_metrics", {})
            out[name] = {"present": True,
                         "fold_calibration": bm.get("fold_calibration"),
                         "decision_ev": d.get("decision_ev_headline"),
                         "legality": bm.get("legality")}
        elif name == "P56_continuum_v31":
            out[name] = {"present": True,
                         "fold_calibration": d.get("fold_calibration"),
                         "before_after": d.get("before_after")}
        else:
            out[name] = {"present": True,
                         "behavioral_calibration": (d.get("behavioral_calibration")
                                                    or {}).get("overall")}
    return out


def compare_with_references(cash: dict) -> dict:
    """与 P69 baseline（既有快照）逐指标对比：一致 / 漂移 + 解释。"""
    refs = reference_metrics()
    cur_fc = (cash.get("fold_calibration") or {}).get("arms") or {}
    cur_de = cash.get("decision_ev") or {}
    diffs: list[dict] = []

    def _cmp(label: str, cur, ref, src: str):
        if cur is None or ref is None:
            diffs.append({"metric": label, "current": cur, "reference": ref,
                          "source": src, "status": "unavailable"})
            return
        same = abs(float(cur) - float(ref)) <= 1e-6
        diffs.append({"metric": label, "current": cur, "reference": ref,
                      "source": src,
                      "status": "identical" if same else "drift",
                      "delta": None if same else round(float(cur) - float(ref), 6)})

    p56 = refs.get("P56_continuum_v31", {})
    p56_fc = (p56.get("fold_calibration") or {}).get("arms") or {}
    for arm in ("BASELINE", "REAL"):
        _cmp(f"fold_brier.{arm}", (cur_fc.get(arm) or {}).get("brier"),
             (p56_fc.get(arm) or {}).get("brier"), "P56 fold_calibration")
        _cmp(f"fold_agreement.{arm}", (cur_fc.get(arm) or {}).get("agreement"),
             (p56_fc.get(arm) or {}).get("agreement"), "P56 fold_calibration")
    p56ba = (p56.get("before_after") or {}).get("v3_1_decision_ev") or {}
    _cmp("observed_agreement_baseline", cur_de.get("observed_agreement_baseline"),
         p56ba.get("observed_agreement_baseline"),
         "P56 before_after.v3_1_decision_ev")
    _cmp("observed_agreement_real", cur_de.get("observed_agreement_real"),
         p56ba.get("observed_agreement_real"), "P56 before_after.v3_1_decision_ev")
    _cmp("action_change_rate", cur_de.get("action_change_rate"),
         p56ba.get("action_change_rate"), "P56 before_after.v3_1_decision_ev")
    _cmp("regret_zero_share", cur_de.get("regret_zero_share"),
         p56ba.get("regret_zero_share"), "P56 before_after.v3_1_decision_ev")
    return {
        "reference_snapshots": refs,
        "comparisons": diffs,
        "identical_share": (round(sum(1 for d in diffs
                                      if d["status"] == "identical")
                                 / len(diffs), 6) if diffs else None),
        "interpretation": ("identical = 与 P69 baseline 数值一致（R1 未引入漂移）；"
                           "drift = 需解释来源（策略核心在 P56 之后的冻结变更/"
                           "数据前缀差异）"),
    }


def build_snapshot(*, decision_hand_limit: int = 400,
                   equity_iterations: int = 1500,
                   out_path: str = "docs/V3F_R1_QUALITY_BASELINE.json",
                   include_cash_run: bool = True) -> dict:
    """组装并写出冻结质量基线快照（确定性；无时间戳）。"""
    manifest = build_manifest(decision_hand_limit=decision_hand_limit,
                              equity_iterations=equity_iterations)
    cash = run_cash_benchmark(decision_hand_limit=decision_hand_limit,
                              equity_iterations=equity_iterations) \
        if include_cash_run else {}
    snap = {
        "snapshot_version": BENCHMARK_VERSION,
        "manifest": manifest,
        "mode_matrix": mode_matrix(),
        "cash": cash,
        "mtt_pko": mechanism_checks_mtt_pko(),
        "known_ceilings": known_ceilings(),
        "p69_baseline_comparison": compare_with_references(cash) if cash else {},
        "layering": {
            "engineering_complete": True,
            "strategy_capability_complete": "单街声明式范畴内完成",
            "strategy_quality_validated": ("部分：CASH 有真实对照；MTT/PKO 不可"
                                           "验证"),
            "real_world_profitability_validated": False,
        },
        "honesty": {
            "no_profitability_claim": "本基线不构成任何盈利性/GTO 证明",
            "no_single_metric_strength": ("agreement / Brier / regret proxy "
                                          "任一单独指标不得作为强度证明"),
            "mtt_pko": "NOT VALIDATABLE WITH CURRENT DATA（数据缺口）",
            "synthetic_excluded": "合成/fixture 结果不得进入质量结论",
            "research_play": "Pluribus 研究对局——不外推自然线上玩家",
            "no_tuning": "本阶段未因指标结果修改任何策略常数/公式",
        },
    }
    out = ROOT / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    return snap
