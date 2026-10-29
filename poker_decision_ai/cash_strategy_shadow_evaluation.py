"""V3H-10 — CASH Shadow Evaluation Record（建议 ↔ 已观测 hero 动作 的对账留痕）。

**为什么这是 V3H-10 的最小下一能力（审计结论）**
--------------------------------------------------
实测现状（13 个 V3H 策略层）：

| 已有 | 实测缺口 |
|---|---|
| V3H-8 `assemble_decision` 产出**可审计建议** | — |
| V3H-9 `audit_experiment_eligibility` 逐决策判断实验入口 | — |
| belief 层消费 `observed_action` | 但那些是**对手**动作（`cash_strategy_dynamic_belief` / `belief_update` / `belief_conditioned_eval`） |
| — | **没有任何层消费"hero 实际执行了什么"**，也没有"建议 ↔ 期望值观测"的对账（实测：`decision_assembly` / `experiment_eligibility` / `action_ranking` 三处 `observed_action` / `hero_action` / `divergence` 计数**全为 0**） |

因此**闭环缺最后一环**：没有"建议 → 已观测动作 → 分歧分类 → 模型内反事实快照"
的可复现记录。本层补上这一环，**不新增任何评分/EV 公式**。

**严格定位（必须如实标注）**
----------------------------
本层产出的是**模型内、事后的对账快照**，具体地：

* `recommended` 与 `observed` 是**两个事实**（建议 / 已观测），分别留痕；
* 分歧分类是**描述性**的（aligned / different / unavailable），
  **不是**"推荐更优"的判断；
* 反事实快照 `shadow_snapshot` 只用**既有** V3H-5
  `conditional_absolute_equity`（条件绝对权益）在**同一 board** 上对两个动作取值，
  该原语由**调用方显式注入**（惯例同 V3H-9 的 `gate=`），本层不自行 import、
  不新增公式；未注入 → 快照 fail closed；
  并**明确标注**它不是收益、不是 EV 差、不构成因果；
* **禁止**从中推出任何盈利/因果结论（`claim_boundary` 机器可读）。

硬约束
------
* CASH-only，fail closed；非 dict / 非 CASH → `present=False`；
* 不 import 任何 V3G 层（任何位置）；不 import 伪随机数模块；无 IO；
* 同输入 → 同输出（含 unknown 列表顺序）；
* UNKNOWN / UNAVAILABLE / abstention / disagreement **原样传播**，不默认、不猜测；
* 不执行任何动作；不做自动下注；不产生收益/可利用度结论；
* `MODEL_INTERNAL` 仅描述**证据等级**，**不是**真实对手证据，也**不是**真实盈利证据。
"""
from __future__ import annotations

from typing import Any

CASH_STRATEGY_SHADOW_EVALUATION_VERSION = "v3h10-cash-strategy-shadow-evaluation-v1"

#: 分歧分类（枚举封闭；**描述性**，不含优劣判断）
DIVERGENCE_CLASSES: tuple[str, ...] = (
    "aligned",        # 建议与观测指向同一候选
    "different",      # 两者不同（描述性事实）
    "unavailable",    # 缺建议或缺观测 → 不可比较（不猜）
)

#: 匹配精度（用于候选键对齐）
DEFAULT_AMOUNT_TOLERANCE_BB = 1e-6

CLAIM_BOUNDARY: dict[str, bool] = {
    "is_gto": False,
    "is_equilibrium": False,
    "is_solver_output": False,
    "is_profit_evidence": False,
    "is_causal_claim": False,
    "is_real_opponent_evidence": False,
    "prefers_recommendation_over_observed": False,
    "adds_scoring_or_ev_formula": False,
    "executes_action": False,
    "auto_betting": False,
    "wired_to_production": False,
}


def _is_cash(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    gt = state.get("game_type")
    return isinstance(gt, str) and gt.strip().lower() == "cash"


def _stub(reason: str) -> dict:
    return {
        "present": False,
        "version": CASH_STRATEGY_SHADOW_EVALUATION_VERSION,
        "reason": reason,
        "evidence_grade": "NONE",
        "unknown": [reason],
        "divergence": "unavailable",
        "record": None,
        "executed": False,
        "claim_boundary": dict(CLAIM_BOUNDARY),
    }


def _norm_action(v: Any) -> str | None:
    if isinstance(v, str) and v.strip():
        return v.strip().lower()
    return None


def _key(action: Any, amount: Any, tol: float) -> tuple | None:
    """候选键（不预先 round，容差由 `_same_key` 精确判定）。"""
    act = _norm_action(action)
    if act is None:
        return None
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        return (act, None)
    return (act, float(amount))


def _same_key(a: tuple | None, b: tuple | None, tol: float) -> bool:
    if a is None or b is None:
        return False
    if a[0] != b[0]:
        return False
    if a[1] is None or b[1] is None:
        return a[1] is None and b[1] is None
    return abs(float(a[1]) - float(b[1])) <= tol


def _hero_cards(state: dict) -> tuple[list[str], str | None]:
    hero_id = state.get("hero_id")
    for p in (state.get("players") or []):
        if not isinstance(p, dict):
            continue
        if bool(p.get("is_hero")) or (hero_id is not None
                                      and p.get("player_id") == hero_id):
            hc = p.get("hole_cards")
            if isinstance(hc, list) and len(hc) == 2 and all(
                    isinstance(c, str) and c.strip() for c in hc):
                return [str(c) for c in hc], None
            return [], "hero_hole_cards_missing"
    return [], "hero_not_found"


def _board(state: dict) -> tuple[list[str], str | None]:
    b = state.get("board")
    if not isinstance(b, list):
        return [], "board_missing"
    cards = [str(c) for c in b if isinstance(c, str) and c.strip()]
    if len(cards) != len(b):
        return [], "board_malformed"
    return cards, None


def _snapshot(state: dict, actions: list[str], equity_snapshot) -> dict:
    """模型内反事实快照：**同一 board** 上按动作名给出候选尺寸的权益量。

    `equity_snapshot` **由调用方显式注入**（惯例同 V3H-9 的 `gate=`）：
    正常应传入既有 V3H-5 `conditional_absolute_equity`；本层**不**自行
    import、**不**新增任何公式、**不**设默认值。未注入 → fail closed。

    该量是"跑完枚举语义的条件绝对权益"，**不是**收益、**不是** EV 差、
    **不构成**因果证据。
    """
    if equity_snapshot is None:
        return {"present": False,
                "unknown": ["equity_snapshot_not_supplied"],
                "adds_new_formula": False}
    hole, why = _hero_cards(state)
    board, b_why = _board(state)
    if why or b_why:
        return {"present": False,
                "unknown": [x for x in (why, b_why) if x],
                "adds_new_formula": False}
    if len(board) not in (0, 3, 4, 5):
        return {"present": False, "unknown": ["board_size_invalid"],
                "adds_new_formula": False}
    try:
        eq = equity_snapshot(hole, board)
    except Exception as exc:                               # noqa: BLE001
        return {"present": False,
                "unknown": [f"equity_snapshot_error:{type(exc).__name__}"],
                "adds_new_formula": False}
    if not isinstance(eq, dict) or not eq.get("present"):
        return {"present": False,
                "unknown": list((eq or {}).get("unknown") or [])
                if isinstance(eq, dict) else ["equity_snapshot_bad_return"],
                "adds_new_formula": False}
    out: dict[str, Any] = {
        "present": True,
        "board": list(board),
        "hero_hole_cards": list(hole),
        "conditional_absolute_equity": eq.get("hero_equity"),
        "samples": eq.get("samples"),
        "method": eq.get("method"),
        "source": eq.get("source"),
        "per_action_amounts": {},
        "note": ("同一 board 上的条件绝对权益；**不是**收益、**不是** EV 差、"
                 "**不构成**因果证据；两个动作共享同一权益值，"
                 "因为尺寸差异未重算权益（本层不新增公式）。"),
        "adds_new_formula": False,
    }
    return out


def _observed_from_state(state: dict) -> dict:
    """从 state 读取 hero **已观测**动作（显式字段优先；缺失 → UNKNOWN，不猜）。"""
    obs = state.get("hero_observed_action")
    if isinstance(obs, dict):
        act = _norm_action(obs.get("action"))
        if act is None:
            return {"present": False, "unknown": ["hero_observed_action_missing_action"]}
        amt = obs.get("amount_bb")
        return {"present": True, "action": act,
                "amount_bb": amt if isinstance(amt, (int, float)) else None,
                "source": obs.get("source") or "state.hero_observed_action"}
    hist = state.get("action_history")
    if isinstance(hist, list):
        hero_id = state.get("hero_id")
        for a in hist:
            if not isinstance(a, dict):
                continue
            if bool(a.get("is_hero")) or (hero_id is not None
                                          and a.get("player_id") == hero_id):
                act = _norm_action(a.get("action"))
                if act is None:
                    return {"present": False,
                            "unknown": ["action_history_hero_action_invalid"]}
                amt = a.get("amount_bb")
                return {"present": True, "action": act,
                        "amount_bb": amt if isinstance(amt, (int, float)) else None,
                        "source": "state.action_history(hero)"}
    return {"present": False, "unknown": ["hero_observed_action_absent"]}


def make_shadow_observation(action: Any, amount_bb: Any = None,
                            source: str | None = None) -> dict:
    """构造"已观测 hero 动作"（**调用方显式提供**，本层不推断）。"""
    act = _norm_action(action)
    if act is None:
        return {"present": False, "unknown": ["observed_action_invalid"]}
    return {"present": True, "action": act,
            "amount_bb": (amount_bb if isinstance(amount_bb, (int, float))
                          and not isinstance(amount_bb, bool) else None),
            "source": source or "caller_supplied"}


def evaluate_shadow_decision(state: Any, *, decision: dict | None = None,
                             action_policy: dict | None = None,
                             eligibility: dict | None = None,
                             observed: dict | None = None,
                             equity_snapshot=None,
                             amount_tolerance_bb: float = DEFAULT_AMOUNT_TOLERANCE_BB
                             ) -> dict:
    """产出"建议 ↔ 已观测动作"的模型内对账记录（**不执行、不判优、不称盈利**）。

    Parameters
    ----------
    state : CASH state（非 CASH → present=False）
    decision : **必填**，V3H-8 `assemble_decision` 输出
    action_policy : 可选，V3H-4 `build_action_policy` 输出（提供候选尺寸与合法集）
    eligibility : 可选，V3H-9 `audit_experiment_eligibility` 输出（资格留痕）
    observed : 可选，`make_shadow_observation(...)` 输出；
        未提供 → 尝试从 state 读取；再缺 → UNKNOWN（**不猜**）
    equity_snapshot : 可选，**调用方注入**的既有 V3H-5
        `conditional_absolute_equity`；未注入 → 快照 fail closed（`UNKNOWN`）
    amount_tolerance_bb : 候选键对齐的金额容差（显式）
    """
    if not _is_cash(state):
        return _stub("not_cash_mode_fail_closed")
    if not isinstance(decision, dict):
        return _stub("decision_required")

    unknown: list[str] = []
    dec = decision

    # ── 建议侧（**不重算**，只转述 V3H-8）──
    preferred = dec.get("preferred") if isinstance(dec.get("preferred"), dict) else None
    recommended = None
    if dec.get("present") and not dec.get("abstained") and preferred:
        act = _norm_action(preferred.get("action"))
        if act is None:
            unknown.append("recommendation_action_invalid")
        else:
            recommended = {
                "action": act,
                "amount_bb": (preferred.get("amount_bb")
                              if isinstance(preferred.get("amount_bb"),
                                            (int, float))
                              and not isinstance(preferred.get("amount_bb"), bool)
                              else None),
                "candidate_key": preferred.get("candidate_key"),
                "selected_by_tier": preferred.get("selected_by_tier"),
                "score_source": preferred.get("score_source"),
                "evidence_grade": dec.get("evidence_grade"),
            }
    else:
        unknown.append("recommendation_unavailable:"
                       + ("abstained" if dec.get("abstained")
                          else str(dec.get("reason") or "no_preferred")))

    # ── 观测侧（显式优先；不推断）──
    if isinstance(observed, dict):
        obs = observed
    else:
        obs = _observed_from_state(state)
    if not obs.get("present"):
        for u in (obs.get("unknown") or ["hero_observed_action_absent"]):
            unknown.append(u)
        observed_rec = None
    else:
        act = _norm_action(obs.get("action"))
        if act is None:
            unknown.append("observed_action_invalid")
            observed_rec = None
        else:
            observed_rec = {
                "action": act,
                "amount_bb": (obs.get("amount_bb")
                              if isinstance(obs.get("amount_bb"), (int, float))
                              and not isinstance(obs.get("amount_bb"), bool)
                              else None),
                "source": obs.get("source"),
            }

    # ── 候选尺寸（来自 V3H-4，不重算）──
    cand_sizes: dict[str, Any] = {}
    legal_actions: list[str] = []
    if isinstance(action_policy, dict) and action_policy.get("present"):
        legal_actions = [str(a) for a in (action_policy.get("legal_actions") or [])]
        for c in (action_policy.get("candidates") or []):
            if not isinstance(c, dict):
                continue
            a = _norm_action(c.get("action"))
            if a is None:
                continue
            cand_sizes.setdefault(a, c.get("amount_bb"))
    else:
        unknown.append("action_policy_not_supplied")

    # ── 分歧分类（**描述性**，不含优劣）──
    if recommended is None or observed_rec is None:
        divergence = "unavailable"
    else:
        rk = _key(recommended["action"], recommended["amount_bb"],
                  amount_tolerance_bb)
        ok = _key(observed_rec["action"], observed_rec["amount_bb"],
                  amount_tolerance_bb)
        divergence = ("aligned" if _same_key(rk, ok, amount_tolerance_bb)
                      else "different")

    # ── 模型内反事实快照（复用 V3H-5；不新增公式）──
    snap_actions: list[str] = []
    if recommended:
        snap_actions.append(recommended["action"])
    if observed_rec and observed_rec["action"] not in snap_actions:
        snap_actions.append(observed_rec["action"])
    snapshot = (_snapshot(state, snap_actions, equity_snapshot) if snap_actions
                else {"present": False, "unknown": ["no_action_to_snapshot"],
                      "adds_new_formula": False})

    # ── 资格留痕（转述 V3H-9；不重算）──
    elig_trace = None
    if isinstance(eligibility, dict) and eligibility.get("present"):
        elig_trace = {
            "eligible": eligibility.get("eligible"),
            "status": eligibility.get("status"),
            "result_status": eligibility.get("result_status"),
            "n_blockers": eligibility.get("n_blockers"),
            "source": "V3H-9 audit_experiment_eligibility",
        }
    else:
        unknown.append("eligibility_not_supplied")

    # ── abstention / disagreement / UNKNOWN 原样传播 ──
    if dec.get("abstained") is True:
        unknown.append("decision_abstained")
    if dec.get("disagreements"):
        unknown.append(f"decision_disagreements:{len(dec['disagreements'])}")
    for u in (dec.get("unknown") or []):
        if u not in unknown:
            unknown.append(u)

    available = recommended is not None and observed_rec is not None
    level = "MODEL_INTERNAL" if available else "UNKNOWN"

    record = {
        "recommended": recommended,
        "observed": observed_rec,
        "divergence": divergence,
        "divergence_is_descriptive_only": True,
        "preference_judgement_made": False,
        "candidate_sizes_from_action_policy": cand_sizes,
        "legal_actions": legal_actions,
        "amount_tolerance_bb": amount_tolerance_bb,
        "shadow_snapshot": snapshot,
        "eligibility_trace": elig_trace,
        "evidence_grade": ("MODEL_INTERNAL" if available else "NONE"),
        "executed": False,
        "is_gto": False,
        "is_equilibrium": False,
        "is_solver_output": False,
        "is_profit_evidence": False,
        "is_causal_claim": False,
    }

    return {
        "present": True,
        "version": CASH_STRATEGY_SHADOW_EVALUATION_VERSION,
        "mode": "cash",
        "street": dec.get("street"),
        "divergence": divergence,
        "divergence_classes": list(DIVERGENCE_CLASSES),
        "record": record,
        "available": available,
        "level": level,
        "evidence_grade": ("MODEL_INTERNAL" if available else "NONE"),
        "executed": False,
        "uncertainty": {
            "snapshot_present": bool(snapshot.get("present")),
            "snapshot_method": snapshot.get("method"),
            "snapshot_samples": snapshot.get("samples"),
            "amount_tolerance_bb": amount_tolerance_bb,
            "note": ("反事实快照为**模型内**条件绝对权益；"
                     "两个动作共享同一 board 权益（未按尺寸重算），"
                     "因此它**不能**被读作 EV 差或因果效应。"),
        },
        "unknown": unknown,
        "claim_boundary": dict(CLAIM_BOUNDARY),
        "note": ("本层只做**事后对账留痕**：分岔分类是描述性的，"
                 "**不判优**、**不称盈利**、**不做因果主张**；不执行任何动作。"),
    }
