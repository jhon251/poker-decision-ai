"""解冻实验研究分支：IMR 独立参照 + 变体计算（**不修改任何生产模块**）。

包含：
  * `PiEvReference`  —— Pluribus 全明牌 PI-EV 参照（perfect-information，NOT
    equilibrium）：全枚举剩余牌，按**声明式 check-down 规则**估值；超预算点
    返回 unavailable（**不得以近似冒充精确**）。
  * `SolverReference` —— 真 Solver 占位：未配置 → `is_configured()=False` +
    `unavailable_reason`（**BLOCKED**）；**禁止任何伪实现**。
  * `compute_ev_variant` —— 变体计算：`frozen` 直接委托冻结核（逐位等价）；
    U1/U2/U3 以**文档化的后修正**实现（不复制冻结逻辑）。

变体定义（审核后确定；见实验报告）
  U1 : raise 分支损失基础改为 hero 实际投入 `r`（冻结公式用增量 `r − to_call`）
       → ΔEV(raise) = −p_c×(1−eq_c)×to_call
  U3 : call/raise 同口径化 —— **审核发现冻结核的 call/bet 分支已满足真实筹码
       口径**，唯一不一致即 U1；故 U1+U3 ≡ U1（作为审核发现如实上报）
  U2 : CV 幅度缩放（λ 锚定在 EV 幅度上的投影）：cv' = λ_scale × cv
       λ_scale ∈ {0.5, 0.75, 1.0, 1.25, 1.5}（敏感性扫描）
"""
from __future__ import annotations

from typing import Any

from ..domain.poker.evaluator import evaluate7
from ..domain.poker.strategy_core import compute_ev
from ..multistreet_v2 import child_equity  # noqa: F401  (复用，便于下游一致)

UNFREEZE_VERSION = "unfreeze-exp-v1"
LAMBDA_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)
PI_EV_MAX_RUNOUTS = 2048          # 预算（flop 需 C(45,2)=990，足够；超限即 unavailable）
_RANKS = "23456789TJQKA"
_SUITS = "shdc"


# ── Solver 参照：不可用即 BLOCKED（严禁伪实现） ──────────────────────────
class SolverReference:
    kind = "solver"
    evidence_level = "L4"

    def __init__(self) -> None:
        import os
        self.path = os.environ.get("POSTFLOP_SOLVER_PATH", "")

    def is_configured(self) -> bool:
        return bool(self.path)

    def unavailable_reason(self) -> str | None:
        if self.is_configured():
            return None
        return ("POSTFLOP_SOLVER_PATH not set → real solver unavailable "
                "(BLOCKED; no pseudo-solver permitted)")

    def reference(self, *a, **kw) -> dict:
        if not self.is_configured():
            return {"available": False,
                    "unavailable_reason": self.unavailable_reason(),
                    "provenance": {"kind": self.kind,
                                   "equilibrium": False,
                                   "blocked": True}}
        raise NotImplementedError(
            "real solver integration is out of scope for this experiment")


# ── PI-EV 参照（perfect-information；NOT equilibrium） ──────────────────
def _deck_excluding(known: set[str]) -> list[str]:
    return [r + s for r in _RANKS for s in _SUITS if (r + s) not in known]


def _pi_equity(hero: list[str], others: list[list[str]], board: list[str]
               ) -> tuple[float | None, int]:
    """全枚举剩余公共牌 → (PI 权益, runout 数)。others = 活跃对手手牌列表。"""
    known = {c for c in hero + board}
    for o in others:
        known |= set(o)
    deck = [c for c in _deck_excluding(known) if c not in board]
    need = 5 - len(board)
    if need < 0 or not others:
        return None, 0
    from itertools import combinations
    runouts = list(combinations(deck, need)) if need else [()]
    if len(runouts) > PI_EV_MAX_RUNOUTS:
        return None, len(runouts)          # 超预算 → unavailable（不近似）
    tot = 0.0
    for ro in runouts:
        full = board + list(ro)
        hs = evaluate7([_card(c) for c in hero + full])
        best_other = max(evaluate7([_card(c) for c in o + full])
                         for o in others)
        if hs > best_other:
            tot += 1.0
        elif hs == best_other:
            tot += 0.5
    return tot / len(runouts), len(runouts)


def _card(code: str):
    from ..domain.poker.cards import parse_card
    return parse_card(str(code))


class PiEvReference:
    """Pluribus 全明牌参照：**perfect-information，非均衡**。"""

    kind = "pi_ev"
    evidence_level = "L2"

    def is_configured(self) -> bool:
        return True

    def unavailable_reason(self) -> str | None:
        return None

    def reference(self, *, hero_cards: list[str], villain_hands: list[list[str]],
                  board: list[str], pot_bb: float, to_call_bb: float,
                  hero_stack_bb: float, candidates: list[dict],
                  rule: str = "villain_calls") -> dict:
        """按声明式 check-down 规则给出各候选的 PI-EV（bb）与排序。

        rule = "villain_calls"（主口径，对激进线保守；声明式、非均衡）
             | "villain_folds"（敏感性口径：仅下注/加注候选受影响）
        """
        eq, n_run = _pi_equity(list(hero_cards), [list(v) for v in
                                                  villain_hands],
                               [str(c) for c in board])
        prov = {"kind": self.kind, "evidence_level": self.evidence_level,
                "reference": "perfect-information",
                "equilibrium": False,
                "rule": f"checkdown_{rule}",
                "runouts_enumerated": n_run, "NOT_exact_gto": True}
        if eq is None:
            return {"available": False,
                    "unavailable_reason": ("runout budget exceeded or no "
                                           "active opponent"),
                    "provenance": prov, "values": {}, "ordering": []}
        vals: dict[str, dict] = {}
        for c in candidates:
            act, amt = str(c["action"]), float(c.get("amount_bb") or 0.0)
            if act == "fold":
                v = 0.0
            elif act == "check":
                v = eq * pot_bb
            elif act == "call":
                s = min(to_call_bb, hero_stack_bb)
                v = eq * pot_bb - (1.0 - eq) * s
            elif act == "bet":
                if rule == "villain_folds":
                    v = pot_bb
                else:
                    v = eq * (pot_bb + amt) - (1.0 - eq) * amt
            elif act == "raise":
                if rule == "villain_folds":
                    v = pot_bb
                else:
                    v = (eq * (pot_bb + max(0.0, amt - to_call_bb))
                         - (1.0 - eq) * amt)
            else:
                continue
            vals[act] = {"value_bb": round(v, 6), "pi_equity": round(eq, 6),
                         "n_runouts": n_run, "ci95": None}
        ordering = [k for k, _ in sorted(vals.items(),
                                         key=lambda kv: -kv[1]["value_bb"])]
        return {"available": True, "unavailable_reason": None,
                "values": vals, "ordering": ordering, "provenance": prov,
                "pi_equity": round(eq, 6), "n_runouts": n_run}


# ── 变体计算（frozen 逐位等价；U1/U2/U3 为文档化后修正） ────────────────
def compute_ev_variant(state: dict, equity_src: dict, range_src: dict | None,
                       opponent_src: dict | None, *, variant: str = "frozen",
                       lambda_scale: float = 1.0, seed: int = 42,
                       base=None):
    """返回 (EvResult, variant_meta)。

    variant = "frozen"       → 直接委托冻结 `compute_ev`（逐位等价，构造保证）
              "U1" / "U1+U3" → raise 损失基础修正（U3 审核为同一修正）
              "U1+U2+U3"     → 上述 + CV 幅度缩放 `lambda_scale`

    `base`：可传入**已算好的冻结核结果**以复用（变体是后修正；函数内部深拷贝，
    绝不改动调用方对象）。
    """
    import copy
    if base is None:
        base = compute_ev(state, equity_src, range_src=range_src,
                          opponent_src=opponent_src, seed=seed)
    meta: dict[str, Any] = {"variant": variant, "lambda_scale": lambda_scale,
                            "u1_applied": False, "u2_applied": False,
                            "u3_note": ("calls/bets 已被冻结核按真实筹码口径计价"
                                        "→ U3 无额外修正")}
    if variant == "frozen" or not base.available:
        return copy.deepcopy(base), meta
    base = copy.deepcopy(base)
    bb = float(state.get("bb") or 0.0)
    to_call_bb = float(state.get("to_call") or 0.0) / bb if bb else 0.0
    eq = float(base.equity or 0.0)
    for c in base.candidates:
        # U2：CV 幅度缩放（λ 锚定在 EV 幅度上的投影）
        if variant == "U1+U2+U3" and lambda_scale != 1.0 and c.cv_bb:
            delta = (lambda_scale - 1.0) * float(c.cv_bb)
            c.ev_bb = (c.ev_bb or 0.0) + delta
            c.utility_bb = (c.utility_bb if c.utility_bb is not None
                            else c.ev_bb) + delta
            meta["u2_applied"] = True
        # U1：raise 损失基础 → hero 实际投入 r
        if c.action == "raise" and to_call_bb > 0:
            p_c = float(c.p_continue if c.p_continue is not None
                        else (1.0 - float(c.p_fold or 0.0)))
            eq_c = eq
            if c.a3v2_delta_bb is not None and p_c > 0 and c.amount_bb:
                denom = p_c * (float(state.get("pot") or 0.0) / bb
                               + (float(c.amount_bb) - to_call_bb))
                if abs(denom) > 1e-9:
                    eq_c = eq + float(c.a3v2_delta_bb) / denom
                    eq_c = min(max(eq_c, 0.0), 1.0)
            corr = p_c * (1.0 - eq_c) * to_call_bb      # 损失增加 → EV 下降
            c.ev_bb = (c.ev_bb or 0.0) - corr
            if c.utility_bb is not None:
                c.utility_bb = c.utility_bb - corr
            meta["u1_applied"] = True
    # 重排候选（保持与冻结核一致的排序语义：按 utility 降序）
    base.candidates.sort(key=lambda c: -(c.utility_bb if c.utility_bb
                                         is not None else (c.ev_bb or 0.0)),
                         reverse=False)
    base.candidates.sort(key=lambda c: (c.utility_bb if c.utility_bb
                                        is not None else (c.ev_bb or 0.0)),
                         reverse=True)
    if base.candidates:
        top = base.candidates[0]
        base.best_action = top.action
        base.best_amount_bb = top.amount_bb
        base.best_ev_bb = top.ev_bb
        base.best_utility_bb = (top.utility_bb if top.utility_bb is not None
                                else top.ev_bb)
    return base, meta
