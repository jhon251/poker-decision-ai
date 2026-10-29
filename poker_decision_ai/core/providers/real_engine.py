"""P34 — REAL Decision Engine（确定性规则/数学决策引擎）。

定位（诚实声明）：
  - REAL = 确定性、可解释、可复现的真实决策计算：输入真实识别的状态与
    胜率，按**显式文档化的数学规则**输出唯一动作（单点策略，频率=1.0，
    不伪造混合策略）。
  - 它 **不是 GTO**：不做平衡求解、不假装纳什均衡；model/notes/raw 中
    永远携带 `gto: False`。BASELINE（mock 启发式）语义保持不变，本引擎
    不是 mock 改名——两者并存，配置选择。

决策数学（全部复用冻结 domain 层，无再推导）：
  - required_equity = pot_odds(pot, to_call).required_equity   （domain 冻结公式）
  - SPR             = spr(effective_stack, pot)                 （domain 冻结公式）
  - call EV（近似、显式假设）: eq*(pot+2*to_call) - to_call
    （假设摊牌即止——保守下界，诚实标注 approx_ev）
  - fold EV = 0；激进动作（bet/raise/allin）EV=None（需对手弃牌率假设，
    无读数不假设——诚实省略）

规则表（阈值全部为显式常量，见各 RULE_* 常量；无随机数）：
  preflop:
    to_call==0 → 未加注局面确定性开注 raise 2.5bb(+1bb/额外limper)；
                 有效栈 ≤ 1.2×开注尺寸 → allin
    to_call>0  → margin = equity - required：
                 margin ≥ RAISE_MARGIN_PP(0.10) → raise 3×to_call（不足则 allin）
                 margin ≥ 0                     → call
                 margin ≥ DEFEND_MARGIN_PP(-0.05) 且 to_call_bb ≤ CHEAP_CALL_BB(1.0)
                                                → call（低价防守）
                 else                           → fold
  postflop:
    to_call==0 → equity ≥ VALUE_BET_EQ(0.62) → bet 0.66×pot；
                 equity ≥ CHECK_BET_EQ(0.50) 且 SPR ≥ MIN_SPR_STAB(3) → bet 0.33×pot；
                 else → check（无读数不做随机诈唬）
    to_call>0  → margin ≥ RAISE_MARGIN_PF(0.15) → raise 2.5×to_call（不足则 allin）
                 margin ≥ 0                     → call
                 margin ≥ DEFEND_MARGIN_PP(-0.05) 且 to_call_bb ≤
                     max(CHEAP_CALL_BB(1.0), DEFEND_STACK_PCT(0.04)×eff_bb) → call
                 else                           → fold

输入要求（不满足 → ProviderNotConfiguredError → 管线诚实降级为
BASELINE(mock) 并带 SOLVER_DEGRADED 警告——绝不静默冒充 REAL）：
  hole_cards 恰好 2 张、hero_equity 为数值、pot>0、effective_stack>0、
  bb>0、street ∈ {preflop,flop,turn,river}、game=="cash"（v1 仅现金桌）。
"""
from __future__ import annotations

from typing import Any

from ..errors import ProviderNotConfiguredError
from .solver import SolverAction, SolverProvider, SolverResult

ENGINE_NAME = "realengine"
ENGINE_MODEL = "realengine-rules-v1"

# ── 显式规则常量（可审计；修改即改变引擎语义——版本号随之升级） ──────────
VALUE_BET_EQ = 0.62          # postflop 无人下注时的价值下注胜率门
CHECK_BET_EQ = 0.50          # 次级下注胜率门（需 SPR 配合）
MIN_SPR_STAB = 3.0           # 次级小注下注的最低 SPR
RAISE_MARGIN_PP = 0.10       # preflop 面对下注转加注的 margin 门
RAISE_MARGIN_PF = 0.15       # postflop 面对下注转加注的 margin 门
DEFEND_MARGIN_PP = -0.05     # 低价防守允许的最大 margin 亏损
CHEAP_CALL_BB = 1.0          # 低价防守的绝对上限（bb）
DEFEND_STACK_PCT = 0.04      # 低价防守的相对上限（有效栈比例）
OPEN_RAISE_BB = 2.5          # preflop 未加注局面开注尺寸
OPEN_RAISE_PER_LIMP_BB = 1.0 # 每个额外 limper 的加价
OPEN_ALLIN_STACK_RATIO = 1.2 # eff ≤ 1.2×开注尺寸 → 直接全下
RAISE_TO_X = 3.0             # preflop 加注至 3×to_call
RAISE_TO_X_PF = 2.5          # postflop 加注至 2.5×to_call
VALUE_BET_FRAC = 0.66        # 价值下注占池比
SMALL_BET_FRAC = 0.33        # 次级下注占池比


def _f(v: Any) -> float | None:
    """严格数值化（bool 不算数——UNKNOWN/脏值一律拒绝）。"""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


class RealEngineSolverProvider(SolverProvider):
    """确定性规则/数学决策引擎（REAL；非 GTO、非 mock）。

    query 契约（由 SolverAgent 归一化）：{board, hole_cards, pot,
    effective_stack, street, to_call, hero_equity, bet_sizings, game, bb}。
    """

    name = ENGINE_NAME
    is_mock = False

    def is_configured(self) -> bool:
        return True          # 引擎自包含（无外部依赖）——始终可配置

    def status_reason(self) -> str:
        return ""

    # ------------------------------------------------------------------
    def _require(self, query: dict) -> dict:
        """输入门：任一关键输入缺失/非法 → ProviderNotConfiguredError
        （管线诚实降级 BASELINE——绝不伪造 REAL）。"""
        hole = [str(c) for c in (query.get("hole_cards") or [])]
        if len(hole) != 2:
            raise ProviderNotConfiguredError(
                f"{ENGINE_MODEL}: requires exactly 2 hole_cards, got {len(hole)}")
        equity = _f(query.get("hero_equity"))
        if equity is None or not (0.0 <= equity <= 1.0):
            raise ProviderNotConfiguredError(
                f"{ENGINE_MODEL}: requires numeric hero_equity in [0,1], "
                f"got {query.get('hero_equity')!r}")
        if query.get("hero_equity_source") != "computed":
            # 上游用 0.5 默认值伪造的胜率不接收——无真实计算 → 拒绝
            # （管线诚实降级 BASELINE）。
            raise ProviderNotConfiguredError(
                f"{ENGINE_MODEL}: requires computed hero_equity "
                f"(hero_equity_source != 'computed'; refusing fabricated "
                f"equity)")
        pot = _f(query.get("pot"))
        if pot is None or pot <= 0:
            raise ProviderNotConfiguredError(
                f"{ENGINE_MODEL}: requires pot > 0, got {query.get('pot')!r}")
        eff = _f(query.get("effective_stack"))
        if eff is None or eff <= 0:
            raise ProviderNotConfiguredError(
                f"{ENGINE_MODEL}: requires effective_stack > 0, got "
                f"{query.get('effective_stack')!r}")
        bb = _f(query.get("bb"))
        if bb is None or bb <= 0:
            raise ProviderNotConfiguredError(
                f"{ENGINE_MODEL}: requires bb > 0 (from state blinds), got "
                f"{query.get('bb')!r}")
        street = str(query.get("street") or "")
        if street not in ("preflop", "flop", "turn", "river"):
            raise ProviderNotConfiguredError(
                f"{ENGINE_MODEL}: unknown street {street!r}")
        to_call = _f(query.get("to_call"))
        if to_call is None or to_call < 0:
            raise ProviderNotConfiguredError(
                f"{ENGINE_MODEL}: requires to_call >= 0, got "
                f"{query.get('to_call')!r}")
        if str(query.get("game") or "") != "cash":
            raise ProviderNotConfiguredError(
                f"{ENGINE_MODEL}: v1 supports cash games only, got "
                f"{query.get('game')!r}")
        return {"hole": hole, "equity": equity, "pot": pot, "eff": eff,
                "bb": bb, "street": street, "to_call": to_call}

    # ------------------------------------------------------------------
    async def solve(self, query: dict) -> SolverResult:
        v = self._require(query)
        # 冻结 domain 层数学（与 Judge 同一公式——零再推导）
        from ...domain.poker import pot_odds as _pot_odds, spr as _spr

        equity = v["equity"]
        pot, eff, bb, to_call = v["pot"], v["eff"], v["bb"], v["to_call"]
        pot_bb = pot / bb
        eff_bb = eff / bb
        to_call_bb = to_call / bb
        req = _pot_odds(pot, to_call).required_equity   # 冻结公式
        margin = equity - req
        spr_v = _spr(eff, pot)

        action, size_bb, rule = self._decide(
            v["street"], to_call > 0, equity, req, margin,
            pot, pot_bb, eff, eff_bb, to_call, to_call_bb, spr_v)
        ev = self._ev(action, equity, pot, to_call)

        conf = self._confidence(margin if to_call > 0 else equity - 0.5)
        return SolverResult(
            solver_name=self.name,
            model=ENGINE_MODEL + " (deterministic rules; NOT GTO)",
            actions=[SolverAction(action, size_bb, 1.0, ev)],
            bet_sizings_bb=[SMALL_BET_FRAC, VALUE_BET_FRAC],
            notes=("REAL 确定性规则引擎：equity vs pot-odds(冻结公式) + SPR 规则；"
                   "单点策略（频率=1.0，不伪造混合）；非 GTO/非平衡求解；"
                   "无随机数。"),
            is_approximate=False,      # 非 mock 占位——真实计算（"非 GTO"见 model/notes/raw）
            raw={"engine": ENGINE_MODEL, "gto": False, "deterministic": True,
                 "rule": rule, "equity": round(equity, 4),
                 "required_equity": round(req, 4), "margin": round(margin, 4),
                 "spr": round(spr_v, 2), "pot_bb": round(pot_bb, 2),
                 "effective_stack_bb": round(eff_bb, 2),
                 "to_call_bb": round(to_call_bb, 3),
                 "action": action, "size": size_bb,
                 "inputs": {"street": v["street"], "hole_cards": v["hole"],
                            "game": str(query.get("game"))}},
        )

    # ------------------------------------------------------------------
    def _decide(self, street: str, facing_bet: bool, equity: float,
                req: float, margin: float, pot: float, pot_bb: float,
                eff: float, eff_bb: float, to_call: float, to_call_bb: float,
                spr_v: float,
                ) -> tuple[str, float | None, str]:
        """返回 (action, size_bb_field, rule_id)。size_bb 字段沿用既有
        约定：bet 为占池比（与 mock/judge 0.66 比较逻辑一致），raise 为
        加注至多少 bb。"""
        if street == "preflop":
            if not facing_bet:
                open_bb = OPEN_RAISE_BB + max(
                    0.0, pot_bb - 1.5) * OPEN_RAISE_PER_LIMP_BB
                if eff_bb <= open_bb * OPEN_ALLIN_STACK_RATIO:
                    return "allin", round(eff_bb, 2), "preflop_open_short_allin"
                return "raise", round(open_bb, 2), "preflop_open_raise"
            # facing bet preflop
            if margin >= RAISE_MARGIN_PP:
                raise_to = RAISE_TO_X * to_call_bb
                if raise_to >= eff_bb:
                    return "allin", round(eff_bb, 2), "preflop_raise_allin"
                return "raise", round(raise_to, 2), "preflop_raise"
            if margin >= 0:
                return "call", None, "preflop_call_pot_odds"
            if (margin >= DEFEND_MARGIN_PP
                    and to_call_bb <= CHEAP_CALL_BB):
                return "call", None, "preflop_call_cheap_defend"
            return "fold", None, "preflop_fold_pot_odds"
        # postflop
        if not facing_bet:
            if equity >= VALUE_BET_EQ:
                if VALUE_BET_FRAC * pot >= eff:
                    return "allin", round(eff_bb, 2), "postflop_value_allin"
                return "bet", VALUE_BET_FRAC, "postflop_value_bet"
            if (equity >= CHECK_BET_EQ
                    and spr_v >= MIN_SPR_STAB):
                if SMALL_BET_FRAC * pot >= eff:
                    return "allin", round(eff_bb, 2), "postflop_small_allin"
                return "bet", SMALL_BET_FRAC, "postflop_small_bet"
            return "check", None, "postflop_check_no_random_bluff"
        # facing bet postflop
        if margin >= RAISE_MARGIN_PF:
            raise_to = RAISE_TO_X_PF * to_call_bb
            if raise_to >= eff_bb:
                return "allin", round(eff_bb, 2), "postflop_raise_allin"
            return "raise", round(raise_to, 2), "postflop_raise"
        if margin >= 0:
            return "call", None, "postflop_call_pot_odds"
        if (margin >= DEFEND_MARGIN_PP
                and to_call_bb <= max(CHEAP_CALL_BB,
                                      DEFEND_STACK_PCT * eff_bb)):
            return "call", None, "postflop_call_cheap_defend"
        return "fold", None, "postflop_fold_pot_odds"

    # ------------------------------------------------------------------
    def _ev(self, action: str, equity: float, pot: float,
            to_call: float) -> float | None:
        """call EV 为显式假设下的近似（摊牌即止——保守下界）；fold=0；
        激进动作需对手弃牌率假设——无读数不假设，诚实省略（None）。"""
        if action == "call":
            return round(equity * (pot + 2 * to_call) - to_call, 3)
        if action == "fold":
            return 0.0
        return None

    # ------------------------------------------------------------------
    def _confidence(self, margin_ref: float) -> float:
        """确定性置信：margin 越大越高，夹在 [0.65, 0.85]。
        上限 0.85——规则引擎诚实不自称满置信。"""
        c = 0.65 + 0.5 * max(0.0, margin_ref)
        return round(min(0.85, c), 3)
