"""V3G-3 — CASH Preflop Decision Layer（只读；结构与合法性层，不含新策略）。

定位（本阶段**只建立正确的决策结构**，不做策略优化、不追求 benchmark 提升）
---------------------------------------------------------------------
为 CASH **Preflop** 提供两件东西，且**都不改变任何既有决策数值**：

1. **Preflop 决策上下文**（`preflop_context`）：模式、准备就绪判定、位置标签、
   有效筹码、blinds、在座玩家、**结构化行动历史**（actor/action/amount/street/
   sequence/position/effective_stack/evidence）、以及 V3G-1/V3G-2 证据；
2. **Preflop 候选动作层**（`candidates`）：明确分离
   **合法性（legality）** 与 **策略偏好（preference）** —— 合法性过滤**先于**
   任何评分，且本层**不产生**偏好（偏好只能来自既有冻结 `compute_ev`）。

复用而非重实现（审计结论）
--------------------------
已存在、本模块**直接复用**（不复制逻辑）：
  * `decision_contract.LEGAL_ACTIONS` / `derive_legal_actions` /
    `validate_action` —— 合法动作派生与判定（唯一口径）；
  * `strategy_mode_switch.resolve_mode` —— 模式解析（唯一口径）；
  * `cash_state_integrity.evaluate_cash_state_integrity` —— V3G-1 门禁；
  * `cash_integrity_context.cash_integrity_context` —— V3G-2 证据旁路；
  * `strategy_core.BET_FRACTIONS` / `RAISE_TO_MULTS` —— **既有**尺寸约定常数
    （只读引用，**不修改**；本层不新增尺寸常数）。

硬约束
------
1. **只读**：不修改输入 state（含嵌套）。
2. **不重算 seat position**：位置只从 state 既有标签（`players[].position`）
   读取；**绝不**调用或修改 `positions_for`，绝不由 seat/button 反推。
3. **UNKNOWN-first**：hero 手牌 / 位置 / 筹码任一不可用 → `ready=False` +
   候选为空 + `incomplete` 说明；**不猜测**任何具体值（不默认 6-max、
   不默认 position、不默认 stack、不默认 action、不默认 dealer/button）。
4. **不产生策略偏好**：`candidates[].preference` 恒为
   `{"present": False, "source": "frozen_reference", ...}`——
   即"偏好来源只能是既有冻结参照"，本层**不发明**新策略，也不伪装 GTO。
5. **门禁只经 V3G-2 适配层到达**：本层**不直接** import
   `cash_state_integrity`，门禁结果从 `cash_integrity_context()` 的
   `state_integrity` 字段读取（保持 V3G-2 建立的唯一边界）。
6. **CASH-only，fail closed**：非 cash 模式返回 `present=False`，
   **不**引入 ICM / bounty / satellite / MTT stack-pressure / PKO / Mystery。
7. **不含 EV**：不调用 `compute_ev` / `evaluate_intel`；无 equity、无胜率。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cash_integrity_context import EVIDENCE_KIND, cash_integrity_context
from .decision_contract import derive_legal_actions, validate_action
from .domain.poker.strategy_core import (BET_FRACTIONS,  # noqa: F401 (只读引用)
                                         RAISE_TO_MULTS)

PREFLOP_LAYER_VERSION = "v3g3-cash-preflop-layer-v1"

#: 本层拒绝的状态（不得猜测）：缺任一 → not ready
REQUIRED_FOR_CANDIDATES = ("hero.hole_cards", "hero.position", "hero.stack")

#: 有效筹码深度分层的上界（bb）——**展示性分层**，非策略阈值
STACK_DEPTH_BANDS = ((0.0, 15.0, "short"), (15.0, 40.0, "medium"),
                     (40.0, 100.0, "deep"), (100.0, 1e18, "very_deep"))


def _num(v: Any) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _is_blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _hero(state: dict) -> dict | None:
    players = state.get("players")
    if not isinstance(players, list):
        return None
    hid = state.get("hero_id")
    for p in players:
        if isinstance(p, dict) and hid is not None and p.get("player_id") == hid:
            return p
    for p in players:
        if isinstance(p, dict) and p.get("is_hero"):
            return p
    return None


def _depth_band(bb_depth: float | None) -> str | None:
    if bb_depth is None:
        return None
    for lo, hi, name in STACK_DEPTH_BANDS:
        if lo <= bb_depth < hi:
            return name
    return None


@dataclass
class PreflopContext:
    """CASH Preflop 决策上下文（纯数据；无策略量）。"""

    version: str = PREFLOP_LAYER_VERSION
    present: bool = False
    reason: str = ""
    mode: str | None = None
    street: str | None = None
    ready: bool = False
    incomplete: list = field(default_factory=list)
    hero: dict = field(default_factory=dict)
    effective_stack_bb: float | None = None
    stack_depth_band: str | None = None
    blinds: dict = field(default_factory=dict)
    seats_in_hand: list = field(default_factory=list)
    action_history: list = field(default_factory=list)
    integrity: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)
    legal_actions: list | None = None

    def to_dict(self) -> dict:
        return {
            "version": self.version, "present": self.present,
            "reason": self.reason, "mode": self.mode, "street": self.street,
            "ready": self.ready, "incomplete": list(self.incomplete),
            "hero": dict(self.hero),
            "effective_stack_bb": self.effective_stack_bb,
            "stack_depth_band": self.stack_depth_band,
            "blinds": dict(self.blinds),
            "seats_in_hand": list(self.seats_in_hand),
            "action_history": list(self.action_history),
            "integrity": dict(self.integrity),
            "evidence": dict(self.evidence),
            "legal_actions": (None if self.legal_actions is None
                              else list(self.legal_actions)),
        }


# ── 结构化行动历史（供 G5/G6/G4 消费；不压缩成单一分数）────────────────
def _action_history(state: dict, bb: float | None) -> list:
    """行动历史 → 结构化条目（保留 actor/action/amount/street/sequence/
    position/effective_stack/evidence）。不猜测缺失字段。
    """
    hist = state.get("action_history")
    if not isinstance(hist, list):
        return []
    pos_of = {}
    stack_of = {}
    players = state.get("players")
    if isinstance(players, list):
        for p in players:
            if isinstance(p, dict) and p.get("player_id") is not None:
                pos_of[str(p["player_id"])] = p.get("position")
                stack_of[str(p["player_id"])] = _num(p.get("stack"))
    out = []
    for i, a in enumerate(hist):
        if not isinstance(a, dict):
            continue
        pid = a.get("player_id")
        pid_s = str(pid) if pid is not None else None
        amount = _num(a.get("amount"))
        stk = stack_of.get(pid_s) if pid_s else None
        out.append({
            "sequence": i,
            "actor": pid_s,
            "action": a.get("action"),
            "amount": amount,
            "amount_bb": (round(amount / bb, 6)
                          if amount is not None and bb else None),
            "street": a.get("street"),
            "position": pos_of.get(pid_s) if pid_s else None,
            "effective_stack": stk,
            "effective_stack_bb": (round(stk / bb, 6)
                                   if stk is not None and bb else None),
            "evidence": ("declared" if a.get("action") is not None
                         else "unknown"),
        })
    return out


def _seats_in_hand(state: dict) -> tuple[list, list]:
    """在座玩家（不含已 fold）与"已 fold 者被要求行动"的冲突。"""
    players = state.get("players")
    if not isinstance(players, list):
        return [], []
    seats = []
    folded_acting = []
    hist = state.get("action_history")
    acted = set()
    if isinstance(hist, list):
        for a in hist:
            if isinstance(a, dict) and a.get("player_id") is not None:
                acted.add(str(a["player_id"]))
    for p in players:
        if not isinstance(p, dict):
            continue
        pid = p.get("player_id")
        pid_s = str(pid) if pid is not None else None
        status = p.get("status")
        if status == "folded":
            # 已 fold 者不得再出现在后续行动中
            if pid_s is not None and pid_s in acted:
                later = [a for a in (hist or [])
                         if isinstance(a, dict)
                         and str(a.get("player_id")) == pid_s
                         and a.get("action") not in (None, "fold")]
                if later:
                    folded_acting.append(pid_s)
            continue
        seats.append({"player_id": pid_s, "seat": p.get("seat"),
                      "position": p.get("position"), "status": status})
    return seats, folded_acting


# ── Preflop 候选动作层（legality 先于 preference）──────────────────────
def _candidates(state: dict, ctx: PreflopContext) -> tuple[list, list]:
    """生成 preflop 候选；返回 (candidates, problems)。

    合法性来自 `derive_legal_actions`（唯一口径）；raise 金额必须**明确**且
    满足 min-raise-to 与筹码上限；无法确定 min-raise-to → 不生成 raise 候选
    （UNKNOWN，不猜测）。
    """
    problems: list[str] = []
    bb = _num(state.get("bb"))
    hero = _hero(state) or {}
    stack = _num(hero.get("stack"))
    pot = _num(state.get("pot"))
    to_call = _num(state.get("to_call"))
    legal = ctx.legal_actions
    if legal is None or bb is None or stack is None or pot is None \
            or to_call is None:
        return [], ["missing_numeric_inputs_for_candidates"]

    eff_bb = stack / bb
    tc_bb = max(to_call, 0.0) / bb
    pot_bb = pot / bb
    out: list[dict] = []

    def _pref() -> dict:
        # 本层**不产生**偏好：偏好只能来自既有冻结参照
        return {"present": False, "source": "frozen_reference",
                "note": "preference 由既有冻结 compute_ev 给出；本层不评分"}

    def _add(action: str, amount_bb: float | None, rule: str, *,
             forced_allin: bool = False, extra: dict | None = None) -> None:
        ok, why = validate_action(action, legal)
        if not ok:
            problems.append(f"illegal_candidate:{action}:{why}")
            return
        if action in ("check", "fold"):
            if amount_bb not in (None, 0.0):
                problems.append(f"amount_on_{action}")
                return
        if action == "call":
            amount_bb = min(tc_bb, eff_bb)
        if action in ("bet", "raise", "allin") and amount_bb is None:
            problems.append(f"missing_amount:{action}")
            return
        if amount_bb is not None and amount_bb > eff_bb + 1e-9:
            problems.append(f"amount_exceeds_effective_stack:{action}")
            return
        entry = {"action": action, "amount_bb": amount_bb,
                 "sizing_rule": rule,
                 "allin": bool(forced_allin)
                 or (amount_bb is not None
                     and abs(amount_bb - eff_bb) <= 1e-9),
                 "legality": {"ok": True, "legal_actions": list(legal)},
                 "preference": _pref()}
        if extra:
            entry.update(extra)
        out.append(entry)

    if "fold" in legal:
        _add("fold", None, "no_investment")
    if "check" in legal:
        _add("check", 0.0, "no_investment")
    if "call" in legal:
        rule = ("allin_call" if tc_bb >= eff_bb - 1e-9 else "match_to_call")
        _add("call", None, rule)
    if "allin" in legal and "bet" not in legal and "raise" not in legal:
        _add("allin", eff_bb, "forced_allin", forced_allin=True)
    if "bet" in legal:
        # bet = 开注；尺寸引用**既有冻结** BET_FRACTIONS（不新增常数）
        seen: set[float] = set()
        for frac in BET_FRACTIONS:
            s = min(pot_bb * frac, eff_bb)
            key = round(s, 6)
            if key in seen or s <= 0:
                continue
            seen.add(key)
            _add("bet", s, f"frozen_BET_FRACTIONS:{frac}")
        if round(eff_bb, 6) not in seen:
            _add("bet", eff_bb, "allin_bet", forced_allin=True)
    if "raise" in legal:
        mrt_raw = _num(state.get("min_raise_to"))
        if mrt_raw is None:
            # UNKNOWN：不猜测 min-raise-to（禁止凭空造 2x 规则）
            problems.append("min_raise_to_unknown_no_raise_candidates")
        else:
            mrt_bb = mrt_raw / bb
            seen_r: set[float] = set()
            # 尺寸约定取自**既有冻结** RAISE_TO_MULTS；**低于最小加注的目标一律
            # 拒绝并记录**（不做"吸附到 min_raise_to"的静默纠正——min_raise_to
            # 本身已作为独立目标在下方加入）。
            targets = [tc_bb * m for m in RAISE_TO_MULTS]
            targets.append(mrt_bb)
            targets.append(eff_bb)
            for t in targets:
                r = min(t, eff_bb)
                if r < mrt_bb - 1e-9:            # 低于最小加注 → 非法，拒绝
                    problems.append(f"raise_below_min_raise_to:{round(r, 6)}")
                    continue
                if r <= tc_bb + 1e-9:            # 不是真正的加注
                    continue
                key = round(r, 6)
                if key in seen_r:
                    continue
                seen_r.add(key)
                _add("raise", r, f"min_raise_aware:{round(mrt_bb, 6)}",
                     extra={"raise_to_bb": r})
    return out, problems


def build_preflop_decision(state: dict | None) -> dict:
    """CASH Preflop 决策结构与候选（只读；确定性；不含策略）。

    返回:
      {"present": bool, "version", "reason"?,
       "preflop_context": {...}, "candidates": [...], "problems": [...]}

    * `present=False`：非 CASH / 非 preflop / 输入非 dict —— **fail closed**。
    * `preflop_context.ready=False`：关键信息不足 —— 候选为空，
      UNKNOWN 如实列出，**不猜测**。
    """
    if not isinstance(state, dict):
        return {"present": False, "version": PREFLOP_LAYER_VERSION,
                "reason": "state_not_a_dict"}

    ev = cash_integrity_context(state)          # V3G-2 证据旁路（含模式判定）
    if not ev.get("present"):
        return {"present": False, "version": PREFLOP_LAYER_VERSION,
                "reason": ev.get("reason", "not_cash_mode")}

    street = state.get("street")
    if not (isinstance(street, str) and street.lower() == "preflop"):
        return {"present": False, "version": PREFLOP_LAYER_VERSION,
                "reason": f"not_preflop:{street!r}"}

    # 门禁结果**经 V3G-2 适配层**取得（唯一合法入口，不直接 import 门禁）
    rep = dict(ev.get("state_integrity") or {})
    ctx = PreflopContext(present=True, mode=ev.get("mode"),
                         street="preflop",
                         integrity=rep, evidence=ev)

    bb = _num(state.get("bb"))
    hero = _hero(state) or {}
    stack = _num(hero.get("stack"))
    hero_pos = hero.get("position")
    holes = hero.get("hole_cards")
    holes_known = (isinstance(holes, list) and len(holes) == 2
                   and all(not _is_blank(c) for c in holes))

    incomplete: list[str] = []
    if not holes_known:
        incomplete.append("hero.hole_cards")
    if _is_blank(hero_pos):
        incomplete.append("hero.position")
    if stack is None:
        incomplete.append("hero.stack")
    if bb is None:
        incomplete.append("bb")
    if _num(state.get("to_call")) is None:
        incomplete.append("to_call")
    if rep.get("level") == "INCOMPLETE":
        incomplete.append("state_integrity_incomplete")

    ctx.hero = {
        "player_id": hero.get("player_id"),
        "position": hero_pos if not _is_blank(hero_pos) else None,
        "seat": hero.get("seat"),
        "hole_cards": ([str(c) for c in holes] if holes_known else None),
        "hole_cards_unknown": not holes_known,
        "stack": stack,
    }
    if bb is not None:
        ctx.blinds = {"sb": _num(state.get("sb")), "bb": bb,
                      "ante": _num(state.get("ante"))}
    if stack is not None and bb:
        ctx.effective_stack_bb = round(stack / bb, 6)
        ctx.stack_depth_band = _depth_band(ctx.effective_stack_bb)

    seats, folded_acting = _seats_in_hand(state)
    ctx.seats_in_hand = seats
    ctx.action_history = _action_history(state, bb)

    ctx.legal_actions = derive_legal_actions(
        state.get("to_call"), hero.get("stack"), state.get("pot"))

    ctx.incomplete = list(incomplete)
    ctx.ready = not incomplete and ctx.legal_actions is not None

    problems = [f"folded_player_acted:{p}" for p in folded_acting]
    candidates: list = []
    if ctx.ready:
        candidates, cand_problems = _candidates(state, ctx)
        problems.extend(cand_problems)

    return {"present": True, "version": PREFLOP_LAYER_VERSION,
            "preflop_context": ctx.to_dict(),
            "candidates": candidates, "problems": problems}


__all__ = ["PREFLOP_LAYER_VERSION", "REQUIRED_FOR_CANDIDATES",
           "STACK_DEPTH_BANDS", "PreflopContext", "build_preflop_decision"]
