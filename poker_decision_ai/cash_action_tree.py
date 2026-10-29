"""V3G-7 — CASH Action Tree / Bet Sizing（只读结构层；不做选择、不改变决策）。

定位
----
建立 CASH-only 的结构化 **Action Tree / Bet Sizing** 层，回答三个问题：

  1. **当前节点有哪些合法动作**（legality，来自既有唯一口径）；
  2. **每个动作的候选尺寸**（sizing candidate，来源/单位/边界/证据齐全）；
  3. **动作之后的结构化 resulting context**（转移，供后续阶段消费）。

**本层不实现 preference/selection** —— 不排序、不打分、不推荐、不选尺寸。
选择仍由既有冻结 `compute_ev`（候选评分）与 Judge 负责，本层不参与。

复用而非重实现（审计结论）
--------------------------
* **合法性**：`decision_contract.derive_legal_actions` / `validate_action`
  / `LEGAL_ACTIONS` —— 唯一口径，**不重写**任何合法性规则；
* **尺寸约定**：`strategy_core.BET_FRACTIONS`（pot 分数）/ `RAISE_TO_MULTS`
  （raise-to 倍数）—— **既有冻结常数**，本层只读引用、**不新增常数**；
* 边界：`to_call` / `min_raise_to` / 有效筹码 全部来自 state，**不默认**。

单位与语义（必须明确，避免歧义）
--------------------------------
* `amount`：该动作要求 hero **投入的筹码**（state 的筹码单位，通常为 bb 面值）；
  * `check` / `fold` → 0；
  * `call` → `min(to_call, effective_stack)`；
  * `bet` → 本街下注额；
  * `raise` → **本街总投入（raise-to）**，非增量；
  * `allin` → `effective_stack`。
* `amount_bb`：`amount / bb`（单位 = 大盲）。
* `raise_to_bb`：**仅** raise 动作携带 = `amount_bb`（本街总投入）。
* `increment_bb`：**仅** raise 动作携带 = `to_call_bb` 之上的**增量**
  （`raise_to_bb − to_call_bb`）。
* `increment_bb` 与 `raise_to_bb` 的区别是**显式**的，避免把"加注到"当作"加注多少"。

硬约束
------
1. **只读 + 确定性**：不修改输入；**不 import `random`**、不采样 runout、
   **不生成未来牌**、无 IO、无 wall-clock；同输入 → 同输出。
2. **不生成非法或超筹码尺寸**：每个尺寸都过 `validate_action` 与
   `amount ≤ effective_stack` 双重检查；不合法者**不进入树**并记入 `rejected`。
3. **UNKNOWN-first**：`bb`/`pot`/`to_call`/hero stack 缺失 → 不猜；
   `min_raise_to` 缺失 → **不生成 raise 候选**（不凭空造 2× 规则）；
   **不默认** stack / table size / position / dealer。
4. **UNKNOWN ≠ EMPTY**：无法确定的尺寸类别标记为 `unknown`，不产出 0 值尺寸。
5. **不完整节点不伪造后续节点**：`complete=False` 时不产出 transitions。
6. **CASH-only，fail closed**：非 CASH → `present=False`；无 ICM / bounty /
   satellite / payout / tournament 逻辑。
7. **不绕过 V3G-1 语义**：`to_call > hero_stack` 的既有冲突判定**原样保留**
   （本阶段不修复、不绕过）；仅如实记录其影响。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cash_integrity_context import cash_integrity_context
from .decision_contract import LEGAL_ACTIONS, derive_legal_actions, validate_action
from .domain.poker.strategy_core import BET_FRACTIONS, RAISE_TO_MULTS

ACTION_TREE_VERSION = "v3g7-cash-action-tree-v1"

#: 尺寸类别（来源标签；**本层不排序、不推荐**）
SIZING_KINDS: tuple[str, ...] = (
    "fold_no_investment", "check_no_investment", "call_match_to_call",
    "call_allin_capped", "bet_pot_fraction", "bet_allin",
    "raise_min_raise_to", "raise_to_multiple", "raise_allin",
    "allin_forced",
)

#: 终结原因（与 V3G-4 街道转移语义一致）
TERMINAL_FOLD = "fold"
TERMINAL_ALLIN = "all_in"
TERMINAL_RIVER = "river"
TERMINAL_SHOWDOWN = "showdown"

_STREET_ORDER = ("preflop", "flop", "turn", "river")
_NEXT_STREET = {"preflop": "flop", "flop": "turn", "turn": "river"}


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


def _current_actor(state: dict) -> str | None:
    """当前行动者：仅当"最后行动属当前街 + 座位序完整"时给出，否则 None。"""
    hist = state.get("action_history")
    if not isinstance(hist, list) or not hist:
        return None
    last = hist[-1]
    if not isinstance(last, dict) or last.get("street") != state.get("street"):
        return None
    live = [p for p in state.get("players") or []
            if isinstance(p, dict) and p.get("status") in ("active", "allin")]
    ids = [(str(p.get("player_id")), p.get("seat")) for p in live
           if p.get("player_id") is not None]
    if not ids or any(s is None for _, s in ids):
        return None
    last_actor = last.get("player_id")
    if last_actor is None:
        return None
    ordered = [pid for pid, _ in sorted(ids, key=lambda x: x[1])]
    if str(last_actor) not in ordered:
        return None
    i = ordered.index(str(last_actor))
    return ordered[(i + 1) % len(ordered)]


def _terminal_reason(state: dict) -> str | None:
    street = state.get("street")
    statuses = [p.get("status") for p in (state.get("players") or [])
                if isinstance(p, dict)]
    if statuses and all(s == "folded" for s in statuses):
        return TERMINAL_FOLD
    hero = _hero(state)
    if hero and hero.get("status") == "allin":
        return TERMINAL_ALLIN
    board = state.get("board")
    n_board = len(board) if isinstance(board, list) else 0
    if street == "river" and n_board == 5:
        return TERMINAL_SHOWDOWN
    if street == "river":
        return TERMINAL_RIVER
    return None


# ── Sizing candidates ─────────────────────────────────────────────────
def _sizing_candidates(state: dict, legal: list, hero_stack,
                       bb: float, pot: float, to_call: float,
                       min_raise_to) -> tuple[list, list]:
    """合法尺寸候选（来源/单位/边界/证据齐全）；返回 (candidates, rejected)。"""
    cands: list[dict] = []
    rejected: list[str] = []
    eff = _num(hero_stack)
    if eff is None:
        return [], ["hero_stack_unknown"]
    eff_bb = eff / bb
    pot_bb = pot / bb
    tc_bb = max(to_call, 0.0) / bb

    def _emit(action: str, amount_bb: float, kind: str, source: str, *,
              raise_to_bb: float | None = None, evidence: str = "") -> None:
        ok, why = validate_action(action, legal)
        if not ok:
            rejected.append(f"illegal:{action}:{why}")
            return
        if amount_bb < 0:
            rejected.append(f"negative_amount:{action}")
            return
        if amount_bb > eff_bb + 1e-9:
            rejected.append(f"exceeds_effective_stack:{action}:{amount_bb}")
            return
        entry = {
            "action": action, "sizing_kind": kind, "source": source,
            "amount_bb": round(amount_bb, 6),
            "amount": round(amount_bb * bb, 6),
            "unit": {"amount": "chips", "amount_bb": "bb"},
            "bounds": {"effective_stack_bb": round(eff_bb, 6),
                       "to_call_bb": round(tc_bb, 6),
                       "pot_bb": round(pot_bb, 6),
                       "min_raise_to_bb": (round(min_raise_to / bb, 6)
                                           if min_raise_to is not None else None)},
            "allin": abs(amount_bb - eff_bb) <= 1e-9,
            "evidence": evidence or f"source={source}",
        }
        if action == "raise":
            if raise_to_bb is None:
                rejected.append("raise_without_raise_to")
                return
            entry["raise_to_bb"] = round(raise_to_bb, 6)
            entry["increment_bb"] = round(raise_to_bb - tc_bb, 6)
        cands.append(entry)

    if "fold" in legal:
        _emit("fold", 0.0, "fold_no_investment", "no_investment")
    if "check" in legal:
        _emit("check", 0.0, "check_no_investment", "no_investment")
    if "call" in legal:
        amt = min(tc_bb, eff_bb)
        kind = ("call_allin_capped" if tc_bb >= eff_bb - 1e-9
                else "call_match_to_call")
        _emit("call", amt, kind, "min(to_call, effective_stack)")
    if "bet" in legal:
        seen: set[float] = set()
        for frac in BET_FRACTIONS:
            amt = min(pot_bb * frac, eff_bb)
            key = round(amt, 6)
            if key in seen or amt <= 0:
                continue
            seen.add(key)
            _emit("bet", amt, "bet_pot_fraction",
                  f"frozen_BET_FRACTIONS:{frac}",
                  evidence=f"pot_bb({round(pot_bb,4)})×{frac}，封顶有效筹码")
        if round(eff_bb, 6) not in seen:
            _emit("bet", eff_bb, "bet_allin", "effective_stack")
    if "raise" in legal:
        if min_raise_to is None:
            rejected.append("min_raise_to_unknown:no_raise_candidates")
        else:
            mrt_bb = min_raise_to / bb
            # 注意：`raise ∈ legal` 已蕴含 `to_call < hero_stack`（见
            # `derive_legal_actions`），故此处无需处理"栈不足以跟注"的退化情形。
            if mrt_bb <= tc_bb + 1e-9:
                rejected.append(f"min_raise_to_not_a_raise:{round(mrt_bb,6)}")
            elif mrt_bb > eff_bb + 1e-9:
                # 最小合法加注至**高于有效栈** → 唯一可能的 raise（全下）
                # 低于最小加注 → **不构成合法加注**，拒绝
                rejected.append(
                    f"raise_below_min_raise_to:{round(eff_bb,6)}")
            else:
                _emit("raise", min(mrt_bb, eff_bb), "raise_min_raise_to",
                      "state.min_raise_to",
                      raise_to_bb=min(mrt_bb, eff_bb),
                      evidence="最小合法加注至（来自 state.min_raise_to）")
                seen_r = {round(min(mrt_bb, eff_bb), 6)}
                for mult in RAISE_TO_MULTS:
                    rt = min(tc_bb * mult, eff_bb)
                    key = round(rt, 6)
                    if key in seen_r or rt <= tc_bb + 1e-9:
                        continue
                    if rt < mrt_bb - 1e-9:
                        rejected.append(
                            f"raise_below_min_raise_to:{round(rt,6)}")
                        continue
                    seen_r.add(key)
                    _emit("raise", rt, "raise_to_multiple",
                          f"frozen_RAISE_TO_MULTS:{mult}",
                          raise_to_bb=rt,
                          evidence=f"to_call_bb×{mult}，封顶有效筹码")
                if round(eff_bb, 6) not in seen_r:
                    _emit("raise", eff_bb, "raise_allin", "effective_stack",
                          raise_to_bb=eff_bb)
    if "allin" in legal and "bet" not in legal and "raise" not in legal:
        _emit("allin", eff_bb, "allin_forced", "forced_all_in")
    return cands, rejected


# ── Action Tree ───────────────────────────────────────────────────────
@dataclass
class ActionNode:
    version: str = ACTION_TREE_VERSION
    present: bool = False
    reason: str = ""
    mode: str | None = None
    complete: bool = False
    street: str | None = None
    next_street: str | None = None
    current_actor: str | None = None
    pot: float | None = None
    pot_bb: float | None = None
    to_call: float | None = None
    to_call_bb: float | None = None
    hero_stack: float | None = None
    effective_stack_bb: float | None = None
    min_raise_to: float | None = None
    min_raise_to_bb: float | None = None
    bb: float | None = None
    legal_actions: list | None = None
    action_history: list = field(default_factory=list)
    terminal: bool = False
    terminal_reason: str | None = None
    sizing: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    unknown: list = field(default_factory=list)
    transitions: list = field(default_factory=list)
    integrity: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version, "present": self.present,
            "reason": self.reason, "mode": self.mode,
            "complete": self.complete, "street": self.street,
            "next_street": self.next_street,
            "current_actor": self.current_actor,
            "pot": self.pot, "pot_bb": self.pot_bb,
            "to_call": self.to_call, "to_call_bb": self.to_call_bb,
            "hero_stack": self.hero_stack,
            "effective_stack_bb": self.effective_stack_bb,
            "min_raise_to": self.min_raise_to,
            "min_raise_to_bb": self.min_raise_to_bb, "bb": self.bb,
            "legal_actions": (None if self.legal_actions is None
                              else list(self.legal_actions)),
            "action_history": list(self.action_history),
            "terminal": self.terminal, "terminal_reason": self.terminal_reason,
            "sizing": list(self.sizing), "rejected": list(self.rejected),
            "unknown": list(self.unknown),
            "transitions": list(self.transitions),
            "integrity": dict(self.integrity),
            "evidence": dict(self.evidence),
        }


def _transition(entry: dict, node: ActionNode, state: dict) -> dict:
    """action → resulting decision context（结构化；**不推断**未知量）。"""
    act = entry["action"]
    amt = entry["amount"]
    pot = node.pot
    eff = node.hero_stack
    new_pot = (pot + amt) if (pot is not None and amt is not None) else None
    new_eff = (eff - amt) if (eff is not None and amt is not None) else None
    terminal_reason = None
    if act == "fold":
        terminal_reason = TERMINAL_FOLD
    elif entry.get("allin"):
        terminal_reason = TERMINAL_ALLIN
    return {
        "action": act,
        "sizing_kind": entry["sizing_kind"],
        "street": node.street,
        "pot_after": (round(new_pot, 6) if new_pot is not None else None),
        "hero_stack_after": (round(new_eff, 6) if new_eff is not None else None),
        "to_call_after": None if act in ("check", "fold") else 0.0,
        "next_street": (None if terminal_reason else node.next_street),
        "terminal": terminal_reason is not None,
        "terminal_reason": terminal_reason,
        "becomes_allin": bool(entry.get("allin")),
        "derivable": (pot is not None and amt is not None and eff is not None),
        "note": ("pots/stack 为**算术结果**；对手响应与未来牌不推断、不生成"),
    }


def build_action_tree(state: dict | None) -> dict:
    """CASH Action Tree + Bet Sizing（只读；确定性；不做选择）。

    * `present=False`：非 CASH / 输入非 dict / street 未知 —— fail closed；
    * `complete=False`：必需字段缺失 —— **不产出 transitions**（不伪造后续节点）。
    """
    if not isinstance(state, dict):
        return {"present": False, "version": ACTION_TREE_VERSION,
                "reason": "state_not_a_dict"}

    ev = cash_integrity_context(state)
    if not ev.get("present"):
        return {"present": False, "version": ACTION_TREE_VERSION,
                "reason": ev.get("reason", "not_cash_mode")}

    street_raw = state.get("street")
    street = street_raw.lower() if isinstance(street_raw, str) else None
    if street is None or street not in _STREET_ORDER:
        return {"present": False, "version": ACTION_TREE_VERSION,
                "reason": f"unknown_street:{street_raw!r}",
                "unknown": ["street"]}

    node = ActionNode(present=True, mode=ev.get("mode"), street=street,
                      next_street=_NEXT_STREET.get(street),
                      integrity=dict(ev.get("state_integrity") or {}),
                      evidence=ev)
    hero = _hero(state) or {}
    hero_stack = _num(hero.get("stack"))
    bb = _num(state.get("bb"))
    pot = _num(state.get("pot"))
    to_call = _num(state.get("to_call"))
    mrt = _num(state.get("min_raise_to"))

    unknown: list[str] = []
    for name, val in (("bb", bb), ("pot", pot), ("to_call", to_call),
                      ("hero.stack", hero_stack)):
        if val is None:
            unknown.append(name)
    if mrt is None:
        unknown.append("min_raise_to")
    if not isinstance(state.get("action_history"), list):
        unknown.append("action_history")
    node.unknown = unknown

    node.hero_stack = hero_stack
    node.bb = bb
    node.pot = pot
    node.to_call = to_call
    node.min_raise_to = mrt
    if bb:
        node.pot_bb = round(pot / bb, 6) if pot is not None else None
        node.to_call_bb = round(to_call / bb, 6) if to_call is not None else None
        node.effective_stack_bb = (round(hero_stack / bb, 6)
                                   if hero_stack is not None else None)
        node.min_raise_to_bb = round(mrt / bb, 6) if mrt is not None else None

    hist = state.get("action_history")
    node.action_history = ([{"street": a.get("street"), "actor": a.get("player_id"),
                             "action": a.get("action"), "amount": a.get("amount")}
                            for a in hist if isinstance(a, dict)]
                           if isinstance(hist, list) else [])

    node.legal_actions = derive_legal_actions(to_call, hero.get("stack"),
                                              state.get("pot"))
    node.current_actor = _current_actor(state)
    reason = _terminal_reason(state)
    node.terminal = reason is not None
    node.terminal_reason = reason

    # 必需字段齐备性（决定是否产出尺寸与转移）
    required_ok = (bb is not None and pot is not None and to_call is not None
                   and hero_stack is not None)
    node.complete = bool(required_ok and node.legal_actions is not None)
    if not node.complete:
        if node.legal_actions is None:
            node.unknown = list(dict.fromkeys(node.unknown + ["legal_actions"]))

    if node.complete:
        cands, rejected = _sizing_candidates(state, node.legal_actions,
                                             hero.get("stack"), bb, pot,
                                             to_call, mrt)
        node.sizing = cands
        node.rejected = rejected
        if not node.terminal:
            node.transitions = [_transition(c, node, state) for c in cands]
        else:
            node.transitions = []

    return node.to_dict()


__all__ = ["ACTION_TREE_VERSION", "SIZING_KINDS", "TERMINAL_FOLD",
           "TERMINAL_ALLIN", "TERMINAL_RIVER", "TERMINAL_SHOWDOWN",
           "ActionNode", "build_action_tree"]
