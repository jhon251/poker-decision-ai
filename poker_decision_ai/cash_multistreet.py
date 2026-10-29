"""V3G-4 — CASH Multi-Street Planning Layer（只读结构层；不做多街 EV 优化）。

定位
----
把 CASH 决策从"当前街单点状态"扩展为**可表达未来街道与行动线路的结构化
上下文**：

    current state → future street context → legal future transitions
                  → line representation

**本阶段不做真正的多街 EV 优化**；不建立策略强度主张。不得声称 GTO /
equilibrium / Solver / 多街策略提升 / benchmark 提升 / 收益提升。

硬约束
------
1. **不虚构未来牌面**：未来街的公共牌一律表示为 `board_state="unresolved"`
   + `unknown=["board"]`。**绝不**随机生成"代表牌"，也**不**做 runout 采样
   （采样属 `multistreet*.py` 的 benchmark 范围，本层不复用、不复制）。
2. **只读**：不修改输入 state；无 IO、无随机、无 wall-clock。
3. **UNKNOWN-first**：未知 street **不**当作 preflop；未知 board / hole cards /
   action history 一律如实标注，**不**默认 6-max / position / stack / street /
   board / future card；UNKNOWN 不转 EMPTY。
4. **CASH-only，fail closed**：非 CASH → `present=False`；不含 ICM / bounty /
   satellite / payout / tournament stack model；不从 MTT/PKO/Mystery 借用规则。
5. **不含 EV**：不调用 `compute_ev` / `evaluate_intel` / `multistreet*`；
   无 equity、无 runout、无 utility。
6. **不重算位置**：位置只从既有 state 标签读取；不调用 `positions_for`。
7. **V3G-1 语义不绕过**：`to_call > hero_stack` 被 V3G-1 判为冲突的既有行为
   **原样保留**（本阶段不修复、不绕过）；影响范围见 closeout。

复用而非复制（审计结论）
------------------------
* `resolve_mode` / V3G-2 适配层：模式判定与证据旁路（唯一口径）；
* `cash_preflop_decision.build_preflop_decision`：**可选**接入 Preflop 上下文；
* `vision2.schemas.STREET_RANK` 的排序语义在本层以**显式常量**表达
  （不 import vision2，避免视觉层耦合）；街道集合与既有实现一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cash_integrity_context import cash_integrity_context

MULTISTREET_LAYER_VERSION = "v3g4-cash-multistreet-layer-v1"

#: 街道推进（**确定性**；与既有 `_FUTURE`/`_FUTURE_STREETS` 同构）
STREET_ORDER: tuple[str, ...] = ("preflop", "flop", "turn", "river")
NEXT_STREET: dict[str, str] = {"preflop": "flop", "flop": "turn",
                               "turn": "river"}

#: 各街应有的公共牌张数
BOARD_SIZE: dict[str, int] = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}

#: 终结原因
TERMINAL_NONE = None
TERMINAL_RIVER = "river_no_future_street"
TERMINAL_FOLD = "fold"
TERMINAL_ALLIN = "all_in_no_future_decision"
TERMINAL_SHOWDOWN = "showdown"


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


def _board_of(state: dict) -> list:
    b = state.get("board")
    if not isinstance(b, list):
        return []
    return [c for c in b]


def _active_players(state: dict) -> list:
    out = []
    for p in state.get("players") or []:
        if isinstance(p, dict) and p.get("status") in ("active", "allin"):
            out.append(p)
    return out


# ── Transition ────────────────────────────────────────────────────────
def next_street_transition(state: dict | None) -> dict:
    """确定性街道转移与终结判定（只读；无随机、无未来牌）。

    返回字段：`current_street` / `next_street` / `terminal` / `terminal_reason`
    / `board_state` / `board_size` / `expected_board_size` / `unknown`
    / `transition_legal`。
    """
    if not isinstance(state, dict):
        return {"present": False, "reason": "state_not_a_dict"}

    street_raw = state.get("street")
    street = street_raw.lower() if isinstance(street_raw, str) else None
    unknown: list[str] = []

    if street is None or street not in STREET_ORDER:
        # UNKNOWN street：**不**当作 preflop、**不**猜测推进
        return {"present": False,
                "reason": f"unknown_street:{street_raw!r}",
                "current_street": None, "next_street": None,
                "terminal": True, "terminal_reason": "unknown_street",
                "board_state": "unknown", "board_size": None,
                "expected_board_size": None, "unknown": ["street"],
                "transition_legal": False}

    board = _board_of(state)
    exp = BOARD_SIZE[street]
    board_state = "known" if len(board) == exp else (
        "unresolved" if len(board) < exp else "conflict")
    if board_state != "known":
        unknown.append("board")

    nxt = NEXT_STREET.get(street)

    # ── 终结判定（优先序：展开面 < fold < all-in < river）──────────────
    terminal_reason = TERMINAL_NONE
    statuses = [p.get("status") for p in (state.get("players") or [])
                if isinstance(p, dict)]
    if statuses and all(s == "folded" for s in statuses):
        terminal_reason = TERMINAL_FOLD
    elif street == "river":
        terminal_reason = TERMINAL_RIVER
    elif _all_in_locked(state):
        terminal_reason = TERMINAL_ALLIN

    return {
        "present": True,
        "current_street": street,
        "next_street": (None if terminal_reason else nxt),
        "terminal": terminal_reason is not None,
        "terminal_reason": terminal_reason,
        "board_state": board_state,
        "board_size": len(board),
        "expected_board_size": exp,
        "unknown": unknown,
        "transition_legal": (terminal_reason is None and nxt is not None),
        # 未来公共牌**未解析**（绝不虚构）
        "future_board": {"state": "unresolved", "cards": None,
                         "must_not_invent": True},
    }


def _all_in_locked(state: dict) -> bool:
    """hero 已 all-in → 本街之后**无 hero 决策**（仍可能有 runout）。

    本函数只回答"hero 是否还能行动"，**不**回答"手牌是否终结"——
    后者见 `showdown_terminal()`。
    """
    hero = _hero(state)
    return bool(hero) and hero.get("status") == "allin"


def showdown_terminal(state: dict | None) -> dict:
    """Showdown / hand-terminal 判定（与街道推进解耦，可单独查询）。"""
    if not isinstance(state, dict):
        return {"present": False, "reason": "state_not_a_dict"}
    trans = next_street_transition(state)
    if not trans.get("present"):
        return {"present": False, "reason": trans.get("reason"),
                "showdown": None}
    street = trans["current_street"]
    board = _board_of(state)
    complete_board = len(board) == BOARD_SIZE["river"]
    statuses = [p.get("status") for p in (state.get("players") or [])
                if isinstance(p, dict)]
    everyone_folded = bool(statuses) and all(s == "folded" for s in statuses)
    return {
        "present": True,
        "street": street,
        "board_complete": complete_board,
        "everyone_folded": everyone_folded,
        # showdown 只在"牌面完整 + 未全员弃牌"时成立；否则 None（不猜）
        "showdown": (True if (complete_board and not everyone_folded)
                     else (False if everyone_folded else None)),
        "hand_terminal": bool(trans["terminal"] or complete_board
                              or everyone_folded),
    }


# ── Line representation ───────────────────────────────────────────────
def _line(state: dict) -> tuple[list, list]:
    """行动线路（结构化节点；不解析字符串日志）。

    节点字段：actor / street / action / amount / pot_before / stack_before /
    resulting_state / evidence。
    """
    hist = state.get("action_history")
    if not isinstance(hist, list):
        return [], ["action_history"]
    bb = _num(state.get("bb"))
    pot = _num(state.get("pot"))
    players = {str(p.get("player_id")): p
               for p in (state.get("players") or []) if isinstance(p, dict)}
    nodes = []
    for i, a in enumerate(hist):
        if not isinstance(a, dict):
            continue
        pid = a.get("player_id")
        pid_s = str(pid) if pid is not None else None
        p = players.get(pid_s) if pid_s else None
        amount = _num(a.get("amount"))
        stack_before = _num((p or {}).get("stack"))
        action = a.get("action")
        street = a.get("street")
        unknown = []
        if action is None:
            unknown.append("action")
        if street is None:
            unknown.append("street")
        if amount is None:
            unknown.append("amount")
        nodes.append({
            "sequence": i,
            "actor": pid_s,
            "street": street if not _is_blank(street) else None,
            "action": action,
            "amount": amount,
            "amount_bb": (round(amount / bb, 6)
                          if amount is not None and bb else None),
            "pot_before": pot,                 # 仅当前街 pot 已知；历史街未知
            "stack_before": stack_before,
            "position": (p or {}).get("position"),
            "resulting_state": {
                "note": "不推断历史结果状态（缺逐节点快照）",
                "derivable": False,
            },
            "evidence": {"status": "declared" if not unknown else "partial",
                         "unknown": unknown},
        })
    return nodes, []


# ── Context ───────────────────────────────────────────────────────────
@dataclass
class MultiStreetContext:
    version: str = MULTISTREET_LAYER_VERSION
    present: bool = False
    reason: str = ""
    mode: str | None = None
    ready: bool = False
    incomplete: list = field(default_factory=list)
    unknown: list = field(default_factory=list)
    current_street: str | None = None
    current_board: list = field(default_factory=list)
    board_state: str = "unknown"
    hero: dict = field(default_factory=dict)
    pot: float | None = None
    to_call: float | None = None
    effective_stack_bb: float | None = None
    legal_actions: list | None = None
    current_actor: str | None = None
    transition: dict = field(default_factory=dict)
    showdown: dict = field(default_factory=dict)
    line: list = field(default_factory=list)
    integrity: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version, "present": self.present,
            "reason": self.reason, "mode": self.mode, "ready": self.ready,
            "incomplete": list(self.incomplete), "unknown": list(self.unknown),
            "current_street": self.current_street,
            "current_board": list(self.current_board),
            "board_state": self.board_state,
            "hero": dict(self.hero),
            "pot": self.pot, "to_call": self.to_call,
            "effective_stack_bb": self.effective_stack_bb,
            "legal_actions": (None if self.legal_actions is None
                              else list(self.legal_actions)),
            "current_actor": self.current_actor,
            "transition": dict(self.transition),
            "showdown": dict(self.showdown),
            "line": list(self.line),
            "integrity": dict(self.integrity),
            "evidence": dict(self.evidence),
        }


def build_multistreet_context(state: dict | None,
                              preflop: dict | None = None) -> dict:
    """CASH 多街决策上下文 + 线路（只读；确定性；不含 EV）。

    * `present=False`：非 CASH / 输入非 dict / street 未知 —— fail closed。
    * `ready=False`：关键信息不足 —— `line` 仍可能部分可用，但 `incomplete`
      如实列出；**不制造**线路、**不虚构**未来牌。

    `preflop`（可选）：V3G-3 `build_preflop_decision()` 的输出，用于在
    preflop 复用其上下文与候选（**只读转述**，不重算）。
    """
    if not isinstance(state, dict):
        return {"present": False, "version": MULTISTREET_LAYER_VERSION,
                "reason": "state_not_a_dict"}

    ev = cash_integrity_context(state)
    if not ev.get("present"):
        return {"present": False, "version": MULTISTREET_LAYER_VERSION,
                "reason": ev.get("reason", "not_cash_mode")}

    trans = next_street_transition(state)
    if not trans.get("present"):
        # UNKNOWN street：**不**当作 preflop，也不制造线路
        return {"present": False, "version": MULTISTREET_LAYER_VERSION,
                "reason": trans.get("reason"),
                "unknown": trans.get("unknown", [])}

    rep = dict(ev.get("state_integrity") or {})
    ctx = MultiStreetContext(present=True, mode=ev.get("mode"),
                             current_street=trans["current_street"],
                             current_board=_board_of(state),
                             board_state=trans["board_state"],
                             transition=trans,
                             showdown=showdown_terminal(state),
                             integrity=rep, evidence=ev)

    hero = _hero(state) or {}
    bb = _num(state.get("bb"))
    stack = _num(hero.get("stack"))
    holes = hero.get("hole_cards")
    holes_known = (isinstance(holes, list) and len(holes) == 2
                   and all(not _is_blank(c) for c in holes))

    incomplete: list[str] = []
    unknown = list(trans.get("unknown") or [])
    if not holes_known:
        incomplete.append("hero.hole_cards")
        unknown.append("hero.hole_cards")
    if stack is None:
        incomplete.append("hero.stack")
        unknown.append("hero.stack")
    if bb is None:
        incomplete.append("bb")
    if not isinstance(state.get("action_history"), list):
        incomplete.append("action_history")
        unknown.append("action_history")
    if rep.get("level") == "INCOMPLETE":
        incomplete.append("state_integrity_incomplete")

    ctx.hero = {
        "player_id": hero.get("player_id"),
        "position": hero.get("position") if not _is_blank(
            hero.get("position")) else None,
        "seat": hero.get("seat"),
        "hole_cards": [str(c) for c in holes] if holes_known else None,
        "hole_cards_unknown": not holes_known,
        "stack": stack,
    }
    ctx.pot = _num(state.get("pot"))
    ctx.to_call = _num(state.get("to_call"))
    if stack is not None and bb:
        ctx.effective_stack_bb = round(stack / bb, 6)

    from .decision_contract import derive_legal_actions
    ctx.legal_actions = derive_legal_actions(
        state.get("to_call"), hero.get("stack"), state.get("pot"))

    # current actor：最后一个行动者之后应行动的人；无法可靠确定 → None（不猜）
    ctx.current_actor = _current_actor(state)

    line, line_missing = _line(state)
    if line_missing:
        incomplete.extend(x for x in line_missing if x not in incomplete)
    ctx.line = line

    # preflop 复用（只读转述）
    preflop_view = None
    if isinstance(preflop, dict) and preflop.get("present"):
        preflop_view = {
            "reused": True,
            "version": preflop.get("version"),
            "ready": (preflop.get("preflop_context") or {}).get("ready"),
            "candidate_count": len(preflop.get("candidates") or []),
            "note": "只读转述 V3G-3 结果；本层不重算候选",
        }
    elif preflop is not None:
        preflop_view = {"reused": False,
                        "reason": (preflop or {}).get("reason", "unavailable")}

    ctx.incomplete = list(dict.fromkeys(incomplete))
    ctx.unknown = list(dict.fromkeys(unknown))
    ctx.ready = not ctx.incomplete

    out = {"present": True, "version": MULTISTREET_LAYER_VERSION,
           "multistreet_context": ctx.to_dict()}
    if preflop_view is not None:
        out["preflop_reuse"] = preflop_view
    return out


def _current_actor(state: dict) -> str | None:
    """当前行动者：仅当 action history 与在座玩家足以**可靠**确定时给出。

    无法可靠确定 → None（**不猜**；不得默认 hero）。
    """
    hist = state.get("action_history")
    if not isinstance(hist, list) or not hist:
        return None
    last = hist[-1]
    if not isinstance(last, dict):
        return None
    if last.get("street") != state.get("street"):
        return None                      # 最后行动不属于当前街 → 不可靠
    actors = [str(a.get("player_id")) for a in hist
              if isinstance(a, dict) and a.get("player_id") is not None]
    if not actors:
        return None
    live = [str(p.get("player_id")) for p in _active_players(state)
            if p.get("player_id") is not None]
    if not live:
        return None
    last_actor = actors[-1]
    if last_actor not in live:
        return None
    # 座位序后的下一个在座玩家；座位缺失 → 不可靠 → None
    seats = {str(p.get("player_id")): p.get("seat")
             for p in _active_players(state)}
    if any(seats.get(pid) is None for pid in live) \
            or seats.get(last_actor) is None:
        return None
    ordered = sorted(live, key=lambda pid: seats[pid])
    idx = ordered.index(last_actor)
    return ordered[(idx + 1) % len(ordered)]


__all__ = ["MULTISTREET_LAYER_VERSION", "STREET_ORDER", "NEXT_STREET",
           "BOARD_SIZE", "MultiStreetContext", "next_street_transition",
           "showdown_terminal", "build_multistreet_context"]
