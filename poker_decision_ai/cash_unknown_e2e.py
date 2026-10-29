"""V3G-8 — CASH UNKNOWN End-to-End（只读聚合层；只分类，不填充）。

定位
----
把 CASH 决策链各层的 UNKNOWN 证据**聚合**成一份端到端可观察的报告，用于验证
UNKNOWN 不会在中途被静默转换成默认值 / EMPTY / 0 / 假造内容。

    state → (V3G-1 gate) → (V3G-2 adapter) → (V3G-3/4/5/6/7 各层) → 本聚合器
          → 统一 {present, complete, unknown[], incomplete[], conflicts[],
                  evidence/provenance, reason, fields{...}}

**本层只做分类与传播**（严格只读）：

* **绝不填充**任何字段、**绝不**发明数值、**绝不**生成候选/尺寸/未来牌/
  对手行动/范围/置信；
* 无法判定的字段标为 `unknown`（附来源），**不写 0、不写默认值**。

三类明确区分（授权核心）
------------------------
* `present` —— 有真实观测证据（附 `evidence`）；
* `unknown` —— 证据缺失或无法确认：**未知**，不得当作"空/否/0"；
* `empty`   —— 证据**确认**为"不存在"（如显式 `folded`、显式空 board 在
  preflop、显式 `occupied=False`）。

`unknown` 与 `empty` 的差别即"没有证据"与"证据说明没有"。二者都**不等于**
默认值；本报告用 `classification` 字段把三者（present/unknown/empty）与
`defaulted`（本层保证恒为 `False`）分开记录。

硬约束
------
1. **只读 + 确定性**：不修改输入；无 IO、**无 random**、无 wall-clock；
   同输入 → 同输出。
2. **不接策略**：不调用 `compute_ev` / `evaluate_intel` / Judge；
   不含 EV、评分、候选、尺寸、置信。
3. **不生成未来牌/runout**：`future_board` 恒为 unresolved。
4. **CASH-only，fail closed**：非 CASH → `present=False`；不得因 UNKNOWN 而
   切换到 MTT/PKO/Mystery 逻辑。
5. **不修 V3G-1 语义**：`to_call > hero_stack` 既有冲突**原样转述**。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cash_integrity_context import cash_integrity_context

UNKNOWN_E2E_VERSION = "v3g8-cash-unknown-e2e-v1"

#: 分类取值
PRESENT = "present"
UNKNOWN = "unknown"
EMPTY = "empty"

#: 端到端必须可观察的字段（授权 §9）
REQUIRED_REPORT_KEYS: tuple[str, ...] = (
    "present", "complete", "unknown", "incomplete", "conflicts",
    "evidence", "reason",
)

#: 合法街道（**唯一集合**；非此集合一律 UNKNOWN，不得当作 preflop）
VALID_STREETS: tuple[str, ...] = ("preflop", "flop", "turn", "river")

#: 被追踪的决策字段 → 说明（分类为 unknown 时**不得**被填充）
_TRACKED_FIELDS: tuple[str, ...] = (
    "table_size", "button_seat", "hero.seat", "hero.hole_cards", "hero.stack",
    "hero.position", "pot", "to_call", "bb", "board", "street",
    "action_history", "current_actor", "min_raise_to",
    "opponent_model", "range", "range_provenance", "sizing", "future_board",
    "confidence",
)


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


def _classify(fields: dict, name: str, value, *, evidence: str,
              empty_when=None) -> None:
    """登记一个字段的分类（**不修改任何输入值**）。"""
    if value is None or _is_blank(value) or (
            isinstance(value, float) and value != value):
        fields[name] = {"classification": UNKNOWN, "value": None,
                        "evidence": evidence, "defaulted": False}
        return
    if empty_when is not None and empty_when is True:
        fields[name] = {"classification": EMPTY, "value": value,
                        "evidence": evidence, "defaulted": False}
        return
    fields[name] = {"classification": PRESENT, "value": value,
                    "evidence": evidence, "defaulted": False}


@dataclass
class UnknownReport:
    version: str = UNKNOWN_E2E_VERSION
    present: bool = False
    complete: bool = False
    reason: str = ""
    mode: str | None = None
    unknown: list = field(default_factory=list)
    incomplete: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    fields: dict = field(default_factory=dict)
    layer_presence: dict = field(default_factory=dict)
    guarantees: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version, "present": self.present,
            "complete": self.complete, "reason": self.reason,
            "mode": self.mode,
            "unknown": list(self.unknown),
            "incomplete": list(self.incomplete),
            "conflicts": list(self.conflicts),
            "evidence": dict(self.evidence),
            "fields": dict(self.fields),
            "layer_presence": dict(self.layer_presence),
            "guarantees": dict(self.guarantees),
        }


def build_unknown_report(state: dict | None, *,
                         layers: dict | None = None) -> dict:
    """端到端 UNKNOWN 报告（只读；确定性；只分类不填充）。

    `layers`（可选）：其它层的输出，用于核对其自身的 `present` 与 UNKNOWN 证据，
    **只读取状态、不复制其计算**。可含键：
    `preflop` / `multistreet` / `opponent_model` / `range` / `action_tree`。
    """
    if not isinstance(state, dict):
        return {"present": False, "version": UNKNOWN_E2E_VERSION,
                "reason": "state_not_a_dict",
                "unknown": [], "incomplete": [], "conflicts": [],
                "fields": {}, "evidence": {"present": False}}

    ev = cash_integrity_context(state)
    if not ev.get("present"):
        return {"present": False, "version": UNKNOWN_E2E_VERSION,
                "reason": ev.get("reason", "not_cash_mode"),
                "unknown": [], "incomplete": [], "conflicts": [],
                "fields": {}, "evidence": {"present": False}}

    # 门禁结果**经 V3G-2 适配层**取得（唯一合法入口，不直接 import 门禁）
    integrity = dict(ev.get("state_integrity") or {})
    rep_missing = list(integrity.get("missing") or [])
    rep_conflicts = list(integrity.get("conflicts") or [])
    rep_unknown = list(integrity.get("unknown") or [])
    rep_ok = not rep_missing and not rep_conflicts

    hero = _hero(state) or {}
    fields: dict[str, dict] = {}
    _classify(fields, "table_size", state.get("table_size"),
              evidence="state.table_size")
    _classify(fields, "button_seat", state.get("button_seat"),
              evidence="state.button_seat（dealer；不得默认）")
    _classify(fields, "hero.seat", hero.get("seat"), evidence="players[is_hero].seat")
    holes = hero.get("hole_cards")
    _classify(fields, "hero.hole_cards",
              ([c for c in holes] if isinstance(holes, list) and len(holes) == 2
               and all(not _is_blank(c) for c in holes) else None),
              evidence="players[is_hero].hole_cards")
    _classify(fields, "hero.stack", _num(hero.get("stack")),
              evidence="players[is_hero].stack")
    _classify(fields, "hero.position", hero.get("position"),
              evidence="players[is_hero].position（只读既有标签，不反推）")
    _classify(fields, "pot", _num(state.get("pot")), evidence="state.pot")
    _classify(fields, "to_call", _num(state.get("to_call")), evidence="state.to_call")
    _classify(fields, "bb", _num(state.get("bb")), evidence="state.bb")

    board = state.get("board")
    street = state.get("street")
    street_l = street.lower() if isinstance(street, str) else None
    street_valid = street_l in VALID_STREETS
    # 非法/未知街道 → UNKNOWN（**不得**当作 preflop，也不得参与 board 判定）
    _classify(fields, "street",
              (street if street_valid else None),
              evidence="state.street（合法集合 = preflop/flop/turn/river；"
                       "非此集合视为 UNKNOWN）")
    # preflop 的空 board 是**确认的空**（EMPTY），不是 UNKNOWN
    board_empty_confirmed = (street_l == "preflop" and isinstance(board, list)
                             and len(board) == 0)
    _classify(fields, "board",
              (list(board) if isinstance(board, list) and len(board) > 0
               else ([] if board_empty_confirmed else None)),
              evidence="state.board",
              empty_when=board_empty_confirmed)
    _classify(fields, "action_history",
              (list(state["action_history"])
               if isinstance(state.get("action_history"), list) else None),
              evidence="state.action_history")
    _classify(fields, "min_raise_to", _num(state.get("min_raise_to")),
              evidence="state.min_raise_to（缺则不生成 raise 候选）")

    # current actor：与 V3G-4/V3G-7 **同一可靠性规则**（不是裸取尾项）：
    # 仅当"尾项属于当前街 + 在座玩家座位号完整"时才给出；否则 None（不猜）。
    ca = None
    hist = state.get("action_history")
    live = [p for p in state.get("players") or []
            if isinstance(p, dict) and p.get("status") in ("active", "allin")]
    if (isinstance(hist, list) and hist and isinstance(hist[-1], dict)
            and hist[-1].get("street") == street
            and hist[-1].get("player_id") is not None
            and live
            and all(p.get("seat") is not None for p in live)):
        ordered = [str(p.get("player_id"))
                   for p in sorted(live, key=lambda x: x.get("seat"))]
        last = str(hist[-1].get("player_id"))
        if last in ordered:
            ca = ordered[(ordered.index(last) + 1) % len(ordered)]
    _classify(fields, "current_actor", ca,
              evidence="同 V3G-4/V3G-7 规则：尾项属当前街 + 座位序完整；"
                       "否则 UNKNOWN（不猜）")

    # future board：**必须** unresolved（不得生成）
    fields["future_board"] = {
        "classification": UNKNOWN, "value": None,
        "evidence": "未生成未来牌（禁止虚构）", "defaulted": False,
        "must_not_invent": True,
    }
    # 本层不产生这些字段的证据 → 一律 UNKNOWN（不得伪造）
    for name, why in (("opponent_model", "本层不构造对手模型；无证据"),
                      ("range", "本层不构造范围；无证据"),
                      ("range_provenance", "无范围 → 无 provenance"),
                      ("sizing", "本层不生成尺寸候选"),
                      ("confidence", "本层不产出置信度")):
        fields[name] = {"classification": UNKNOWN, "value": None,
                        "evidence": why, "defaulted": False}

    # 层存在性（只读核对，不复制计算）
    layer_presence: dict[str, dict] = {}
    for key in ("preflop", "multistreet", "opponent_model", "range",
                "action_tree"):
        d = (layers or {}).get(key)
        if isinstance(d, dict):
            layer_presence[key] = {
                "supplied": True, "present": bool(d.get("present")),
                "reason": d.get("reason"),
            }
        else:
            layer_presence[key] = {"supplied": False, "present": None,
                                   "reason": "not_supplied"}

    unknown_fields = sorted(k for k, v in fields.items()
                            if v["classification"] == UNKNOWN)
    empty_fields = sorted(k for k, v in fields.items()
                          if v["classification"] == EMPTY)

    unknown = list(dict.fromkeys(
        rep_unknown + unknown_fields
        + [f"layer:{k}" for k, v in layer_presence.items()
           if v["supplied"] and v["present"] is False]))
    incomplete = list(rep_missing)
    conflicts = list(rep_conflicts)
    if not rep_ok:
        incomplete = list(dict.fromkeys(incomplete + ["state_integrity_incomplete"]))

    # 两个层级的完备性（**区分核心与上下文**，避免把两者混为一谈）：
    #   complete             = 核心决策字段齐备 **且** V3G-1 门禁无 missing/conflict
    #   complete_with_context = 上者 **且** 可选上下文（table_size/button_seat/
    #                           hero.position）亦为 present
    #: 核心字段：缺任一 → 无法形成可信决策上下文
    core_unknown = [k for k in ("street", "hero.hole_cards", "hero.stack", "bb",
                                "pot", "to_call")
                    if fields[k]["classification"] == UNKNOWN]
    complete = (rep_ok and not core_unknown)
    context_unknown = [k for k in ("table_size", "button_seat",
                                    "hero.position", "hero.seat",
                                    "min_raise_to", "action_history")
                       if fields[k]["classification"] == UNKNOWN]
    complete_with_context = complete and not context_unknown

    reason = ""
    if not complete:
        parts = []
        if rep_missing:
            parts.append(f"missing={rep_missing}")
        if rep_conflicts:
            parts.append(f"conflicts={rep_conflicts}")
        if core_unknown:
            parts.append(f"unknown_core={core_unknown}")
        reason = "incomplete:" + ";".join(parts)
    elif not complete_with_context:
        reason = f"complete_core_but_context_unknown={context_unknown}"

    return {
        "present": True, "version": UNKNOWN_E2E_VERSION,
        "complete": complete,
        "complete_with_context": complete_with_context,
        "reason": reason, "mode": ev.get("mode"),
        "unknown": unknown,
        "incomplete": incomplete,
        "conflicts": conflicts,
        "evidence": {
            "present": True,
            "state_integrity": integrity,
            "v3g2_evidence": ev,
            "empty_fields": empty_fields,
            "note": ("present=有观测证据；unknown=无证据（不得当作空/否/0）；"
                     "empty=证据确认不存在。三者均**不等于**默认值。"),
        },
        "fields": fields,
        "layer_presence": layer_presence,
        "guarantees": {
            "no_default_fill": True,
            "no_empty_substitution": True,
            "no_fabricated_candidates": True,
            "no_fabricated_sizing": True,
            "no_fabricated_future_cards": True,
            "no_fabricated_opponent_actions": True,
            "no_fabricated_range": True,
            "no_fabricated_confidence": True,
            "defaulted_fields": sorted(k for k, v in fields.items()
                                       if v.get("defaulted")),
            "unknown_is_not_empty": True,
            "unknown_is_not_fallback": True,
        },
    }


__all__ = ["UNKNOWN_E2E_VERSION", "REQUIRED_REPORT_KEYS", "PRESENT", "UNKNOWN",
           "EMPTY", "UnknownReport", "build_unknown_report"]
