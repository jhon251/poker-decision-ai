"""V3F-P11-A4 — DecisionInput Contract（未来 Decision Engine 的输入契约）。

- 只描述【事实 + 事实状态 + provenance】——零策略/EV/推荐/动作选择；
- Field 包装：value + status（RECOGNIZED/UNKNOWN/NOT_OBSERVED/
  EVIDENCE_LIMITED/NOT_READY）——严格区分 known-zero vs UNKNOWN；
- 版本化（version="1.0"）——未来 Bet/Action CLOSED 后可增字段而不破坏；
- 纯数据契约（可 JSON 序列化）——不实现 Decision Engine。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class FieldStatus(str, enum.Enum):
    RECOGNIZED = "RECOGNIZED"
    UNKNOWN = "UNKNOWN"
    NOT_OBSERVED = "NOT_OBSERVED"
    EVIDENCE_LIMITED = "EVIDENCE_LIMITED"
    NOT_READY = "NOT_READY"


@dataclass
class Field:
    """事实字段 + 证据状态（区分 known-zero 与 UNKNOWN）。"""

    value: Any
    status: FieldStatus

    def to_dict(self) -> dict:
        return {"value": self.value,
                "status": self.status.value}

    @classmethod
    def from_dict(cls, d: dict) -> "Field":
        return cls(value=d.get("value"),
                   status=FieldStatus(d.get("status", "UNKNOWN")))


def recognized(value: Any) -> Field:
    return Field(value=value, status=FieldStatus.RECOGNIZED)


def unknown() -> Field:
    return Field(value=None, status=FieldStatus.UNKNOWN)


@dataclass
class DecisionInput:
    """Cash Game AI 的决策输入契约（versioned——事实层，非策略层）。"""

    version: str = "1.0"
    hand_context: dict = field(default_factory=dict)       # hand_id/street/board/hero_cards/position...
    players: dict = field(default_factory=dict)           # seat -> {status,stack,current_bet,committed,position}
    pot_betting: dict = field(default_factory=dict)       # pot/current_street/total_committed/current_bet
    evidence: dict = field(default_factory=dict)          # 字段级 provenance（Field）
    ground_truth_status: str = "runtime_state_non_gt"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "hand_context": self.hand_context,
            "players": self.players,
            "pot_betting": self.pot_betting,
            "evidence": {k: (v.to_dict() if isinstance(v, Field) else v)
                         for k, v in self.evidence.items()},
            "ground_truth_status": self.ground_truth_status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionInput":
        return cls(
            version=d.get("version", "1.0"),
            hand_context=d.get("hand_context", {}),
            players=d.get("players", {}),
            pot_betting=d.get("pot_betting", {}),
            evidence={k: (Field.from_dict(v) if isinstance(v, dict)
                          and "status" in v else v)
                      for k, v in d.get("evidence", {}).items()},
            ground_truth_status=d.get("ground_truth_status",
                                      "runtime_state_non_gt"),
        )


def from_snapshot(snapshot: dict) -> DecisionInput:
    """RuntimeSnapshot → DecisionInput（纯数据映射——UNKNOWN 保留）。

    - bet → EVIDENCE_LIMITED · action → NOT_READY（显式标注——不置 0）；
    - stack UNKNOWN 原样（718,902 不自动调用 Foundation 值）；
    - 零推断/策略。
    """
    evidence = {}
    stack = snapshot.get("stack") or {}
    bet = snapshot.get("bet") or {}
    action = snapshot.get("action") or {}
    for seat, v in (stack.get("last_seats") or {}).items():
        status = FieldStatus(v.get("status", "UNKNOWN")) \
            if v.get("status") == "RECOGNIZED" \
            else FieldStatus.UNKNOWN
        evidence[f"stack_{seat}"] = Field(value=v.get("stack"),
                                          status=status)
    # V3F-P13-C Bet 透传（仅数据透传，无策略/推荐/动作派生）：
    # BetObserver summary 存在 last_seats → 逐 zone 证据 + 顶层 bet 字段；
    # 无 BetObserver（P9 占位 {status:UNKNOWN, reason:evidence_limited}）→
    # 保持 EVIDENCE_LIMITED 语义不变。
    bet_seats = bet.get("last_seats")
    if bet_seats is not None:
        for zone, v in bet_seats.items():
            rec = v.get("status") == "RECOGNIZED"
            evidence[f"bet_{zone}"] = Field(
                value=v.get("bet"),
                status=FieldStatus.RECOGNIZED if rec
                else FieldStatus.UNKNOWN)
        evidence["bet"] = Field(
            value={z: v.get("bet") for z, v in bet_seats.items()},
            status=FieldStatus.RECOGNIZED
            if any(v.get("status") == "RECOGNIZED"
                   for v in bet_seats.values())
            else FieldStatus.UNKNOWN)
    else:
        evidence["bet"] = Field(value=None,
                                status=FieldStatus.EVIDENCE_LIMITED)
    # V3F-P13-ACTION 透传（仅数据透传，无策略）：
    # ActionObserver summary 存在 last_actions → 原样透传动作状态；
    # 无 observer（P9 占位 NOT_READY）→ 保持 NOT_READY 语义不变。
    action_seats = action.get("last_actions")
    if action_seats is not None:
        st = action.get("last_status")
        if st in ("RECOGNIZED", "NOT_OBSERVED", "EVIDENCE_LIMITED",
                  "UNKNOWN"):
            evidence["action"] = Field(value=action_seats,
                                       status=FieldStatus(st))
        else:
            evidence["action"] = Field(value=None,
                                       status=FieldStatus.UNKNOWN)
    else:
        evidence["action"] = Field(value=None, status=FieldStatus.NOT_READY)
    return DecisionInput(
        hand_context={"street": (snapshot.get("street") or {}).get(
            "last_street"),
                      "hero_cards": (snapshot.get("hero_cards") or {}).get(
                          "last_cards"),
                      "board": (snapshot.get("board") or {}).get(
                          "last_cards")},
        pot_betting={"pot": (snapshot.get("pot") or {}).get("last_pot")},
        players={seat: {"stack": (v.get("stack") if isinstance(v, dict)
                                  else None),
                        "status": (v.get("status") if isinstance(v, dict)
                                   else "UNKNOWN")}
                 for seat, v in (stack.get("last_seats") or {}).items()},
        evidence=evidence,
    )
