"""V3F-P14 — Complete State（版本化完整字段矩阵）。

- 纯数据装配：Snapshot → 逐字段 Field{value, status}（复用 DecisionInput 的
  Field/FieldStatus——不复制）；
- 每个字段状态 = 实际证据状态（RECOGNIZED/UNKNOWN/NOT_OBSERVED/
  EVIDENCE_LIMITED/NOT_READY），非枚举状态（如 UNCERTAIN）安全收敛为 UNKNOWN；
- 不推导 to_call / blind / legal_actions（依赖未闭环 → NOT_READY），
  不从 Pot 推 Bet、不从 Bet 推 Action 等跨字段伪造；
- 零策略/EV/推荐；零 V1 agents；零视觉推理。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .decision_input import Field, FieldStatus

_FIELD_STATUSES = {"RECOGNIZED", "UNKNOWN", "NOT_OBSERVED",
                   "EVIDENCE_LIMITED", "NOT_READY"}


def _safe(st) -> str:
    return st if st in _FIELD_STATUSES else "UNKNOWN"


def _f(value: Any, status: str) -> Field:
    return Field(value=value, status=FieldStatus(_safe(status)))


def _v(d: dict, key: str) -> Any:
    return d.get(key) if isinstance(d, dict) else None


@dataclass
class CompleteState:
    """现金桌决策所需状态的完整字段矩阵（versioned——事实层）。"""

    version: str = "1.0"
    fields: dict = field(default_factory=dict)
    ground_truth_status: str = "runtime_state_non_gt"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "fields": {k: (v.to_dict() if isinstance(v, Field) else v)
                       for k, v in self.fields.items()},
            "ground_truth_status": self.ground_truth_status,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CompleteState":
        return cls(
            version=d.get("version", "1.0"),
            fields={k: (Field.from_dict(v) if isinstance(v, dict)
                        and "status" in v else v)
                    for k, v in d.get("fields", {}).items()},
            ground_truth_status=d.get("ground_truth_status",
                                      "runtime_state_non_gt"),
        )


def build_complete_state(snapshot: dict) -> CompleteState:
    """Snapshot（P9 感知快照）→ CompleteState（完整字段矩阵，纯数据）。

    字段状态 = 观察者实际证据状态；依赖未闭环字段显式 NOT_READY（不推导）。
    """
    s = snapshot or {}
    street = _v(s, "street")
    hero = _v(s, "hero_cards")
    board = _v(s, "board")
    position = _v(s, "position")
    pot = _v(s, "pot")
    stack = _v(s, "stack")
    status = _v(s, "status")
    bet = _v(s, "bet")
    action = _v(s, "action")

    def _st(d):
        return (d or {}).get("last_status") if isinstance(d, dict) else None

    fields = {
        "hand_context.street": _f(_v(street, "last_street"),
                                  _st(street) or "UNKNOWN"),
        "hand_context.hero_cards": _f(
            _v(hero, "last_cards"),
            "RECOGNIZED" if _st(hero) == "RECOGNIZED" else "UNKNOWN"),
        "hand_context.board": _f(
            _v(board, "last_cards"),
            "RECOGNIZED" if _st(board) == "RECOGNIZED" else "UNKNOWN"),
        "position": _f(_v(position, "last_hero_position"),
                       "RECOGNIZED" if _st(position) == "RECOGNIZED"
                       else "UNKNOWN"),
        "pot": _f(_v(pot, "last_pot"), _st(pot) or "UNKNOWN"),
        "stack": _f(_v(stack, "last_seats"),
                    "RECOGNIZED" if _v(stack, "last_seats") else "UNKNOWN"),
        "status": _f(_v(status, "last_seats"), _st(status) or "UNKNOWN"),
        "bet": _f(_v(bet, "last_seats"),
                  _st(bet) if _st(bet) in ("RECOGNIZED", "EVIDENCE_LIMITED",
                                           "UNKNOWN")
                  else "EVIDENCE_LIMITED"),
        "action": _f(_v(action, "last_actions"),
                     _st(action) if _st(action) in (
                         "RECOGNIZED", "NOT_OBSERVED", "EVIDENCE_LIMITED",
                         "UNKNOWN") else "NOT_READY"),
        # 依赖未闭环字段（诚实——不推导、不补值、不置 0）
        "blind": _f(None, "NOT_READY"),
        "to_call": _f(None, "NOT_READY"),
        "legal_actions": _f(None, "NOT_READY"),
    }
    return CompleteState(fields=fields)
