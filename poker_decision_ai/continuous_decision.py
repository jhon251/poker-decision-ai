"""V3F-P30-7-G — Continuous Decision Contract（连续决策 provenance 契约）。

每次 decision 必须携带：
  hand_id / street / decision_id / state_provenance /
  action_history_status / range_mode / equity_status / strategy_mode

最小数据类（不依赖 previous frame 补 UNKNOWN；无状态——每次全量快照）。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContinuousDecisionProvenance:
    hand_id: str = ""
    street: str = ""
    decision_id: str = ""
    state_provenance: dict = field(default_factory=dict)
    action_history_status: str = "UNKNOWN"     # RECOGNIZED | UNKNOWN
    range_mode: str = "BASELINE"               # REAL | APPROXIMATE | BASELINE
    equity_status: str = "UNKNOWN"             # RECOGNIZED | UNKNOWN
    strategy_mode: str = "UNKNOWN"             # REAL | BASELINE | APPROXIMATE | UNKNOWN

    def to_dict(self) -> dict:
        return {"hand_id": self.hand_id, "street": self.street,
                "decision_id": self.decision_id,
                "state_provenance": dict(self.state_provenance),
                "action_history_status": self.action_history_status,
                "range_mode": self.range_mode,
                "equity_status": self.equity_status,
                "strategy_mode": self.strategy_mode}


def build_continuous_provenance(
        *, hand_id: str = "", street: str = "",
        decision_id: str = "", action_history_status: str = "UNKNOWN",
        range_mode: str = "BASELINE", equity_status: str = "UNKNOWN",
        strategy_mode: str = "UNKNOWN",
        source: str = "frame") -> ContinuousDecisionProvenance:
    """构建连续决策 provenance（显式来源——不依赖 previous frame）。"""
    return ContinuousDecisionProvenance(
        hand_id=hand_id, street=street, decision_id=decision_id,
        state_provenance={"source": source, "deduced": False},
        action_history_status=action_history_status,
        range_mode=range_mode, equity_status=equity_status,
        strategy_mode=strategy_mode)
