"""V3F-P30-5-B — Strategy Provider Contract（最小）。

统一现有 SolverProvider 抽象 → 明确 strategy 边界：
- BaselineStrategyProvider（mock）——is_approximate=True，绝不标记 GTO；
- ExternalStrategyProvider（postflop CLI）——is_approximate=False；
- provider failure → UNKNOWN/UNAVAILABLE（绝不随机 action fallback）；
- 输出必须经 Judge 安全门（本模块不直接出 action）。

复用（不新建 Solver）：core/providers/solver.py 的
MockSolverProvider / PostflopSolverCLIProvider 已存在——本模块是
strategy_mode 标注 + failure 语义的薄层。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: strategy_mode 明确区分（红线：baseline 不得冒充 GTO）
STRATEGY_MODES = ("REAL", "BASELINE", "APPROXIMATE", "UNKNOWN")


@dataclass
class StrategyResult:
    strategy_mode: str                      # REAL | BASELINE | APPROXIMATE | UNKNOWN
    provider_name: str = ""
    is_approximate: bool = True
    gto_action: str | None = None
    actions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"strategy_mode": self.strategy_mode,
                "provider_name": self.provider_name,
                "is_approximate": self.is_approximate,
                "gto_action": self.gto_action, "actions": self.actions,
                "warnings": self.warnings, "provenance": self.provenance}


def classify_solver_result(solver_name: str, is_approximate: bool,
                           actions: list | None = None,
                           gto_action: str | None = None) -> StrategyResult:
    """SolverProvider 输出 → StrategyResult（strategy_mode 诚实分类）。

    - real provider（非 mock、非近似）→ REAL；
    - mock/近似 → BASELINE（is_approximate=True）；
    - failure（actions 空 + provider 空）→ UNKNOWN；
    - 绝不标记 mock 为 GTO。
    """
    acts = list(actions or [])
    if not solver_name:
        return StrategyResult(strategy_mode="UNKNOWN", provider_name="",
                              is_approximate=True, gto_action=None,
                              actions=[], provenance={"reason":
                                                     "no_solver_provider"})
    if not is_approximate and solver_name not in ("mock", ""):
        mode = "REAL"
    elif is_approximate:
        mode = "BASELINE"
    else:
        mode = "UNKNOWN"
    return StrategyResult(strategy_mode=mode, provider_name=solver_name,
                          is_approximate=bool(is_approximate),
                          gto_action=gto_action, actions=acts,
                          provenance={"solver_name": solver_name,
                                      "is_approximate": bool(is_approximate)})


def strategy_failure(provider_name: str, reason: str) -> StrategyResult:
    """provider failure → UNKNOWN（无随机 fallback——绝不猜测 action）。"""
    return StrategyResult(strategy_mode="UNKNOWN", provider_name=provider_name,
                          is_approximate=True, gto_action=None, actions=[],
                          warnings=[{"code": "STRATEGY_UNAVAILABLE",
                                     "message": reason}],
                          provenance={"failure": reason})
