"""V3F-P30-11-B — Solver Contract（冻结语义）。

REAL：
  - 真实 Solver executable 成功运行；输入完整；输出可解析；provenance 明确。
BASELINE：
  - Mock / heuristic / 弱先验；永远不得标记 REAL。
APPROXIMATE：
  - 明确近似算法；永远不得标记 REAL。
UNKNOWN：
  - solver failure / 输入不足 / 输出非法 / timeout / provider 不可用。

任何 failure：禁止随机 action fallback（输出空 actions + UNKNOWN）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

SOLVER_MODES = ("REAL", "BASELINE", "APPROXIMATE", "UNKNOWN")
#: mock 永不 REAL（classify 断言）
MOCK_NEVER_REAL = True


@dataclass
class SolverContractResult:
    mode: str                       # REAL | BASELINE | APPROXIMATE | UNKNOWN
    provider_name: str = ""
    is_approximate: bool = True
    gto_action: str | None = None
    actions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"mode": self.mode, "provider_name": self.provider_name,
                "is_approximate": self.is_approximate,
                "gto_action": self.gto_action, "actions": self.actions,
                "warnings": self.warnings, "provenance": dict(self.provenance)}


def freeze_mode(solver_name: str, is_approximate: bool,
                actions: list | None = None,
                gto_action: str | None = None) -> SolverContractResult:
    """Solver output → Solver Contract（模式冻结——绝不混淆）。

    - real provider（非近似、非 mock）→ REAL；
    - mock/近似 → BASELINE（is_approximate 保留）;
    - 无 provider → UNKNOWN（无随机 fallback——actions 保持原样记录，
      gto_action=None）。
    """
    acts = list(actions or [])
    prov = {"solver_name": solver_name, "is_approximate": bool(is_approximate)}
    if not solver_name:
        return SolverContractResult(mode="UNKNOWN", provider_name="",
                                    is_approximate=True, gto_action=None,
                                    actions=[],
                                    warnings=[{"code": "SOLVER_UNAVAILABLE",
                                               "message": "no solver provider"}],
                                    provenance={**prov,
                                                "reason": "no_provider"})
    if not is_approximate and solver_name not in ("mock", ""):
        mode = "REAL"
    elif is_approximate:
        mode = "BASELINE"
    else:
        mode = "UNKNOWN"
    return SolverContractResult(mode=mode, provider_name=solver_name,
                                is_approximate=bool(is_approximate),
                                gto_action=gto_action, actions=acts,
                                provenance=prov)
