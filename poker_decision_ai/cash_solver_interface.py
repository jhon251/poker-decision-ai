"""V3G-9 — CASH Solver Fail-Closed Interface（只读状态接口；绝不调用 Solver）。

定位
----
为 CASH 提供 **Solver 状态**的诚实、可审计表达：是否配置、是否运行、结果是否
有效、是否均衡。**本层只回答状态，不产生任何 solver 结果。**

当前项目环境下，本层**必须**持续报告 `BLOCKED`（治理层禁用）或
`UNAVAILABLE`（无配置/无依赖），且 `equilibrium=false` / `gto_exact=false`。

状态枚举（**必须区分**）
------------------------
  * `UNAVAILABLE`         —— 无配置 / 无可执行文件 / 依赖缺失 / 路径不存在；
  * `BLOCKED`             —— 治理层明确禁止使用（**本层策略**）；
  * `UNKNOWN`             —— 状态证据不足，**不得**默认可用或不可用；
  * `CONFIGURED_NOT_RUN`  —— 有配置但本阶段**未实际运行**（不得声称已验证）；
  * `VALID_RESULT`        —— **仅**在真实执行 + 输出校验 + 版本/配置/输入完整时
                             允许；**当前项目不得伪造此状态**。

复用而非重实现（审计结论）
--------------------------
* `core/providers/solver.py::discover_solver_environment()` —— 既有 P61
  **只读**环境发现（executable_path / path_search / pip_package /
  sample_outputs 四通道），**直接复用**；本层不重写发现逻辑；
* `PostflopSolverCLIProvider.is_configured()` —— 既有"路径存在"判定语义
  被**复述**在 `configured` 字段中（本层不调用其 `solve()`）。

**治理策略的来源（重要，如实声明）**
------------------------------------
本层是**生产包**模块，受 V3G-1/V3G-2 的既有隔离约束（生产包不得 import
`research_unfreeze`），因此**不 import** `research_unfreeze.governance`。
故本层把"治理层禁止使用 Solver"作为**本层声明的策略**（`SOLVER_POLICY`）并
附带 provenance，而不是复制该模块的代码；其与
`governance.LIMITATIONS["solver"]`（`value="BLOCKED"`、
`equilibrium=false`、`NOT_exact_gto=true`）**语义一致**（由测试核对）。

硬约束
------
1. **绝不调用 Solver**：不 import `subprocess` / `asyncio`，不调用
   `solve()` / `create_subprocess_exec`；本层无执行能力。
2. **不伪造结果**：不产出 `strategy` / `ev` / `frequencies`；
   `VALID_RESULT` 在当前项目**不可达**（`valid_result_reachable=False`）。
3. **不冒充**：不把 PI-EV / heuristic / range equity / EV estimator 当作 Solver；
   输出不含任何此类数值。
4. **只读 + 确定性**：不修改输入；无 random、无 clock 依赖（除显式时间戳留空）。
5. **CASH-only，fail closed**：非 CASH / mode 缺失或非法 → `present=False`；
   不得让非 CASH 自动触发 solver 分支。
6. **不改变 Judge**：本层不参与决策；接线只允许决策后 metadata。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cash_integrity_context import cash_integrity_context

SOLVER_INTERFACE_VERSION = "v3g9-cash-solver-interface-v1"

#: ── 状态枚举 ────────────────────────────────────────────────────────
UNAVAILABLE = "UNAVAILABLE"
BLOCKED = "BLOCKED"
UNKNOWN = "UNKNOWN"
CONFIGURED_NOT_RUN = "CONFIGURED_NOT_RUN"
VALID_RESULT = "VALID_RESULT"

SOLVER_STATES: tuple[str, ...] = (UNAVAILABLE, BLOCKED, UNKNOWN,
                                  CONFIGURED_NOT_RUN, VALID_RESULT)

#: ── 本层声明的治理策略（不 import research_unfreeze；语义与
#:    governance.LIMITATIONS["solver"] 一致，由测试核对）────────────────
SOLVER_POLICY: dict[str, Any] = {
    "solver_use_forbidden": True,
    "equilibrium": False,
    "gto_exact": False,
    "valid_result_reachable": False,
    "declared_by": "v3g9-cash-solver-interface",
    "semantics_match": "research_unfreeze.governance.LIMITATIONS['solver']",
    "note": ("真实 postflop solver 未配置；治理层禁止使用；"
             "不得伪造 GTO/equilibrium/REAL solver 结果"),
}

#: 非 CASH 模式（fail closed；不得继承 CASH 的 solver metadata）
_NON_CASH_MODES = ("mtt", "tournament", "pko", "ko", "mystery", "spin",
                   "satellite")


def _env_discovery() -> dict:
    """复用既有 P61 只读发现（**不调用 solver**）。"""
    try:
        from .core.providers.solver import discover_solver_environment
        d = discover_solver_environment()
        return d if isinstance(d, dict) else {"available": None,
                                              "channels": {},
                                              "blocked_reason": "malformed"}
    except Exception as exc:                   # 发现失败 → 证据不足
        return {"available": None, "channels": {},
                "blocked_reason": f"discovery_failed:{type(exc).__name__}"}


def _configured_from_channels(channels: dict) -> bool | None:
    """`configured` = **既有**语义"可执行路径存在"（CLI provider 同义）。"""
    ex = channels.get("executable_path")
    if isinstance(ex, dict):
        return bool(ex.get("found"))
    return None


@dataclass
class SolverStatus:
    version: str = SOLVER_INTERFACE_VERSION
    present: bool = False
    reason: str = ""
    mode: str | None = None
    solver_status: str = UNKNOWN
    discovery_state: str = UNKNOWN
    discovery_reason: str = ""
    configured: bool | None = None
    executed: bool = False
    valid: bool = False
    equilibrium: bool = False
    gto_exact: bool = False
    available: bool | None = None
    channels: dict = field(default_factory=dict)
    policy: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    forbidden_uses: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "version": self.version, "present": self.present,
            "reason": self.reason, "mode": self.mode,
            "solver_status": self.solver_status,
            "discovery_state": self.discovery_state,
            "discovery_reason": self.discovery_reason,
            "configured": self.configured,
            "executed": self.executed, "valid": self.valid,
            "equilibrium": self.equilibrium, "gto_exact": self.gto_exact,
            "available": self.available,
            "channels": dict(self.channels),
            "policy": dict(self.policy),
            "provenance": dict(self.provenance),
            "forbidden_uses": list(self.forbidden_uses),
        }


def solver_status(state: dict | None, *, discovery: dict | None = None) -> dict:
    """CASH Solver 状态接口（只读；确定性；**不调用 Solver**）。

    * `present=False`：非 CASH / mode 缺失或非法 / 输入非 dict —— fail closed；
    * 当前环境下 `solver_status` 为 `BLOCKED`（治理禁用）或 `UNAVAILABLE`
      （无配置/无依赖）；`equilibrium` / `gto_exact` 恒为 `False`；
      `executed` / `valid` 恒为 `False`（本层无执行能力）。
    """
    if not isinstance(state, dict):
        return {"present": False, "version": SOLVER_INTERFACE_VERSION,
                "reason": "state_not_a_dict",
                "solver_status": UNKNOWN, "equilibrium": False,
                "gto_exact": False, "executed": False, "valid": False}

    ev = cash_integrity_context(state)
    if not ev.get("present"):
        return {"present": False, "version": SOLVER_INTERFACE_VERSION,
                "reason": ev.get("reason", "not_cash_mode"),
                "solver_status": UNKNOWN, "equilibrium": False,
                "gto_exact": False, "executed": False, "valid": False}

    disc = discovery if isinstance(discovery, dict) else _env_discovery()
    channels = dict(disc.get("channels") or {})
    available = disc.get("available")
    configured = _configured_from_channels(channels)

    forbidden = bool(SOLVER_POLICY["solver_use_forbidden"])

    # ── ① 发现态（discovery_state）：如实反映环境证据，**不受策略影响** ──
    if available is None:
        discovery_state = UNKNOWN
        discovery_reason = (f"discovery evidence insufficient: "
                            f"{disc.get('blocked_reason')}")
    elif available is False:
        discovery_state = UNAVAILABLE
        discovery_reason = str(disc.get("blocked_reason") or "no solver available")
    elif configured is True:
        discovery_state = CONFIGURED_NOT_RUN
        discovery_reason = ("solver executable present but NOT executed → "
                            "不得声称已验证")
    else:
        discovery_state = UNKNOWN
        discovery_reason = ("discovery reports availability but executability "
                            "is undetermined (configured=%r)" % (configured,))

    # ── ② 策略叠加（policy）：治理层禁止使用 → **压制为 BLOCKED** ────────
    #   注意：BLOCKED 是"禁用"判定，**不抹掉**发现态（两者都如实输出），
    #   因此 UNAVAILABLE / UNKNOWN 等状态仍可被观察与测试。
    if forbidden:
        status = BLOCKED
        reason = (f"governance: solver use forbidden → BLOCKED（不得调用、"
                  f"不得产生 solver 结果）；discovery_state={discovery_state}"
                  f"（{discovery_reason}）")
    else:
        status = discovery_state
        reason = discovery_reason

    # `valid` 仅在**非禁用**且发现态为 VALID_RESULT 时成立；当前恒为 False
    valid = (status == VALID_RESULT)
    st = SolverStatus(
        present=True, reason=reason, mode=ev.get("mode"),
        solver_status=status, configured=configured,
        executed=False, valid=valid,
        equilibrium=bool(SOLVER_POLICY["equilibrium"]),
        gto_exact=bool(SOLVER_POLICY["gto_exact"]),
        available=available, channels=channels,
        policy=dict(SOLVER_POLICY),
        discovery_state=discovery_state,
        discovery_reason=discovery_reason,
        provenance={
            "discovery": "core.providers.solver.discover_solver_environment"
                         "（P61 只读四通道）",
            "policy": "本层声明（生产包不得 import research_unfreeze；"
                      "语义与 governance.LIMITATIONS['solver'] 一致）",
            "path_env": "POSTFLOP_SOLVER_PATH",
            "provider": "core.providers.solver.PostflopSolverCLIProvider",
            "valid_result_reachable": bool(
                SOLVER_POLICY["valid_result_reachable"]),
        },
        forbidden_uses=[
            "禁止调用真实 solver（本层无执行能力）",
            "禁止引入 fake/mock solver 结果作为生产证据",
            "禁止以 PI-EV 冒充 solver",
            "禁止以 heuristic / range equity / EV estimator 冒充 solver",
            "禁止把「solver path 存在」当作「solver 已运行」",
            "禁止声称 GTO / equilibrium / exact",
        ],
    )
    return st.to_dict()


__all__ = ["SOLVER_INTERFACE_VERSION", "UNAVAILABLE", "BLOCKED", "UNKNOWN",
           "CONFIGURED_NOT_RUN", "VALID_RESULT", "SOLVER_STATES",
           "SOLVER_POLICY", "SolverStatus", "solver_status"]
