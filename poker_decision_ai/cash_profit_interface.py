"""V3G-10 — CASH 收益实验**接口**（只读设计与执行前契约；**不执行实验**）。

定位
----
只提供**实验设计**与**执行前契约**：如何做、需要什么、何时才允许进入真实执行。
**本层不执行收益实验、不产生任何收益结果。**

状态区分（授权 §1）
------------------
  * `DESIGN_ONLY`                     —— 只有设计；无执行环境、无随机指派；
  * `EXECUTION_ENVIRONMENT_REQUIRED`  —— 设计已就绪，但**必须有可执行环境**才能开始；
  * `RESULT_UNAVAILABLE`              —— 已尝试获取结果但**结果不可得**（如缺字段）；
  * `VALID_RESULT`                    —— **仅**在真实执行 + 真实随机指派 + 完整
                                        结果字段 + 校验通过时成立；
                                        **当前项目不可伪造**。

当前环境（授权 §2 强制）
------------------------
  * `status = BLOCKED`（B3 UNIDENTIFIABLE）；
  * `result_status = EXECUTION_ENVIRONMENT_REQUIRED`；
  * `profit_identifiable = False`；
  * **不输出**任何 EV / 收益 / ROI / 胜率 / A-B 结果（连字段都不存在）。

与既有工件的一致性（审计结论）
------------------------------
设计与 `docs/V3F_M3_OUTCOME_AUDIT.json` 的
`preregistered_method_if_data_were_available` **逐项对齐**：
决策点为样本单位、按 `hand_id` 聚类稳健 SE、50/50 system vs frozen 随机指派、
bootstrap CI（n=2000, seed=42）、预注册分层与 MDE、不做多重比较漂移；
并沿用其 `missing_outcome_variables`（执行结果 / 反事实识别 / 货币映射 / 样本量）
与 `minimal_acquisition_plan`（P1 唯一可识别；P2 模型内；P3 禁止）。

**不修改任何既有定义**：本层不 import `strategy_core` / Judge / Vision / State /
`research_unfreeze` / PI-EV / λ / CFI / Solver；不定义新的 EV 或阈值。

硬约束
------
1. **不执行**：不 import `subprocess` / `asyncio` / `random` / `socket`；
   无执行能力；无网络。
2. **不伪造**：输出中**不存在** `profit` / `roi` / `ev` / `win_rate` /
   `ab_result` / `effect` 等数值字段；`VALID_RESULT` 显式不可达。
3. **只读 + 确定性**：不修改输入；同输入 → 同输出。
4. **CASH-only，fail closed**：非 CASH / mode 缺失或非法 → `present=False`，
   **不继承**任何 CASH 收益实验字段。
5. **不接 Judge**：本层不参与决策；若未来接线，只允许决策冻结后 metadata。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cash_integrity_context import cash_integrity_context

PROFIT_INTERFACE_VERSION = "v3g10-cash-profit-experiment-interface-v1"

#: ── 结果状态枚举（授权 §1）──────────────────────────────────────────
DESIGN_ONLY = "DESIGN_ONLY"
EXECUTION_ENVIRONMENT_REQUIRED = "EXECUTION_ENVIRONMENT_REQUIRED"
RESULT_UNAVAILABLE = "RESULT_UNAVAILABLE"
VALID_RESULT = "VALID_RESULT"

RESULT_STATUSES: tuple[str, ...] = (DESIGN_ONLY, EXECUTION_ENVIRONMENT_REQUIRED,
                                   RESULT_UNAVAILABLE, VALID_RESULT)

#: ── 顶层状态（授权 §2：当前环境必须保持 BLOCKED）────────────────────
BLOCKED = "BLOCKED"
READY_FOR_DESIGN = "READY_FOR_DESIGN"
READY_FOR_EXECUTION = "READY_FOR_EXECUTION"

#: ── 预注册实验设计（与 V3F_M3_OUTCOME_AUDIT 的 preregistered_method 对齐）──
DESIGN: dict[str, Any] = {
    "design_version": "cash-profit-ab-v1",
    "design_kind": "randomized A/B at the decision point（唯一可识别设计 = P1）",
    "sample_unit": "decision point（手内聚类）",
    "cluster_key": "hand_id",
    "assignment": {
        "scheme": "50/50 randomized assignment: system vs frozen",
        "randomized": True,
        "per_unit": True,
        "stratified_by": ["stakes", "table_size", "position", "street",
                          "facing_bet", "stack_depth_band"],
        "note": "预注册分层；分层维度在运行前固定，运行后不得增删",
    },
    "arms": {"control": "frozen", "treatment": "system"},
    "outcome": {
        "primary": "决策点之后 hero 的净筹码变化（bb）",
        "secondary": "该手 hero 的净结果（bb）",
        "unit": "bb",
        "requires": "system 与 frozen 各自的**真实动作执行环境**",
    },
    "analysis": {
        "estimator": "聚类稳健的均值差（cluster-robust mean difference）",
        "cluster_robust_se": True,
        "bootstrap": {"scheme": "cluster bootstrap", "n": 2000, "seed": 42,
                      "ci_level": 0.95, "paired": True},
        "seed": 42,
        "preregistered_strata": True,
        "mde": {"reported": True,
                "note": "MDE 必须在运行前按目标 power/α 与预期方差预注册"},
        "no_multiple_comparison_drift": True,
        "no_optional_stopping": True,
    },
    "result_fields": [
        "unit_id（decision point）", "hand_id（聚类键）", "arm（system|frozen）",
        "stratum", "outcome_bb", "outcome_missing_reason",
    ],
    "missing_value_handling": {
        "policy": "缺失即缺失：不插补、不删除样本、不以估计值替代",
        "missing_recorded_as": "outcome_bb=null + outcome_missing_reason",
        "excluded_from_estimate": True,
        "zero_filled": False,
        "note": "缺失样本必须可追溯；禁止以 PI-EV/EV 估计器插补",
    },
    "execution_environment": {
        "required": True,
        "capabilities": [
            "在决策点真实执行 system 与 frozen 两条臂的动作",
            "对手为真实/经验证的对手（非假设模型）",
            "可观测决策点之后 hero 的净筹码变化",
        ],
        "venue_and_compliance_authorization": True,
        "not_available_in_this_project": True,
    },
    "entry_gate": [
        "获得可执行环境与合规授权（P1）",
        "或：自对弈/模拟器 + **先冻结并验证的对手模型**（P2，结论仅限模型内）",
        "运行前封存设计（分层/seed/MDE/字段）并记录哈希",
        "样本量达到 MDE 所需量级",
    ],
    "not_sufficient": [
        "PI-EV / heuristic / range equity / EV estimator 作为收益结果（P3：禁止）",
        "历史人类策略结果冒充 system 策略结果",
        "无随机指派、无执行环境的观察性一致率分析",
        "Solver BLOCKED 被转换成 GTO/equilibrium 结论",
    ],
}

#: ── 明确禁止用作收益证据的东西（授权 §5）────────────────────────────
FORBIDDEN_AS_PROFIT: tuple[str, ...] = (
    "PI-EV / 任何 EV 估计器",
    "heuristic / 规则引擎输出",
    "range equity / 胜率估计",
    "历史人类策略的已实现结果（冒充 system 策略结果）",
    "benchmark 一致率",
    "Solver BLOCKED 状态（不得转成 GTO/equilibrium 结论）",
)

#: 本层**永远不产出**的结果字段（防伪造的机器可核查清单）
NEVER_EMITTED_FIELDS: tuple[str, ...] = (
    "profit", "profit_bb", "roi", "ev", "ev_bb", "win_rate", "ab_result",
    "effect", "effect_bb", "ci", "p_value", "net_bb", "bb_per_100",
)

_NON_CASH_MODES = ("mtt", "tournament", "pko", "ko", "mystery", "spin",
                   "satellite")


@dataclass
class ProfitExperimentInterface:
    version: str = PROFIT_INTERFACE_VERSION
    present: bool = False
    reason: str = ""
    mode: str | None = None
    status: str = BLOCKED
    result_status: str = EXECUTION_ENVIRONMENT_REQUIRED
    profit_identifiable: bool = False
    executed: bool = False
    valid_result_reachable: bool = False
    executable_environment_available: bool = False
    randomized_assignment_available: bool = False
    design: dict = field(default_factory=dict)
    entry_gate: list = field(default_factory=list)
    not_sufficient: list = field(default_factory=list)
    forbidden_as_profit: list = field(default_factory=list)
    never_emitted_fields: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": self.version, "present": self.present,
            "reason": self.reason, "mode": self.mode,
            "status": self.status, "result_status": self.result_status,
            "profit_identifiable": self.profit_identifiable,
            "executed": self.executed,
            "valid_result_reachable": self.valid_result_reachable,
            "executable_environment_available":
                self.executable_environment_available,
            "randomized_assignment_available":
                self.randomized_assignment_available,
            "design": dict(self.design),
            "entry_gate": list(self.entry_gate),
            "not_sufficient": list(self.not_sufficient),
            "forbidden_as_profit": list(self.forbidden_as_profit),
            "never_emitted_fields": list(self.never_emitted_fields),
            "provenance": dict(self.provenance),
        }


def profit_experiment_interface(state: dict | None, *,
                                environment: dict | None = None) -> dict:
    """CASH 收益实验接口（只读；确定性；**不执行实验**）。

    `environment`（可选，供测试/未来接线传入**外部声明的**能力）：
      {"executable_environment": bool, "randomized_assignment": bool,
       "venue_and_compliance_authorized": bool}

    * `present=False`：非 CASH / mode 缺失或非法 / 输入非 dict —— fail closed；
    * 当前环境：`status=BLOCKED`、`result_status=EXECUTION_ENVIRONMENT_REQUIRED`、
      `profit_identifiable=False`、`executed=False`；
    * **任何情况下都不产出**收益/EV/ROI/胜率/A-B 结果字段。
    """
    if not isinstance(state, dict):
        return {"present": False, "version": PROFIT_INTERFACE_VERSION,
                "reason": "state_not_a_dict",
                "status": BLOCKED,
                "result_status": EXECUTION_ENVIRONMENT_REQUIRED,
                "profit_identifiable": False, "executed": False,
                "valid_result_reachable": False}

    ev = cash_integrity_context(state)
    if not ev.get("present"):
        return {"present": False, "version": PROFIT_INTERFACE_VERSION,
                "reason": ev.get("reason", "not_cash_mode"),
                "status": BLOCKED,
                "result_status": EXECUTION_ENVIRONMENT_REQUIRED,
                "profit_identifiable": False, "executed": False,
                "valid_result_reachable": False}

    env = environment if isinstance(environment, dict) else {}
    exe_env = env.get("executable_environment") is True
    rand_ok = env.get("randomized_assignment") is True
    auth_ok = env.get("venue_and_compliance_authorized") is True

    # ── 可识别性判定（诚实；缺任一必要条件 → 不可识别）────────────────
    identifiable = bool(exe_env and rand_ok and auth_ok)

    if identifiable:
        # 设计就绪 + 环境齐备 → **仍不执行**（本层无执行能力）
        status = READY_FOR_EXECUTION
        result_status = RESULT_UNAVAILABLE
        reason = ("execution preconditions declared satisfied, but this layer "
                  "has NO execution capability → 结果仍不可得（RESULT_UNAVAILABLE）；"
                  "真实执行需单独授权")
    else:
        status = BLOCKED
        result_status = EXECUTION_ENVIRONMENT_REQUIRED
        missing = []
        if not exe_env:
            missing.append("executable_environment")
        if not rand_ok:
            missing.append("randomized_assignment")
        if not auth_ok:
            missing.append("venue_and_compliance_authorization")
        reason = ("B3 UNIDENTIFIABLE：缺 " + ", ".join(missing)
                  + " → 收益不可识别（历史人类结果不可用作 system 策略结果）")

    iface = ProfitExperimentInterface(
        present=True, reason=reason, mode=ev.get("mode"),
        status=status, result_status=result_status,
        profit_identifiable=identifiable, executed=False,
        valid_result_reachable=False,          # 当前项目**不可达**
        executable_environment_available=exe_env,
        randomized_assignment_available=rand_ok,
        design=dict(DESIGN),
        entry_gate=list(DESIGN["entry_gate"]),
        not_sufficient=list(DESIGN["not_sufficient"]),
        forbidden_as_profit=list(FORBIDDEN_AS_PROFIT),
        never_emitted_fields=list(NEVER_EMITTED_FIELDS),
        provenance={
            "design_source": "docs/V3F_M3_OUTCOME_AUDIT.json"
                             "::preregistered_method_if_data_were_available",
            "blocker": "B3 BLOCKED / UNIDENTIFIABLE（V3F 已确认）",
            "m3_facts": {
                "action_differs_from_realized": "54.6% (4083/7483)",
                "missing_outcome_variables": [
                    "系统自身动作下实现的真实结果（需执行/A-B 随机指派）",
                    "反事实识别（需随机化或经验证的对手行为模型）",
                    "货币/效用映射（盲注级别、抽水、场次方差）",
                    "足够样本量（bb/100 的方差需要 10^5–10^6 手量级）",
                ],
            },
            "not_wired": "本层不接 Judge；无调用者",
            "no_execution": "本层无执行能力（无 subprocess/网络/random）",
            "valid_result_reachable": False,
        },
    )
    return iface.to_dict()


def evaluate_entry_gate(*, environment: dict | None = None) -> dict:
    """执行前门禁（只读；确定性）：**明确回答"何时允许进入真实执行"**。

    返回 `{"allowed": bool, "missing": [...], "reason": str}`。
    当前项目环境下 `allowed=False`。
    """
    env = environment if isinstance(environment, dict) else {}
    need = {
        "executable_environment": env.get("executable_environment") is True,
        "randomized_assignment": env.get("randomized_assignment") is True,
        "venue_and_compliance_authorization":
            env.get("venue_and_compliance_authorized") is True,
        "design_sealed": env.get("design_sealed") is True,
        "sample_size_meets_mde": env.get("sample_size_meets_mde") is True,
    }
    missing = [k for k, v in need.items() if not v]
    return {
        "allowed": not missing,
        "missing": missing,
        "reason": ("" if not missing else
                   "entry gate not satisfied: " + ", ".join(missing)),
        "criteria": list(DESIGN["entry_gate"]),
        "note": ("即使门禁通过，本层也**不执行**实验（无执行能力）；"
                 "真实执行需单独授权"),
    }


__all__ = ["PROFIT_INTERFACE_VERSION", "DESIGN_ONLY",
           "EXECUTION_ENVIRONMENT_REQUIRED", "RESULT_UNAVAILABLE",
           "VALID_RESULT", "RESULT_STATUSES", "BLOCKED", "READY_FOR_DESIGN",
           "READY_FOR_EXECUTION", "DESIGN", "FORBIDDEN_AS_PROFIT",
           "NEVER_EMITTED_FIELDS", "ProfitExperimentInterface",
           "profit_experiment_interface", "evaluate_entry_gate"]
