"""V3F-P30-9-C/D — Production Capture → Decision Loop + Decision Safety。

最小生产循环（复用既有组件——不复制 Runtime/Vision/Decision Engine）：
```
FrameStream（去重/节流）
    → frame_change / trigger
    → frame_to_decision_request（readiness gate）
    → RealtimeRunner.decide()
    → DecisionResult（decision_id + provenance）
```

Safety Contract（幂等/防重复）：
- decision_id 唯一（frame_id + hash 内容摘要——同帧不重复决策）；
- dedup：FrameStream 已去重（duplicate_frame——复用）；
- 同街重复帧 → 同 decision_id（幂等返回——不重复执行 decide）；
- solver timeout/failure → 冻结 P30-9-E 映射：degraded → BASELINE
  （mock 降级 + SOLVER_DEGRADED 警告——绝不 REAL；P34 实测语义）；
- mandatory UNKNOWN / illegal action → 不决策（gate 保持）；
- R14-EXE-3（方案 Y）：legal_actions UNKNOWN（bridge OK + to_call UNKNOWN）
  → ADVICE_REFERENCE 参考建议通道（action=声明表/spot 库对该 street+手牌
  +pot 的标准打法参考；advice_type="reference" / legal_actions_status=
  "unknown" / action_validated 恒 False——与正常决策严格可区分，绝不被
  下游当成"针对当前下注的决策"消费；诚实警告文案）；
- 无自动点击下注（只输出 DecisionResult——供 person/服务消费）。

红线：无 previous-frame 补值；无 frame→action 归因；无 UNKNOWN→0/[]。
"""
from __future__ import annotations

import hashlib
import time as _time
from dataclasses import dataclass, field

import numpy as np

from .advice_reference import (ADVICE_TYPE_REFERENCE,
                               ADVICE_WARNING_ZH,
                               DECISION_STATUS_ADVICE_REFERENCE,
                               LEGAL_ACTIONS_STATUS_UNKNOWN)
from .frame_decision import FrameDecisionResult, frame_to_decision


@dataclass
class DecisionSafetyResult:
    decision_id: str = ""
    hand_id: str = ""
    street: str = ""
    decision_status: str = "UNKNOWN"      # READY | SKIPPED_DUP | UNAVAILABLE |
    #                                       ERROR | ADVICE_REFERENCE
    action: str | None = None
    action_validated: bool = False
    strategy_mode: str = "UNKNOWN"
    solver_mode: str = "UNKNOWN"
    range_mode: str = "UNKNOWN"
    equity_status: str = "UNKNOWN"
    source_frame_id: int = 0
    provenance: dict = field(default_factory=dict)
    warning: str = ""
    # R14-EXE-3：参考建议模式契约（仅 decision_status=ADVICE_REFERENCE 时
    # advice_type 非空；legal_actions_status：known | unknown）
    advice_type: str | None = None
    legal_actions_status: str = "UNKNOWN"
    # P30-11-D：完整 provenance（solver/input 级）
    solver_provenance: dict | None = None
    input_provenance: dict | None = None

    def to_dict(self) -> dict:
        return {"decision_id": self.decision_id, "hand_id": self.hand_id,
                "street": self.street,
                "decision_status": self.decision_status,
                "action": self.action,
                "action_validated": self.action_validated,
                "advice_type": self.advice_type,
                "legal_actions_status": self.legal_actions_status,
                "strategy_mode": self.strategy_mode,
                "solver_mode": self.solver_mode,
                "range_mode": self.range_mode,
                "equity_status": self.equity_status,
                "source_frame_id": self.source_frame_id,
                "solver_provenance": self.solver_provenance,
                "input_provenance": self.input_provenance,
                "provenance": dict(self.provenance),
                "warning": self.warning}


class ProductionDecisionLoop:
    """生产决策循环（每帧调用——幂等去重；不自动点击下注）。

    用户（person/服务/脚本）在每帧调用 tick(frame_id, frame_rgba, w, h)
    ——内部 gate + 幂等后调用 decide()；同帧/同状态返回 SKIPPED_DUP。
    """

    def __init__(self, runtime: Any, *, hero_id: str = "Hero",
                 throttle_s: float = 1.0,
                 extra_request: dict | None = None,
                 event_consumer: Any | None = None):
        self.runtime = runtime
        self.hero_id = hero_id
        self.throttle_s = throttle_s
        self.extra_request = dict(extra_request or {})
        # P30-10-E：可选事件流消费者（视觉事实层——绝不归因行为）
        self.event_consumer = event_consumer
        self._last_decision_id = None
        self._last_state_key = None
        self._last_ts = 0.0
        self.decisions: list[dict] = []

    def _state_key(self, frame_rgba, w: int, h: int) -> str:
        """帧内容摘要（帧指纹——同内容幂等；不同于 FrameStream dedup，
        用于"同街重复帧不重复决策"语义）。"""
        b = (frame_rgba if isinstance(frame_rgba, bytes)
             else frame_rgba.tobytes())
        return hashlib.sha1(b[:4096]).hexdigest()

    def _decision_id(self, frame_id: int, state_key: str) -> str:
        return f"d-{frame_id}-{state_key[:8]}"

    def tick(self, frame_id: int, frame_rgba, w: int, h: int) -> dict:
        now = _time.monotonic()
        state_key = self._state_key(frame_rgba, w, h)
        decision_id = self._decision_id(frame_id, state_key)
        # 幂等 1：同帧/同状态（同 decision_id）→ 返回上次（不重复 decide）
        if decision_id == self._last_decision_id:
            return self.decisions[-1] if self.decisions else \
                DecisionSafetyResult(
                    decision_id=decision_id,
                    decision_status="SKIPPED_DUP",
                    warning="same_frame_decision_already_emitted"
                ).to_dict()
        # 幂等 2：同街状态短时内重复帧（state_key 相同 + throttle）→ 跳过
        if state_key == self._last_state_key and \
                (now - self._last_ts) < self.throttle_s:
            return DecisionSafetyResult(
                decision_id=decision_id,
                decision_status="SKIPPED_DUP",
                warning="same_state_within_throttle_window"
            ).to_dict()

        # gate + decide
        result = frame_to_decision(self.runtime, frame_rgba, w, h,
                                   hero_id=self.hero_id,
                                   extra_request=self.extra_request or None)
        safe = self._to_safety(result, frame_id, decision_id, state_key)

        self._last_decision_id = decision_id
        self._last_state_key = state_key
        self._last_ts = now
        self.decisions.append(safe.to_dict())
        return safe.to_dict()

    def _to_safety(self, res: FrameDecisionResult, frame_id: int,
                   decision_id: str, state_key: str) -> DecisionSafetyResult:
        prov = dict(res.provenance or {})
        # P30-10-E：事件流 provenance（视觉事实 trace——行为始终 UNKNOWN）
        if self.event_consumer is not None:
            prov["visual_fact_trace"] = self.event_consumer.summary()
        # ── R14-EXE-3（方案 Y）：参考建议模式 ────────────────────────────
        # bridge OK + legal_actions UNKNOWN → 参考建议（非决策）。
        # action 只读 advice_reference 通道（judge 输出仅存
        # provenance.judge_action_not_consumed——绝不当决策消费）。
        if res.status == "ADVICE_REFERENCE":
            adv = dict(prov.get("advice_reference") or {})
            return DecisionSafetyResult(
                decision_id=decision_id,
                hand_id=prov.get("hand_id", ""),
                street=prov.get("street", ""),
                decision_status=DECISION_STATUS_ADVICE_REFERENCE,
                action=adv.get("action"),
                action_validated=False,       # legal_actions 未知——恒 False
                advice_type=ADVICE_TYPE_REFERENCE,
                legal_actions_status=LEGAL_ACTIONS_STATUS_UNKNOWN,
                strategy_mode="BASELINE",     # chart/声明表——绝不 REAL/RULES
                solver_mode="BASELINE",
                range_mode="BASELINE",
                equity_status="UNKNOWN",
                source_frame_id=frame_id,
                provenance={**prov, "state_fingerprint": state_key},
                warning=ADVICE_WARNING_ZH,
                solver_provenance={
                    "mode": "BASELINE",
                    "solver_source": adv.get("source"),
                    "solver_status": "advice_reference"},
                input_provenance={
                    "action_history_status": (prov.get("action_history")
                                              or {}).get("status", "UNKNOWN")
                    if isinstance(prov.get("action_history"), dict)
                    else "UNKNOWN",
                    "current_bet_status": "UNKNOWN",
                    "legal_actions_status": LEGAL_ACTIONS_STATUS_UNKNOWN,
                    "fields_ready": prov.get("mandatory_fields_ready", [])})
        fd = ((res.report or {}).get("final_decision") or {}) \
            if res.report else {}
        ah = (prov.get("action_history") or {})
        # P30-9-E：模式从 report 实际执行情况读取（不合成/不隐式升级）
        rep = res.report or {}
        agents_ran = list(rep.get("agents_ran") or [])
        fd_sum = ((res.report or {}).get("final_decision") or {}).get(
            "final_inputs_summary") or {} if res.report else {}
        strategy_mode = "UNKNOWN"
        solver_source = None                  # P38-F2：实际 provider 来源
        if isinstance(fd_sum, dict):
            s = fd_sum.get("solver")
            # P38-F1：solver 来源 = runtime 实际配置的 provider（诚实声明）。
            # summary "ok" 仅代表 provider 调用成功且未自称 approximate——
            #   REAL   仅指外部可验证 solver（postflop CLI，结果可复核）；
            #   RULES  指 P34 realengine（确定性规则引擎——真实计算但非 GTO、
            #          非外部 solver；算法零修改，仅标签分离）；
            #   其余来源（mock 等，或无法证明来源）→ BASELINE（保守，绝不
            #   伪造 REAL）。
            ps = ((rep.get("provider_status") or {}).get("solver") or {})
            solver_source = ps.get("provider") or None
            if s == "ok":
                if solver_source == "postflop":
                    strategy_mode = "REAL"
                elif solver_source == "realengine":
                    strategy_mode = "RULES"
                else:
                    strategy_mode = "BASELINE"
            elif s == "chart":
                # SOLVER-2：preflop 图表基线（source=chart）**非 solver 产物**
                # ——绝不 REAL/RULES（诚实 BASELINE）。
                strategy_mode = "BASELINE"
            elif s in ("degraded", "missing"):
                strategy_mode = "BASELINE"
        solver_mode = strategy_mode           # P38-F2：provenance 与标签同源
        range_mode = "BASELINE"
        equity_status = ("RECOGNIZED"
                         if isinstance(fd_sum, dict)
                         and fd_sum.get("equity") == "ok" else "UNKNOWN")
        return DecisionSafetyResult(
            decision_id=decision_id,
            hand_id=prov.get("hand_id", ""),
            street=prov.get("street", ""),
            decision_status=("READY" if res.status == "OK" else
                             ("UNAVAILABLE" if res.status in
                              ("UNAVAILABLE", "DECISION_UNAVAILABLE")
                              else "ERROR")),
            action=fd.get("action"),
            action_validated=bool(prov.get("action_validated", False)),
            legal_actions_status=("known" if prov.get("legal_actions")
                                  else "UNKNOWN"),
            strategy_mode=strategy_mode,
            solver_mode=strategy_mode,
            range_mode=range_mode,
            equity_status=equity_status,
            source_frame_id=frame_id,
            provenance={**prov, "state_fingerprint": state_key},
            warning=res.reason or "",
            # P30-11-D + P38-F2：solver/input 级 provenance——mode 与
            # strategy_mode 同源同值；solver_source 表达真实 provider。
            solver_provenance={"mode": strategy_mode,
                               "solver_source": solver_source,
                               "solver_status": fd_sum.get("solver")
                               if isinstance(fd_sum, dict) else None},
            input_provenance={
                "action_history_status": ah.get("status", "UNKNOWN")
                if isinstance(ah, dict) else "UNKNOWN",
                "current_bet_status": "UNKNOWN",
                "fields_ready": prov.get("mandatory_fields_ready", [])})

    def history(self) -> list:
        return list(self.decisions)
