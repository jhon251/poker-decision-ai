"""V3F-P30-3-B — Frame → Production Decision 最小适配器。

真实 1876 帧 → V3F observers → snapshot → production_bridge →
RealtimeRunner.decide(request)。复用既有 Runtime/RealtimeRunner 生命周期
（无第二套 realtime loop；无第二套视觉识别；无伪造 request）。

Gate（UNKNOWN-first）：
- bridge status==OK → request 带 vision_observation → decide()；
- bridge status==UNAVAILABLE → 返回 UNAVAILABLE + reason（不喂决策——
  绝不把 UNKNOWN 转 0 后给决策引擎）。

R14-EXE-3（方案 Y）：bridge OK + legal_actions UNKNOWN（to_call UNKNOWN，
未识别"轮到你/当前下注额"）→ 参考建议通道 ADVICE_REFERENCE——action=
声明表/spot 库对该 street+手牌+pot 的标准打法参考（advice_reference 模块
查表），带诚实标注；与正常决策严格可区分（advice_type="reference" /
legal_actions_status="unknown" / action_validated=False）。legal_actions
恢复识别的帧自动走既有正常路径（结构性升级）。参考通道 miss → 维持
DECISION_UNAVAILABLE（诚实拒绝，绝不编造标准打法）。

Action history：本模块不伪造帧→事件推断（禁止 previous-frame fallback /
poker-rule 补历史）。若调用方提供 request["hand_history_text"]，既有 V2
StateEngine 路径天然提供 action_history（_from_hand_history）；否则
action_history 保持 UNKNOWN/空（Range/对家建模退化——D 类，诚实记录）。

使用（CLI）：poker-ai analyze --frame <png> [--text <hh>]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FrameDecisionResult:
    status: str                        # OK | UNAVAILABLE | ERROR
    reason: str | None = None
    report: dict | None = None         # RealtimeReport.to_dict()
    bridge: dict | None = None
    warnings: list = field(default_factory=list)
    # P30-3-D D4 provenance：帧级决策可追溯性
    provenance: dict | None = None


def frame_to_decision(runtime: Any, frame_rgba, w: int, h: int, *,
                      hero_id: str = "Hero",
                      extra_request: dict | None = None) -> FrameDecisionResult:
    """frame → bridge → RealtimeRunner.decide(request)（同步 wrapper——
    非 async 上下文用；async 上下文用 frame_to_decision_async）。

    runtime = build_runtime()（复用既有实体——不新建 loop）。
    """
    import asyncio
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run_impl(runtime, frame_rgba, w, h,
                                     hero_id=hero_id,
                                     extra_request=extra_request))
    # 已在运行 loop（可疑调用）→ 返回明确错误而非嵌套 run
    return FrameDecisionResult(status="ERROR",
                               reason="running_event_loop_use_async_variant")


async def frame_to_decision_async(runtime: Any, frame_rgba, w: int, h: int,
                                  *, hero_id: str = "Hero",
                                  extra_request: dict | None = None
                                  ) -> FrameDecisionResult:
    """async 版 frame→decide（CLI async 上下文用——避免 asyncio.run 嵌套）。"""
    return await _run_impl(runtime, frame_rgba, w, h, hero_id=hero_id,
                           extra_request=extra_request)


async def _run_impl(runtime: Any, frame_rgba, w: int, h: int, *,
                    hero_id: str = "Hero",
                    extra_request: dict | None = None) -> FrameDecisionResult:
    from .vision2.pixel.realclient.production_bridge import (
        frame_to_decision_request)

    got = frame_to_decision_request(frame_rgba, w, h, hero_id=hero_id)
    if got["status"] != "OK":
        return FrameDecisionResult(status="UNAVAILABLE",
                                   reason=got["reason"],
                                   bridge=got["bridge"],
                                   warnings=got["bridge"].get(
                                       "warnings", []))
    request = got["request"]
    if extra_request:
        request = {**extra_request, **request}
    try:
        from .realtime import RealtimeRunner
        rr = RealtimeRunner(runtime)
        report = await rr.decide(request)
    except Exception as e:  # pragma: no cover - defensive
        return FrameDecisionResult(status="ERROR", reason=str(e)[:120],
                                   bridge=got["bridge"])
    # P30-3-D D4 provenance：帧级决策可追溯（frame→observation→bridge gate
    # →决策引擎）；UNKNOWN_ALLOWED 明确列出并说明为何不阻断。
    br = got["bridge"]
    unknown_allowed = br.get("unknown_fields", [])
    prov = {
        "source_frame_geometry": f"{int(w)}x{int(h)}",
        "observation_source": "v3f_snapshot",
        "bridge_status": br.get("status"),
        "mandatory_fields_ready": [f for f in (
            "street", "pot", "to_call", "hero_cards", "hero_seat", "blinds",
            "ante") if f in (br.get("field_confidences") or {})
            and (br.get("field_confidences") or {}).get(f) == 1.0],
        "unknown_allowed_fields": list(unknown_allowed),
        "unknown_allowed_rationale": (
            "该字段 UNKNOWN 不影响本决策：决策引擎不消费该值或消费后"
            "仅保守降级（不制造值/不伪造）——见 bridge warnings"),
        "decision_engine": "realtime",
        "reported": bool(report),
    }

    # P30-6-B：action_history provenance（合法事件源 vs UNKNOWN——
    # 单帧绝不猜历史）。
    from .action_event_source import (from_hand_history, from_event_stream,
                                      no_source)
    hh_from_request = (extra_request or {}).get("hand_history_text")
    ah_src = no_source()
    if hh_from_request:
        from .domain.poker.hand_history import parse_hand_history
        try:
            parsed = parse_hand_history(hh_from_request)
            ah_src = from_hand_history(parsed.actions)
        except Exception:
            ah_src = no_source()
    prov["action_history"] = ah_src.to_dict()

    # P30-4 D4-5 Judge 安全门（边界层——不改 Judge 内部）：
    # final_decision.action 必须 ∈ 合法动作集（由已知 DecisionInput 派生）。
    fd = (report.to_dict().get("final_decision") or {}) if report else {}
    action = fd.get("action")
    obs = got["request"].get("vision_observation") or {}
    hero_stack = None
    for seat in (obs.get("seats") or []):
        if isinstance(seat, dict) and seat.get("seat") == obs.get("hero_seat"):
            hero_stack = seat.get("stack")
            break
    from .decision_contract import (derive_legal_actions, validate_action)
    legal = derive_legal_actions(obs.get("to_call"), hero_stack,
                                 obs.get("pot"))
    if legal is None:
        # R14-EXE-3 方案 Y：bridge readiness OK + legal_actions UNKNOWN →
        # 参考建议通道（advice_reference——替换原硬
        # judge_safety_gate:legal_actions_unknown UNAVAILABLE）。
        # 语义红线：judge 输出不作为决策消费（单帧下其 to_call 输入是 V2
        # 布局缺省 0.0、语义 UNKNOWN）；参考打法 = 声明表/spot 库对该
        # street+手牌+pot 的标准推荐——advice_type/legal_actions_status/
        # action_validated=False 严格可区分（见 advice_reference 模块头）。
        from .advice_reference import (ADVICE_TYPE_REFERENCE,
                                       ADVICE_WARNING_ZH,
                                       LEGAL_ACTIONS_STATUS_UNKNOWN,
                                       advice_from_observation)
        prov["advice_type"] = ADVICE_TYPE_REFERENCE
        prov["legal_actions_status"] = LEGAL_ACTIONS_STATUS_UNKNOWN
        prov["legal_actions"] = None          # 显式 UNKNOWN（不推导）
        prov["judge_action_not_consumed"] = action   # 审计——绝不作为决策
        advice = advice_from_observation(obs)
        prov["advice_reference"] = advice
        adv_warnings = list(advice.get("warnings") or [ADVICE_WARNING_ZH])
        if advice.get("available"):
            prov["action"] = advice.get("action")
            return FrameDecisionResult(
                status="ADVICE_REFERENCE",
                reason="advice_reference:legal_actions_unknown",
                report=report.to_dict() if report else None,
                bridge=got["bridge"], provenance=prov,
                warnings=adv_warnings)
        # 参考通道 miss（场景未声明/表缺失/字段不足）→ 维持诚实拒绝
        # （绝不编造"标准打法"）；advice_type/legal_actions_status 保留
        # （该帧的语义状态仍如实可审计）。
        return FrameDecisionResult(
            status="DECISION_UNAVAILABLE",
            reason=f"advice_reference_miss:{advice.get('reason')}",
            report=report.to_dict() if report else None,
            bridge=got["bridge"], provenance=prov,
            warnings=adv_warnings)
    ok, why = validate_action(action, legal)
    if not ok:
        return FrameDecisionResult(
            status="DECISION_UNAVAILABLE",
            reason=f"judge_safety_gate:{why}",
            report=report.to_dict() if report else None,
            bridge=got["bridge"], provenance=prov)
    prov["action_validated"] = True
    prov["legal_actions"] = legal
    return FrameDecisionResult(status="OK", report=report.to_dict(),
                               bridge=got["bridge"], provenance=prov)


def frame_to_decision_sync_available() -> bool:
    """检测运行时可用（未配置 capture 等）——CLI 用。"""
    return True
