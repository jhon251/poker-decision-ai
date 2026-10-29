"""Realtime 快路径引擎（V2 Phase 7 / §十一）。

REALTIME 只使用快路径 Agent：
    State → Board → Statistics → Population/Player 缓存 → Fast Range → Fast Equity
    → Fast Solver(缓存) → Fast Bluff → Fast Exploit → Fast Risk → Fast ICM →
    Fast Bounty → Fast Judge → Deviation

不等待：Research / Hypothesis / Debate / Deep Solver / Deep Judge / Learning /
Evaluation / Emotion / Image / Strategy Evolution。

快路径手段（全部可配置，config.yaml realtime 块）：
  - equity_iterations 2000（完整路径 20000）
  - 单 Agent 超时 300ms；整路径硬时限 3000ms（超时降级继续，不崩溃）
  - 默认 allow_llm=false（确定性快引擎）；外部 LLM/Solver 不可用自动降级
  - 复用 GameModePlugin（模式隔离与 V1.1 一致）+ 全部 Cache

性能目标：P50 < 500ms，P90 < 1s（由 LatencyTracker 持续统计，验收时报告实测值）。

注意（诚实声明）：Realtime 的 "Fast Range / Fast Equity / Fast Bluff" 是同一批
确定性 Agent 在快参数 + 缓存下运行——V2 不重复造 Agent（spec §三十五）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .core.cache import CacheManager
from .core.errors import OrchestrationError
from .core.gamemode import enrich_from_state, resolve_game_context
from .core.orchestration.context import AgentContext
from .core.orchestration.engine import OrchestrationEngine
from .core.orchestration.plan import AgentTask, Plan, Stage
from .core.orchestration.runner import PipelineRunner, extract_final_decision
from .core.orchestration.runtime import Runtime
from .core.perf import LatencyTracker
from .core.schemas import new_id, utc_now_iso

#: Realtime 快路径允许的 Agent 集合（其余全部跳过）
FAST_AGENTS = {
    "vision", "state", "board", "statistics", "population", "opponent",
    "frequency_prior",
    "confidence", "range", "equity", "solver", "exploit", "bluff", "risk",
    "icm", "tournament_stage", "blind_pressure", "satellite",
    "bounty", "bounty_equity", "judge", "deviation",
}


@dataclass
class RealtimeReport:
    request_id: str
    status: str
    latency_ms: float
    final_decision: dict | None
    base_decision: dict | None
    deviation_applied: bool
    game_context: dict | None
    provider_status: dict | None
    agents_ran: list = field(default_factory=list)
    degraded: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    latency_snapshot: dict = field(default_factory=dict)
    path: str = "realtime"
    #: P30-9-GV：多人池（≥3 人）无 GTO 参考的**明确告知**（用户可见 ✓）
    multiway_note: dict | None = None
    #: P30-9-HH：翻后 to_call UNKNOWN 的**明确告知**（用户可见 ✓）
    to_call_note: dict | None = None
    #: V3G-2：CASH 状态完整性 **证据旁路**（不改任何决策数值）；
    #: 非 CASH 模式恒为 None（fail closed，不继承 CASH 专属字段）
    state_integrity: dict | None = None
    #: V3G-8：CASH UNKNOWN 端到端报告（只分类不填充；不改任何决策数值）
    unknown_report: dict | None = None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id, "path": self.path, "status": self.status,
            "latency_ms": self.latency_ms, "final_decision": self.final_decision, "multiway_note": self.multiway_note, "to_call_note": self.to_call_note,
            "base_decision": self.base_decision, "deviation_applied": self.deviation_applied,
            "game_context": self.game_context, "provider_status": self.provider_status,
            "agents_ran": self.agents_ran, "degraded": self.degraded,
            "skipped": self.skipped, "latency_snapshot": self.latency_snapshot,
            "state_integrity": self.state_integrity,
            "unknown_report": self.unknown_report,
        }


class RealtimeRunner:
    """REALTIME 决策路径：快参数 + 缓存 + 紧超时 + 确定性降级。"""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.tracker = LatencyTracker("realtime")
        self.engine = OrchestrationEngine(runtime.registry)
        self.mode_registry = runtime.runner.mode_registry

    def _fast_plan(self, plugin) -> Plan:
        """从模式插件派生快路径计划：过滤 FAST_AGENTS，应用紧超时。"""
        rt_cfg = self.runtime.config.realtime_config()
        timeout_ms = int(rt_cfg.get("agent_timeout_ms", 300))
        # P30-9-DG：solver 走**库查表**（预热后 ~ms，但冷启动需 150–300ms）⇒ 单独放宽，
        # 否则实时决策里 solver 常因超时降级 ⇒ judge 拿不到求解器通道（实测复现）。
        solver_timeout_ms = int(rt_cfg.get("solver_timeout_ms", 1200))
        stages: list[Stage] = []
        for stage in plugin.plan_stages():
            tasks = []
            for t in stage.tasks:
                if t.agent in FAST_AGENTS:
                    _to = (solver_timeout_ms if t.agent == "solver" else timeout_ms)
                    tasks.append(AgentTask(agent=t.agent, input_template=t.input_template,
                                           timeout_ms=_to,
                                           retries=t.retries, on_failure=t.on_failure))
            if tasks:
                stages.append(Stage(stage.name, tasks))
        return Plan(f"realtime:{plugin.spec()}", stages)

    async def decide(self, request: dict) -> RealtimeReport:
        t0 = time.perf_counter()
        # P30-9-HP：**请求边界** —— 本次请求内缓存的新写入不可见 ✓（同输入同输出 ✓）
        try:
            from .core.providers.spot_table_6max import begin_request
            begin_request()
        except Exception:                          # noqa: BLE001
            pass
        rt_cfg = self.runtime.config.realtime_config()
        game_context = resolve_game_context(request, self.runtime.config)
        plugin = self.mode_registry.resolve(game_context)
        plan = self._fast_plan(plugin)

        ctx = self.runtime.new_context(request.get("request_id") or new_id("rt"),
                                       request.get("session_id") or "realtime",
                                       allow_llm=bool(rt_cfg.get("allow_llm", False)))
        ctx.game_context = game_context
        ctx.extra["realtime"] = rt_cfg            # equity 等 Agent 读取快参数
        caches = self.runtime.caches or CacheManager()
        ctx.caches = caches
        ctx.extra["solver_cache"] = caches.solver

        deadline_ms = int(rt_cfg.get("deadline_ms", 3000))
        engine = OrchestrationEngine(self.runtime.registry)
        try:
            result = await __import__("asyncio").wait_for(
                engine.execute(plan, ctx, request), timeout=deadline_ms / 1000.0)
        except __import__("asyncio").TimeoutError:
            raise OrchestrationError(f"realtime pipeline exceeded deadline {deadline_ms}ms")

        state_out = (result.results["state"].envelope.output
                     if "state" in result.results and result.results["state"].envelope else None)
        if state_out:
            enrich_from_state(game_context, state_out)

        base, deviation, deviation_applied, final = extract_final_decision(result)
        total_ms = (time.perf_counter() - t0) * 1000.0
        self.tracker.record(total_ms)
        for name, r in result.results.items():
            if r.envelope and r.envelope.latency_ms is not None:
                self.tracker.record_part(name, r.envelope.latency_ms)

        # ── V3G-2：CASH 状态完整性 **证据旁路**（只读；不改任何决策数值）──
        # 在决策已完成后计算并附加；失败不影响决策；非 CASH → present=False。
        integrity: dict | None = None
        try:
            from .cash_integrity_context import cash_integrity_context
            _ctx = cash_integrity_context(state_out)
            if _ctx.get("present"):
                integrity = _ctx
        except Exception:                     # 旁路失败绝不阻断决策
            integrity = None

        # ── V3G-8：CASH UNKNOWN 端到端报告（只分类不填充；只读）──────────
        unknown_report: dict | None = None
        try:
            from .cash_unknown_e2e import build_unknown_report
            _ur = build_unknown_report(state_out)
            if _ur.get("present"):
                unknown_report = _ur
        except Exception:                     # 旁路失败绝不阻断决策
            unknown_report = None

        # ── P30-9-GV：**多人池（≥3 人）无 GTO 参考 ⇒ 明确告知**（不得看起来像正常建议 ✗）──
        #   取证：此前唯一提示藏在 ev_core.assumptions[2]，用户可见的建议里没有说明 ✗。
        multiway_note: dict | None = None
        try:
            _players = (state_out or {}).get("players") or []
            _act = [p for p in _players if isinstance(p, dict)
                    and p.get("status") in ("active", "allin")]
            if len(_act) >= 3:
                multiway_note = {
                    "n_active": len(_act),
                    "text": (f"多人池（{len(_act)} 人）：**没有多人 GTO 参考** ⇒ "
                              "本建议基于单对手近似与启发式，仅供参考；不构成多人均衡建议"),
                    "code": "MULTIWAY_NO_REFERENCE",
                }
                if isinstance(final, dict):
                    _r = str(final.get("reasoning") or "")
                    if multiway_note["text"] not in _r:
                        final["reasoning"] = (_r + "；" if _r else "") + multiway_note["text"]
        except Exception:                     # 旁路失败绝不阻断决策
            multiway_note = None

        # ── P30-9-HH：**翻后 to_call UNKNOWN ⇒ 必须明确告知**（不得当成 first_in 静默处理 ✗）──
        #   取证：41 个真实翻后请求里 25 个 to_call 为 UNKNOWN（完整牌面 19/22 ✗）
        #   ⇒ 引擎无法区分"面对下注 / 首行动"，却照常给建议 ✗。
        to_call_note: dict | None = None
        try:
            _st_x = str((state_out or {}).get("street") or "")
            # P30-9-HH：用**原始观测**判定 —— 状态层会把"未知"强制成 0.0 ✗
            #   实测：观测里不给 to_call 时 state.to_call == 0.0（无法用状态判定 ✗）。
            _obs_x = request.get("vision_observation")
            _tc_raw = (_obs_x.get("to_call", "__absent__")
                        if isinstance(_obs_x, dict) else "__absent__")
            if _st_x in ("flop", "turn", "river") and (_tc_raw is None
                                                         or _tc_raw == "__absent__"):
                to_call_note = {
                    "code": "TO_CALL_UNKNOWN",
                    "street": _st_x,
                    "text": (f"{_st_x}：**未能判定是否面对下注**（动作面板/投注额未识别）⇒ "
                              "本建议按**未面对下注**处理；若实际面对下注，请重拍或手动确认投注额"),
                }
                if isinstance(final, dict):
                    _r2 = str(final.get("reasoning") or "")
                    if to_call_note["text"] not in _r2:
                        final["reasoning"] = (_r2 + "；" if _r2 else "") + to_call_note["text"]
        except Exception:                     # 旁路失败绝不阻断决策
            to_call_note = None

        # 决策持久化（path=realtime；与 Deep 路径互不覆盖）
        caches = self.runtime.caches or caches
        try:
            ctx.repos.decisions.save({
                "request_id": ctx.request_id, "session_id": ctx.session_id,
                "path": "realtime",
                "kind": request.get("kind", "realtime"),
                "game_context": game_context.to_dict(),
                "provider_status": ctx.providers.status_summary(),
                "base_decision": base, "deviated_decision": deviation,
                "final_decision": final,
                "deviation_applied": deviation_applied,
                "model_versions": dict(caches.model_versions),
                "latency_ms": round(total_ms, 2),
                "state_integrity": integrity,
                "unknown_report": unknown_report,
                "created_at": utc_now_iso(),
            })
        except Exception:
            pass    # 决策持久化失败不阻塞快路径

        return RealtimeReport(
            request_id=ctx.request_id, status=result.status,
            latency_ms=round(total_ms, 2),
            final_decision=final, base_decision=base, multiway_note=multiway_note, to_call_note=to_call_note,
            deviation_applied=deviation_applied,
            game_context=game_context.to_dict(),
            provider_status=ctx.providers.status_summary(),
            agents_ran=[n for n in result.order if result.results[n].status in ("ok", "degraded")],
            degraded=result.degraded_agents(), skipped=result.skipped_agents(),
            latency_snapshot=self.tracker.snapshot(),
            state_integrity=integrity,
            unknown_report=unknown_report,
        )
