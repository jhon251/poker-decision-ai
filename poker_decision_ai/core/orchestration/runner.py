"""PipelineRunner: end-to-end request execution (Manager's workhorse).

Flow for one hand request (V1.1 可插拔游戏模式架构):
    request (hand history / vision observation / replay record)
      -> resolve_game_context(request, config)      # 显式 CLI/config > 手牌推断 > kind > 默认
      -> ModeRegistry.resolve(game_context) -> GameModePlugin
      -> plugin.plan_stages() -> Plan               # 模式自动装配 Agent 集合
      -> OrchestrationEngine.execute                # 能力过滤 + applicable + 并行调度
      -> final decision extraction                  # judge -> deviation -> final
      -> decision persisted + logged
      -> PipelineReport

The report keeps base_decision vs deviated_decision permanently separated
(spec §I) so Evaluation can judge the raw strategy and the deviation
experiments independently.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..config import AppConfig
from ..errors import OrchestrationError
from ..gamemode import ModeRegistry, enrich_from_state, register_default_modes, resolve_game_context
from ..logging import EventLogger, MetricsCollector
from ..schemas import new_id
from .context import AgentContext
from .engine import EngineResult, OrchestrationEngine
from .registry import AgentRegistry


@dataclass
class PipelineReport:
    request_id: str
    session_id: str
    kind: str
    status: str
    latency_ms: float
    final_decision: dict | None
    base_decision: dict | None
    deviated_decision: dict | None
    deviation_applied: bool
    envelopes: dict[str, dict] = field(default_factory=dict)
    degraded: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    game_context: dict | None = None                # V1.1 模式上下文快照
    provider_status: dict | None = None             # V2 诚实能力状态（MOCK/REAL/UNAVAILABLE）

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id, "session_id": self.session_id, "kind": self.kind,
            "status": self.status, "latency_ms": self.latency_ms,
            "final_decision": self.final_decision, "base_decision": self.base_decision,
            "deviated_decision": self.deviated_decision, "deviation_applied": self.deviation_applied,
            "envelopes": self.envelopes, "degraded": self.degraded, "skipped": self.skipped,
            "metrics": self.metrics, "game_context": self.game_context,
            "provider_status": self.provider_status,
        }


def extract_final_decision(result: EngineResult) -> tuple:
    """从引擎结果提取 (base, deviation, deviation_applied, final)。

    canonical final shape == judge shape（action/recommended_size/...），所有消费方
    （CLI/demo/evaluation/realtime）读同一契约；原始偏离输出保留在 deviation 供审计。
    Realtime/Deep 路径复用本函数保证 base/deviated 永久分离。
    """
    base = (result.results["judge"].envelope.output
            if "judge" in result.results and result.results["judge"].envelope else None)
    deviation = None
    deviation_applied = False
    if "deviation" in result.results and result.results["deviation"].envelope:
        deviation = result.results["deviation"].envelope.output
        deviation_applied = bool(deviation and deviation.get("deviated"))
    final = base
    if deviation_applied and deviation:
        final = dict(base or {})
        final["action"] = deviation.get("final_action")
        final["recommended_size"] = deviation.get("final_size")
        final["deviation"] = {
            "deviated": True,
            "deviation_probability": deviation.get("deviation_probability"),
            "original_action": deviation.get("original_action"),
            "original_size": deviation.get("original_size"),
        }
    return base, deviation, deviation_applied, final


class PipelineRunner:
    def __init__(self, registry: AgentRegistry, config: AppConfig,
                 logger: EventLogger, metrics: MetricsCollector) -> None:
        self.registry = registry
        self.config = config
        self.logger = logger
        self.metrics = metrics
        self.engine = OrchestrationEngine(registry)
        self.mode_registry: ModeRegistry = register_default_modes(ModeRegistry())

    async def run(self, request: dict, ctx: AgentContext | None = None) -> PipelineReport:
        """Execute one request end-to-end."""
        t0 = time.perf_counter()
        kind = str(request.get("kind") or "cash_hand")
        if ctx is None:
            raise OrchestrationError("PipelineRunner.run requires an AgentContext (use RuntimeBuilder.build_runtime())")

        # ---- V1.1 模式解析：GameContext -> Plugin -> Plan（无模式 if/else） ----
        game_context = resolve_game_context(request, self.config)
        plugin = self.mode_registry.resolve(game_context)
        from poker_decision_ai.domain.poker.gamemodes.base import materialize_plan

        plan = materialize_plan(plugin.plan_kind(), plugin.plan_stages(), self.config)
        ctx.game_context = game_context          # 全系统标准输入：所有 Agent 可见
        result: EngineResult = await self.engine.execute(plan, ctx, request)

        # 状态落地后充实 GameContext（盲注/筹码/赏金/阶段），并写回报告
        state_out = (result.results["state"].envelope.output
                     if "state" in result.results and result.results["state"].envelope else None)
        if state_out:
            enrich_from_state(game_context, state_out)

        # ---- final decision extraction ----
        base, deviation, deviation_applied, final = extract_final_decision(result)

        # ---- persistence + logging (decision must NEVER be lost) ----
        if final:
            self.logger.log_decision(ctx.request_id, ctx.session_id,
                                     final.get("action"), final.get("recommended_size"),
                                     final.get("confidence", 0.0), "deviated" if deviation_applied else "judge")
            if deviation and deviation_applied:
                self.logger.log_deviation(
                    ctx.request_id, ctx.session_id,
                    float(deviation.get("deviation_probability", 0.0)), True,
                    str(deviation.get("original_action")), str(deviation.get("final_action")))
            try:
                self._persist_decision(request, ctx, base, deviation, final, result, game_context)
            except Exception as e:  # decision repo failure must not crash the reply
                self.logger.log_error(ctx.request_id, "runner", "REPOSITORY_ERROR", str(e))

        report = PipelineReport(
            request_id=ctx.request_id, session_id=ctx.session_id, kind=kind,
            status=result.status, latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            final_decision=final, base_decision=base, deviated_decision=deviation,
            deviation_applied=deviation_applied,
            envelopes=result.envelopes(),
            degraded=result.degraded_agents(), skipped=result.skipped_agents(),
            metrics=self.metrics.snapshot(),
            game_context=game_context.to_dict(),
            provider_status=ctx.providers.status_summary(),   # V2 诚实能力状态
        )
        self.logger.log("pipeline", request_id=ctx.request_id, kind=kind, status=report.status,
                        latency_ms=report.latency_ms, degraded=report.degraded,
                        skipped=report.skipped, game_mode=game_context.spec())
        return report

    def _persist_decision(self, request: dict, ctx: AgentContext,
                          base: dict | None, deviation: dict | None,
                          final: dict | None, result: EngineResult,
                          game_context) -> None:
        from ..schemas import utc_now_iso

        ctx.repos.decisions.save({
            "request_id": ctx.request_id, "session_id": ctx.session_id,
            "kind": request.get("kind", "cash_hand"),
            "state": (result.results.get("state").envelope.output if result.results.get("state") and result.results["state"].envelope else None),
            "base_decision": base, "deviated_decision": deviation, "final_decision": final,
            "deviation_applied": bool(deviation and deviation.get("deviated")),
            "game_context": game_context.to_dict(),
            "provider_status": ctx.providers.status_summary(),
            "degraded_agents": result.degraded_agents(), "skipped_agents": result.skipped_agents(),
            "created_at": utc_now_iso(),
        })
