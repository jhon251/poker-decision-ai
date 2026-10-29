"""Deep / Background 路径（V2 Phase 8 / §十二·十三）。

Realtime 决策产生后，后台异步运行完整 Pipeline（全 Agent + Deep 模型）：
    Deep Range → Deep Bluff → Deep Exploit → Deep Solver → Debate →
    Deep Judge → Evaluation → Learning → Research → Hypothesis → Experiment

关键约束：
  1. 后台结果**不能修改**已产生的 Realtime 决策——只产生 Decision Revision Record。
  2. Realtime 决策记录（path=realtime）永不被覆盖；revision 是独立记录。
  3. revision 字段：realtime_action / deep_action / reason / evidence /
     confidence_change / model_versions（§十三）。
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from .core.errors import OrchestrationError
from .core.gamemode import resolve_game_context
from .core.orchestration.engine import OrchestrationEngine
from .core.orchestration.plan import AgentTask, Plan, Stage
from .core.orchestration.runner import extract_final_decision
from .core.orchestration.runtime import Runtime
from .core.perf import LatencyTracker
from .core.schemas import new_id, utc_now_iso


@dataclass
class DecisionRevision:
    """Realtime 与 Deep 的差异记录（只记录，绝不覆盖 realtime 结果）。"""

    revision_id: str
    request_id: str
    realtime_action: str
    deep_action: str
    realtime_confidence: float | None
    deep_confidence: float | None
    confidence_change: float | None
    reason: str
    evidence: list = field(default_factory=list)
    model_versions: dict = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "revision_id": self.revision_id, "request_id": self.request_id,
            "realtime_action": self.realtime_action, "deep_action": self.deep_action,
            "realtime_confidence": self.realtime_confidence,
            "deep_confidence": self.deep_confidence,
            "confidence_change": self.confidence_change,
            "reason": self.reason, "evidence": list(self.evidence),
            "model_versions": dict(self.model_versions),
            "created_at": self.created_at or utc_now_iso(),
        }


@dataclass
class DeepReport:
    request_id: str
    status: str
    latency_ms: float
    final_decision: dict | None
    base_decision: dict | None
    revision: dict | None
    agents_ran: list = field(default_factory=list)
    degraded: list = field(default_factory=list)
    provider_status: dict | None = None
    path: str = "deep"

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id, "path": self.path, "status": self.status,
            "latency_ms": self.latency_ms, "final_decision": self.final_decision,
            "base_decision": self.base_decision, "revision": self.revision,
            "agents_ran": self.agents_ran, "degraded": self.degraded,
            "provider_status": self.provider_status,
        }


class DeepRunner:
    """DEEP 后台路径：完整 Plugin Pipeline + Deep LLM 模型 + 宽松超时。"""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.tracker = LatencyTracker("deep")
        self.mode_registry = runtime.runner.mode_registry

    def _deep_plan(self, plugin) -> Plan:
        deep_cfg = self.runtime.config.deep_config()
        timeout_ms = int(deep_cfg.get("agent_timeout_ms", 30000))
        stages = []
        for stage in plugin.plan_stages():
            tasks = [AgentTask(agent=t.agent, input_template=t.input_template,
                               timeout_ms=timeout_ms, retries=t.retries,
                               on_failure=t.on_failure)
                     for t in stage.tasks]
            stages.append(Stage(stage.name, tasks))
        return Plan(f"deep:{plugin.spec()}", stages)

    async def deepen(self, request: dict, realtime_report) -> DeepReport:
        """对同一请求执行 Deep 路径，并与 realtime 决策对比生成 revision。"""
        t0 = time.perf_counter()
        game_context = resolve_game_context(request, self.runtime.config)
        plugin = self.mode_registry.resolve(game_context)
        plan = self._deep_plan(plugin)

        ctx = self.runtime.new_context(new_id("deep"), request.get("session_id") or "deep",
                                       allow_llm=bool(self.runtime.config.deep_config().get("allow_llm", True)))
        ctx.game_context = game_context
        ctx.extra["deep"] = self.runtime.config.deep_config()
        ctx.extra["solver_cache"] = (self.runtime.caches or None)

        engine = OrchestrationEngine(self.runtime.registry)
        result = await engine.execute(plan, ctx, request)
        base, deviation, deviation_applied, final = extract_final_decision(result)
        total_ms = (time.perf_counter() - t0) * 1000.0
        self.tracker.record(total_ms)

        # ---- Decision Revision：realtime vs deep（比较 base 决策，避免偏差噪声） ----
        revision = None
        caches = self.runtime.caches
        model_versions = dict(caches.model_versions) if caches else {}
        rt_base = (realtime_report.base_decision if realtime_report is not None else None) or {}
        if base is not None and rt_base:
            rt_action = rt_base.get("action")
            deep_action = base.get("action")
            if rt_action != deep_action:
                rt_conf = rt_base.get("confidence")
                dp_conf = base.get("confidence")
                change = (float(dp_conf) - float(rt_conf)) if (rt_conf is not None and dp_conf is not None) else None
                revision = DecisionRevision(
                    revision_id=f"{realtime_report.request_id}:revision:deep",
                    request_id=realtime_report.request_id,
                    realtime_action=str(rt_action), deep_action=str(deep_action),
                    realtime_confidence=rt_conf, deep_confidence=dp_conf,
                    confidence_change=change,
                    reason=(f"Deep 路径复核：{rt_action} → {deep_action}（base 决策比较，"
                            f"后台结果不覆盖 realtime）"),
                    evidence=[str(e) for e in (base.get("key_evidence") or [])[:3]],
                    model_versions=model_versions,
                )
                try:
                    ctx.repos.decisions.save({
                        "request_id": revision.revision_id,
                        "path": "revision",
                        "revision": revision.to_dict(),
                        "created_at": utc_now_iso(),
                    })
                except Exception:
                    pass    # revision 持久化失败不阻塞 deep 路径

        # Deep 决策记录（独立 request_id，不覆盖 realtime 记录）
        try:
            ctx.repos.decisions.save({
                "request_id": ctx.request_id, "session_id": ctx.session_id,
                "path": "deep",
                "kind": request.get("kind", "deep"),
                "game_context": game_context.to_dict(),
                "provider_status": ctx.providers.status_summary(),
                "base_decision": base, "deviated_decision": deviation,
                "final_decision": final, "deviation_applied": deviation_applied,
                "revision": revision.to_dict() if revision else None,
                "model_versions": model_versions,
                "latency_ms": round(total_ms, 2),
                "created_at": utc_now_iso(),
            })
        except Exception:
            pass

        return DeepReport(
            request_id=ctx.request_id, status=result.status,
            latency_ms=round(total_ms, 2), final_decision=final, base_decision=base,
            revision=revision.to_dict() if revision else None,
            agents_ran=[n for n in result.order if result.results[n].status in ("ok", "degraded")],
            degraded=result.degraded_agents(),
            provider_status=ctx.providers.status_summary(),
        )

    def spawn(self, request: dict, realtime_report) -> "asyncio.Task":
        """后台异步运行 Deep 路径（不阻塞调用方）。"""
        return asyncio.create_task(self.deepen(request, realtime_report))
