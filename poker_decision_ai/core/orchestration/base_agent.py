"""BaseAgent: the single contract every agent implements.

Lifecycle of one invocation (async):
    raw_input (JSON dict)
      -> input-schema validation   (dataclass Input, strict, deterministic)
      -> handle(input, ctx)        (agent logic; deterministic + optional LLM)
      -> output-schema validation  (dataclass Output, strict)
      -> AgentEnvelope             (uniform protocol; logging + metrics)

Agents annotate optional meta fields on their Output dataclass:
    confidence: float | None   -> envelope.confidence
    evidence:   list           -> envelope.evidence
    warnings:   list           -> envelope.warnings
Everything is recorded (input, output, latency, status) for later
evaluation of "which agent actually helped?".
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from ..errors import AgentExecutionError, AgentOutputError, AgentValidationError, SchemaError
from ..schemas import (
    AgentEnvelope,
    ErrorEntry,
    WarningEntry,
    EvidenceItem,
    from_dict,
    json_schema_for,
    utc_now_iso,
    validate_instance,
)
from .context import AgentContext


@dataclass
class AgentMeta:
    """Static description of an agent (registry + docs generation)."""

    name: str
    version: str
    zh_name: str
    role: str
    layer: str                      # orchestration|perception|analysis|human|tournament|learning|decision|experiment
    depends: list
    description: str = ""


class BaseAgent(ABC):
    """Abstract base for all 31 agents."""

    #: stable agent identifier (used in pipeline.yaml)
    name: ClassVar[str] = "base"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = ""
    layer: ClassVar[str] = "core"
    #: names of agents whose outputs this agent consumes (documentation + validation)
    depends: ClassVar[list[str]] = []
    #: 能力声明（V1.1 可插拔模式架构）：模式规格列表，如
    #:   ["*"] / ["tournament", "bounty"] / ["tournament:satellite"] / ["bounty:pko"]
    #: 引擎根据 ctx.game_context 自动跳过不支持的 Agent（Agent Capability Registry）。
    supported_modes: ClassVar[list[str]] = ["*"]
    #: input / output contracts as dataclasses (schemas are derived from them)
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = dict

    def __init__(self) -> None:
        if self.Input is dict:
            self.Input = dict  # allow plain dict input (Manager only)
        self._input_schema = json_schema_for(self.Input) if self.Input is not dict else {"type": "object"}
        self._output_schema = json_schema_for(self.Output) if self.Output is not dict else {"type": "object"}

    # ---- schemas ----
    def input_schema(self) -> dict:
        return self._input_schema

    def output_schema(self) -> dict:
        return self._output_schema

    def meta(self) -> AgentMeta:
        return AgentMeta(self.name, self.version, self.zh_name, self.__doc__ or "", self.layer, list(self.depends))

    # ---- dynamic applicability ----
    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        """Manager calls this to decide whether to run the agent at all.

        Default: always applicable. Agents override to skip themselves when
        their required inputs are missing (e.g. ICM Agent outside a
        tournament, Satellite Agent in a cash game).
        """
        return True, ""

    # ---- core logic ----
    @abstractmethod
    async def handle(self, inp: Any, ctx: AgentContext) -> Any:
        """Process a validated Input instance; return an Output instance."""

    # ---- uniform runner ----
    async def run(self, raw_input: dict, ctx: AgentContext) -> AgentEnvelope:
        t0 = time.perf_counter()
        envelope = AgentEnvelope(
            request_id=ctx.request_id, session_id=ctx.session_id, agent=self.name,
            version=self.version, timestamp=utc_now_iso(), input=raw_input,
        )

        # 1) input validation (deterministic)
        try:
            inp = from_dict(self.Input, raw_input) if self.Input is not dict else raw_input
        except SchemaError as e:
            envelope.status = "error"
            envelope.errors = [ErrorEntry(AgentValidationError.code, str(e), self.name).to_dict()]
            self._record(envelope, t0, ctx)
            raise AgentValidationError(str(e), detail={"agent": self.name})

        # 2) execute
        try:
            out = await self.handle(inp, ctx)
        except Exception as e:
            envelope.status = "error"
            code = getattr(e, "code", AgentExecutionError.code)
            envelope.errors = [ErrorEntry(code, str(e), self.name).to_dict()]
            self._record(envelope, t0, ctx)
            raise AgentExecutionError(f"{self.name}: {e}", detail={"agent": self.name}) from e

        # 3) output validation (deterministic)
        try:
            data = validate_instance(self.Output, out) if self.Output is not dict else (out if isinstance(out, dict) else {})
        except SchemaError as e:
            envelope.status = "error"
            envelope.errors = [ErrorEntry(AgentOutputError.code, str(e), self.name).to_dict()]
            self._record(envelope, t0, ctx)
            raise AgentOutputError(str(e), detail={"agent": self.name})

        envelope.output = data
        envelope.status = "ok"
        envelope.confidence = data.get("confidence") if isinstance(data, dict) else None
        ev = data.get("evidence") if isinstance(data, dict) else None
        if ev:
            envelope.evidence = [e.to_dict() if isinstance(e, EvidenceItem) else e for e in ev]
        warn = data.get("warnings") if isinstance(data, dict) else None
        if warn:
            envelope.warnings = [w.to_dict() if isinstance(w, WarningEntry) else w for w in warn]
        if envelope.warnings:
            envelope.status = "degraded"
        self._record(envelope, t0, ctx)
        return envelope

    def _record(self, envelope: AgentEnvelope, t0: float, ctx: AgentContext) -> None:
        envelope.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        ctx.metrics.record_call(self.name, envelope.status, envelope.latency_ms)
        ctx.logger.log_agent(envelope)
