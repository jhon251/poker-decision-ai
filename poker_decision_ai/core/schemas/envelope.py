"""Unified Agent Envelope - the single communication protocol for all agents.

Every agent invocation in the framework is wrapped in an :class:`AgentEnvelope`:

    request_id   - unique id of this top-level request (one hand = one request)
    session_id   - table / study session id
    agent        - agent name
    version      - agent version (schema evolution)
    timestamp    - ISO-8601 UTC
    input/output - JSON dicts validated against the agent's schemas
    confidence   - agent's own confidence in its output (0..1 or null)
    evidence     - machine-readable evidence items the agent relied on
    warnings     - degradations / missing data the agent noticed
    errors       - structured errors (code + message + agent)
    status       - ok | degraded | error
    latency_ms   - measured processing latency

Natural language is only allowed inside explanation/note fields of an agent's
OUTPUT SCHEMA.  Agent-to-agent communication is always this structured envelope.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def new_id(prefix: str = "req") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class EvidenceItem:
    """A single machine-readable piece of evidence an agent used."""

    kind: str            # e.g. "population_stat" | "player_stat" | "range" | "solver"
    text: str            # short factual statement
    source: str = ""     # e.g. "population:NL10:CO" | "player:abc123"
    weight: float = 1.0  # 0..1, how strongly this evidence influenced the agent
    def to_dict(self) -> dict: return {"kind": self.kind, "text": self.text, "source": self.source, "weight": self.weight}
    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceItem": return cls(str(d.get("kind", "")), str(d.get("text", "")), str(d.get("source", "")), float(d.get("weight", 1.0)))


@dataclass
class WarningEntry:
    code: str
    message: str
    agent: str = ""
    def to_dict(self) -> dict: return {"code": self.code, "message": self.message, "agent": self.agent}
    @classmethod
    def from_dict(cls, d: dict) -> "WarningEntry": return cls(str(d.get("code", "WARN")), str(d.get("message", "")), str(d.get("agent", "")))


@dataclass
class ErrorEntry:
    code: str
    message: str
    agent: str = ""
    detail: dict = field(default_factory=dict)
    def to_dict(self) -> dict: return {"code": self.code, "message": self.message, "agent": self.agent, "detail": self.detail}
    @classmethod
    def from_dict(cls, d: dict) -> "ErrorEntry": return cls(str(d.get("code", "ERROR")), str(d.get("message", "")), str(d.get("agent", "")), dict(d.get("detail") or {}))


@dataclass
class AgentEnvelope:
    request_id: str
    session_id: str
    agent: str
    version: str
    timestamp: str
    input: dict = field(default_factory=dict)
    output: dict | None = None
    confidence: float | None = None
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    status: str = "ok"          # ok | degraded | error
    latency_ms: float | None = None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "agent": self.agent,
            "version": self.version,
            "timestamp": self.timestamp,
            "input": self.input,
            "output": self.output,
            "confidence": self.confidence,
            "evidence": [e.to_dict() if isinstance(e, EvidenceItem) else e for e in self.evidence],
            "warnings": [w.to_dict() if isinstance(w, WarningEntry) else w for w in self.warnings],
            "errors": [e.to_dict() if isinstance(e, ErrorEntry) else e for e in self.errors],
            "status": self.status,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AgentEnvelope":
        return cls(
            request_id=str(d.get("request_id", "")),
            session_id=str(d.get("session_id", "")),
            agent=str(d.get("agent", "")),
            version=str(d.get("version", "")),
            timestamp=str(d.get("timestamp", "")),
            input=dict(d.get("input") or {}),
            output=d.get("output"),
            confidence=d.get("confidence"),
            evidence=[EvidenceItem.from_dict(e) if isinstance(e, dict) else e for e in d.get("evidence") or []],
            warnings=[WarningEntry.from_dict(w) if isinstance(w, dict) else w for w in d.get("warnings") or []],
            errors=[ErrorEntry.from_dict(e) if isinstance(e, dict) else e for e in d.get("errors") or []],
            status=str(d.get("status", "ok")),
            latency_ms=d.get("latency_ms"),
        )


ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "AgentEnvelope",
    "properties": {
        "request_id": {"type": "string"},
        "session_id": {"type": "string"},
        "agent": {"type": "string"},
        "version": {"type": "string"},
        "timestamp": {"type": "string"},
        "input": {"type": "object"},
        "output": {"type": ["object", "null"]},
        "confidence": {"type": ["number", "null"], "minimum": 0.0, "maximum": 1.0},
        "evidence": {"type": "array", "items": {"type": "object"}},
        "warnings": {"type": "array", "items": {"type": "object"}},
        "errors": {"type": "array", "items": {"type": "object"}},
        "status": {"type": "string", "enum": ["ok", "degraded", "error"]},
        "latency_ms": {"type": ["number", "null"]},
    },
    "required": ["request_id", "session_id", "agent", "version", "timestamp", "status"],
    "additionalProperties": False,
}
