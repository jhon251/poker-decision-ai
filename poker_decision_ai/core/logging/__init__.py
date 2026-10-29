"""Structured event logging (JSON Lines) + in-process metrics.

Events recorded (so that later research can answer "which agent actually
helped?"):
  - agent_invocation : agent name, input, output, latency, status, warnings, errors
  - llm_usage        : model, tokens in/out, cost
  - decision         : final decision (judge + deviation, base vs deviated)
  - deviation        : deviation triggered (probability, original vs final)
  - outcome          : actual result of a hand (for evaluation/learning)
  - retry            : agent retried after failure
  - replay           : replay run summary
  - error            : top-level errors

The EventLogger appends one JSON object per line to a daily JSONL file, so
logs are queryable with standard tools (jq, pandas, duckdb, ...).
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..schemas.envelope import AgentEnvelope

_LOGGER = logging.getLogger("poker_ai")


@dataclass
class AgentMetrics:
    calls: int = 0
    ok: int = 0
    degraded: int = 0
    errors: int = 0
    timeouts: int = 0
    retries: int = 0
    skipped: int = 0
    total_latency_ms: float = 0.0
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    llm_cost_usd: float = 0.0

    def snapshot(self) -> dict:
        return {
            "calls": self.calls, "ok": self.ok, "degraded": self.degraded,
            "errors": self.errors, "timeouts": self.timeouts, "retries": self.retries,
            "skipped": self.skipped,
            "avg_latency_ms": round(self.total_latency_ms / self.calls, 2) if self.calls else 0.0,
            "llm_tokens_in": self.llm_tokens_in, "llm_tokens_out": self.llm_tokens_out,
            "llm_cost_usd": round(self.llm_cost_usd, 6),
        }


class MetricsCollector:
    """Thread-safe per-agent metrics accumulator.

    Uses a reentrant lock: record_* methods hold the lock while calling
    _get(), which also locks (a plain Lock would self-deadlock).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._agents: dict[str, AgentMetrics] = {}

    def _get(self, name: str) -> AgentMetrics:
        with self._lock:
            return self._agents.setdefault(name, AgentMetrics())

    def record_call(self, agent: str, status: str, latency_ms: float) -> None:
        m = self._get(agent)
        with self._lock:
            m.calls += 1
            m.total_latency_ms += latency_ms
            if status == "ok":
                m.ok += 1
            elif status == "degraded":
                m.degraded += 1
            else:
                m.errors += 1

    def record_timeout(self, agent: str) -> None:
        with self._lock:
            self._get(agent).timeouts += 1

    def record_retry(self, agent: str) -> None:
        with self._lock:
            self._get(agent).retries += 1

    def record_skipped(self, agent: str) -> None:
        with self._lock:
            self._get(agent).skipped += 1

    def record_llm(self, agent: str, tokens_in: int, tokens_out: int, cost_usd: float) -> None:
        m = self._get(agent)
        with self._lock:
            m.llm_tokens_in += tokens_in
            m.llm_tokens_out += tokens_out
            m.llm_cost_usd += cost_usd

    def snapshot(self) -> dict:
        with self._lock:
            return {name: m.snapshot() for name, m in sorted(self._agents.items())}

    def reset(self) -> None:
        with self._lock:
            self._agents.clear()


class EventLogger:
    """Append-only JSONL event logger with size-based rotation."""

    def __init__(self, logs_dir: str, max_mb: int = 50) -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_mb * 1024 * 1024
        self._lock = threading.Lock()

    def _path(self) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.logs_dir / f"events-{day}.jsonl"

    def _rotate(self) -> None:
        p = self._path()
        if p.exists() and p.stat().st_size > self.max_bytes:
            i = 1
            while (self.logs_dir / f"{p.name}.{i}").exists():
                i += 1
            p.rename(self.logs_dir / f"{p.name}.{i}")

    def log(self, event: str, **fields: Any) -> None:
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "event": event}
        record.update(fields)
        import json

        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            try:
                self._rotate()
                with open(self._path(), "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError as e:  # logging must never crash the pipeline
                _LOGGER.warning("event log write failed: %s", e)

    # ---- semantic helpers ----
    def log_agent(self, env: AgentEnvelope) -> None:
        self.log(
            "agent_invocation",
            request_id=env.request_id, session_id=env.session_id, agent=env.agent,
            version=env.version, status=env.status, latency_ms=env.latency_ms,
            confidence=env.confidence, input=env.input, output=env.output,
            evidence=env.evidence, warnings=env.warnings, errors=env.errors,
        )

    def log_llm_usage(self, agent: str, model: str, tokens_in: int, tokens_out: int, cost_usd: float,
                      request_id: str = "", session_id: str = "") -> None:
        self.log("llm_usage", agent=agent, model=model, tokens_in=tokens_in, tokens_out=tokens_out,
                 cost_usd=cost_usd, request_id=request_id, session_id=session_id)

    def log_decision(self, request_id: str, session_id: str, action: str, size: float | None,
                     confidence: float, source: str) -> None:
        self.log("decision", request_id=request_id, session_id=session_id, action=action,
                 size=size, confidence=confidence, source=source)

    def log_deviation(self, request_id: str, session_id: str, probability: float, triggered: bool,
                      original_action: str, final_action: str) -> None:
        self.log("deviation", request_id=request_id, session_id=session_id, probability=probability,
                 triggered=triggered, original_action=original_action, final_action=final_action)

    def log_outcome(self, request_id: str, session_id: str, outcome: dict) -> None:
        self.log("outcome", request_id=request_id, session_id=session_id, outcome=outcome)

    def log_retry(self, request_id: str, agent: str, attempt: int, error: str) -> None:
        self.log("retry", request_id=request_id, agent=agent, attempt=attempt, error=error)

    def log_error(self, request_id: str, agent: str, code: str, message: str) -> None:
        self.log("error", request_id=request_id, agent=agent, code=code, message=message)

    def log_replay(self, request_id: str, hands: int, decisions: int, summary: dict) -> None:
        self.log("replay", request_id=request_id, hands=hands, decisions=decisions, summary=summary)


def configure_logging(level: str = "INFO", logs_dir: str = "logs") -> EventLogger:
    """Configure the standard logger + return a structured EventLogger."""
    os.makedirs(logs_dir, exist_ok=True)
    root = logging.getLogger("poker_ai")
    root.setLevel(level.upper())
    if not root.handlers:
        fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)
        fh = logging.FileHandler(os.path.join(logs_dir, "app.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return EventLogger(logs_dir)
