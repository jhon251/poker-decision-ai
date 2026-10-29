"""Framework-wide exception hierarchy.

Every layer raises exceptions derived from :class:`FrameworkError` so the
Manager / top-level runner can distinguish framework failures from bugs and
apply the configured degradation policy.
"""
from __future__ import annotations


class FrameworkError(Exception):
    """Base class for all framework errors."""

    code = "FRAMEWORK_ERROR"
    retryable = False

    def __init__(self, message: str = "", *, detail: dict | None = None):
        super().__init__(message)
        self.detail = detail or {}


class ConfigError(FrameworkError):
    """Invalid or missing configuration."""

    code = "CONFIG_ERROR"


class SchemaError(FrameworkError):
    """A payload does not satisfy a JSON Schema / dataclass contract."""

    code = "SCHEMA_ERROR"


class AgentValidationError(SchemaError):
    """Agent input failed input-schema validation."""

    code = "AGENT_INPUT_INVALID"


class AgentOutputError(SchemaError):
    """Agent produced output that fails its output schema."""

    code = "AGENT_OUTPUT_INVALID"


class AgentExecutionError(FrameworkError):
    """Agent raised while processing (after input validation)."""

    code = "AGENT_EXECUTION_ERROR"
    retryable = True


class AgentTimeoutError(AgentExecutionError):
    """Agent exceeded its configured timeout."""

    code = "AGENT_TIMEOUT"
    retryable = True


class OrchestrationError(FrameworkError):
    """Plan / pipeline construction or execution failure."""

    code = "ORCHESTRATION_ERROR"


class DegradedExecutionError(OrchestrationError):
    """Pipeline finished but some agents degraded / were skipped."""

    code = "DEGRADED_EXECUTION"


class ProviderError(FrameworkError):
    """A provider call failed at runtime (network / API / parsing)."""

    code = "PROVIDER_ERROR"
    retryable = True


class ProviderNotConfiguredError(ProviderError):
    """A real provider was selected but is not configured (key / path missing).

    This is *not* a runtime failure: it signals that the adapter exists but
    the user has not wired it up yet. Callers should degrade to mock/deterministic.
    """

    code = "PROVIDER_NOT_CONFIGURED"
    retryable = False


class RepositoryError(FrameworkError):
    """Data access layer failure (storage unavailable / corrupted)."""

    code = "REPOSITORY_ERROR"
    retryable = True
