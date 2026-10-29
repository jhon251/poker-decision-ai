"""Repository interfaces - the ONLY data access path for agents.

Agents never touch a database or files directly; they call these interfaces.
V1 ships two implementations (see memory.py / json_file.py):
  - InMemoryRepositories : dict-based, for tests & transient sessions
  - JsonFileRepositories : append-only JSONL + index, persistent, no DB needed

A future SQL repository (SQLite/MySQL/PostgreSQL) implements the same ABCs,
so no agent ever changes.  This is the reserved Data Storage seam - the
Data Storage Agent itself is intentionally NOT part of V1 (spec §二).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


class HandRepository(ABC):
    """Historical hands (replay corpus + outcome store)."""

    @abstractmethod
    def save(self, hand: dict) -> str:
        """Save a hand record; returns its hand_id."""

    @abstractmethod
    def get(self, hand_id: str) -> dict | None: ...

    @abstractmethod
    def list(self, limit: int = 1000, offset: int = 0) -> list: ...

    @abstractmethod
    def count(self) -> int: ...


class PlayerRepository(ABC):
    """Individual player profiles (three-layer modelling source data)."""

    @abstractmethod
    def upsert(self, player: dict) -> None: ...

    @abstractmethod
    def get(self, player_id: str) -> dict | None: ...

    @abstractmethod
    def all(self) -> list: ...


class PopulationRepository(ABC):
    """Population statistics keyed by segment (stakes/game/position/stack...)."""

    @abstractmethod
    def upsert_segment(self, segment_key: str, stats: dict) -> None: ...

    @abstractmethod
    def get_segment(self, segment_key: str) -> dict | None: ...

    @abstractmethod
    def all_segments(self) -> list: ...


class DecisionRepository(ABC):
    """Every decision the framework produces (base + deviated), for evaluation."""

    @abstractmethod
    def save(self, decision: dict) -> str: ...

    @abstractmethod
    def get(self, request_id: str) -> dict | None: ...

    @abstractmethod
    def list(self, limit: int = 1000) -> list: ...


class ResearchRepository(ABC):
    """Long-term research knowledge base: notes, hypotheses, experiments, findings."""

    @abstractmethod
    def add(self, kind: str, record: dict) -> str: ...

    @abstractmethod
    def list(self, kind: str | None = None, limit: int = 500) -> list: ...

    @abstractmethod
    def get(self, record_id: str) -> dict | None: ...


@dataclass
class RepositorySet:
    """Bundle of all repositories used by one runtime."""

    hands: HandRepository
    players: PlayerRepository
    populations: PopulationRepository
    decisions: DecisionRepository
    research: ResearchRepository


def _iso_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def build_repositories(config) -> RepositorySet:
    """Factory: build the repository set from configuration."""
    storage = config.storage_config()
    backend = (storage.get("backend") or "json").lower()
    if backend == "memory":
        from .memory import build_in_memory

        return build_in_memory()
    if backend == "json":
        from .json_file import JsonFileRepositories

        return JsonFileRepositories(
            data_dir=storage.get("dir") or config.paths()["data_dir"] + "/repository",
            seed_data=storage.get("seed_data"),
        )
    raise ValueError(f"unknown storage.backend '{backend}' (expected: memory|json)")
