"""Data access layer: agents -> Repository interfaces -> storage backend."""
from .interfaces import (
    DecisionRepository,
    HandRepository,
    PlayerRepository,
    PopulationRepository,
    RepositorySet,
    ResearchRepository,
    build_repositories,
)
from .memory import build_in_memory
from .json_file import JsonFileRepositories

__all__ = [
    "DecisionRepository", "HandRepository", "PlayerRepository",
    "PopulationRepository", "RepositorySet", "ResearchRepository",
    "build_repositories", "build_in_memory", "JsonFileRepositories",
]
