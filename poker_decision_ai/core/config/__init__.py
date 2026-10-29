"""Configuration loading: YAML files + ${ENV:NAME} interpolation.

All configuration lives in `configs/` (config.yaml, stakes.yaml, ranges.yaml,
pipeline.yaml).  Nothing is hard-coded in agents: stakes, deviation
probability, equity iterations, provider endpoints etc. are all configurable.
Secrets come from environment variables via python-dotenv.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..errors import ConfigError

_ENV_PATTERN = re.compile(r"\$\{ENV:([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate(value: Any) -> Any:
    if isinstance(value, str):
        def _sub(m: re.Match) -> str:
            env = os.environ.get(m.group(1))
            if env is None or env == "":
                default = m.group(2)
                if default is None:
                    raise ConfigError(f"environment variable '{m.group(1)}' referenced in config is not set")
                return default
            return env
        return _ENV_PATTERN.sub(_sub, value)
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class NestedDict:
    """Read-only nested dict with dotted-path access: cfg.get('equity.iterations')."""

    def __init__(self, data: dict):
        self._data = data

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NestedDict":
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"config file not found: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls(_interpolate(data))

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def require(self, path: str) -> Any:
        v = self.get(path)
        if v is None:
            raise ConfigError(f"missing required config key: {path}")
        return v

    def dict(self) -> dict:
        return self._data

    def merged(self, other: "NestedDict") -> "NestedDict":
        return NestedDict(_deep_merge(self._data, other._data))


@dataclass
class AppConfig:
    """Typed facade over the merged YAML configuration."""

    root: NestedDict = field(default_factory=lambda: NestedDict({}))
    stakes: NestedDict = field(default_factory=lambda: NestedDict({}))
    ranges: NestedDict = field(default_factory=lambda: NestedDict({}))
    pipeline: NestedDict = field(default_factory=lambda: NestedDict({}))

    # ---- convenience accessors (all overridable in yaml) ----
    def llm_enabled(self) -> bool:
        return bool(self.root.get("llm.enabled", True))

    def llm_config(self) -> dict:
        return dict(self.root.get("llm", {}) or {})

    def vision_config(self) -> dict:
        return dict(self.root.get("vision", {}) or {})

    def solver_config(self) -> dict:
        return dict(self.root.get("solver", {}) or {})

    def storage_config(self) -> dict:
        return dict(self.root.get("storage", {}) or {})

    def agent_defaults(self) -> dict:
        return dict(self.root.get("agents.defaults", {}) or {})

    def agent_setting(self, agent: str, key: str, default: Any = None) -> Any:
        return self.root.get(f"agents.{agent}.{key}", self.root.get(f"agents.defaults.{key}", default))

    def equity_config(self) -> dict:
        return dict(self.root.get("equity", {}) or {})

    def icm_config(self) -> dict:
        return dict(self.root.get("icm", {}) or {})

    def deviation_config(self) -> dict:
        return dict(self.root.get("deviation", {}) or {})

    def learning_config(self) -> dict:
        return dict(self.root.get("learning", {}) or {})

    def realtime_config(self) -> dict:
        """V2 快路径配置（equity_iterations/agent_timeout_ms/deadline_ms/allow_llm）。"""
        return dict(self.root.get("realtime", {}) or {})

    def frequency_prior_config(self) -> dict:
        """MC1-R5：对手频率先验（确定性；权重/阈值全部声明于此）。"""
        return dict(self.root.get("frequency_prior", {}) or {})

    def strategy_core_config(self) -> dict:
        """SOLVER-2：求解器锚定 + 有界剥削偏离（基座反转预算全部声明于此）。

        M3（R5）：若声明 pool_budget_profile 且命中 pool_budget_profiles，
        则 `dev_max_ev_loss_bb` 解析为该档值（剥削预算按池强度选档；
        解析痕迹记入 `pool_budget_resolved`，未声明/未命中 → 原值透传，
        生产默认 strong 档与 0.05 基线一致 → 零回退）。
        """
        cfg = dict(self.root.get("strategy_core", {}) or {})
        profiles = cfg.get("pool_budget_profiles") or {}
        active = cfg.get("pool_budget_profile")
        if active and isinstance(profiles, dict) and active in profiles:
            try:
                budget = float(profiles[active])
            except (TypeError, ValueError):
                return cfg
            cfg["pool_budget_resolved"] = {"profile": active,
                                           "budget_bb": budget}
            if cfg.get("dev_max_ev_loss_bb") is not None:
                cfg["dev_max_ev_loss_bb"] = budget
        return cfg

    def deep_config(self) -> dict:
        """V2 Deep 后台路径配置。"""
        return dict(self.root.get("deep", {}) or {})

    def paths(self) -> dict:
        return {
            "logs_dir": str(self.root.get("app.logs_dir", "logs")),
            "data_dir": str(self.root.get("app.data_dir", "data")),
            "replay_dir": str(self.root.get("replay.output_dir", "data/replay")),
        }

    def stakes_table(self) -> dict:
        return dict(self.stakes.get("stakes", {}) or {})

    def stake_info(self, name: str) -> dict:
        table = self.stakes_table()
        if name not in table:
            raise ConfigError(f"unknown stake level '{name}'. Configured: {sorted(table)}")
        return dict(table[name])

    def default_ranges(self) -> dict:
        return dict(self.ranges.get("ranges", {}) or {})


def load_config(
    config_path: str | None = None,
    stakes_path: str | None = None,
    ranges_path: str | None = None,
    pipeline_path: str | None = None,
) -> AppConfig:
    """Load and merge all YAML configs. Paths default to the configs/ directory."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # pragma: no cover - dotenv is optional sugar
        pass

    default = os.environ.get("POKER_AI_CONFIG", "configs/config.yaml")
    cfg = NestedDict.from_yaml(config_path or default)
    stakes = NestedDict.from_yaml(stakes_path or cfg.get("configs.stakes", "configs/stakes.yaml"))
    ranges = NestedDict.from_yaml(ranges_path or cfg.get("configs.ranges", "configs/ranges.yaml"))
    pipeline = NestedDict.from_yaml(pipeline_path or cfg.get("configs.pipeline", "configs/pipeline.yaml"))
    return AppConfig(root=cfg, stakes=stakes, ranges=ranges, pipeline=pipeline)
