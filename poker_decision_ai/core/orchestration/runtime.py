"""RuntimeBuilder: assemble a full runtime from configuration.

This is the composition root used by CLI, scripts, replay and tests:
    config -> logging + metrics -> providers -> repositories
           -> registry(36 agents) -> caches(V2) -> runner -> runtime

Everything is dependency-injected; no module-level singletons.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..cache import CacheManager
from ..config import AppConfig, load_config
from ..logging import EventLogger, MetricsCollector, configure_logging
from ..providers import ProviderRegistry, build_providers
from ..repositories.interfaces import RepositorySet, build_repositories
from .context import AgentContext
from .registry import AgentRegistry, register_all_agents
from .runner import PipelineRunner


@dataclass
class Runtime:
    config: AppConfig
    providers: ProviderRegistry
    repos: RepositorySet
    registry: AgentRegistry
    runner: PipelineRunner
    logger: EventLogger
    metrics: MetricsCollector
    caches: CacheManager = None  # type: ignore[assignment]  # V2（dataclass 默认；build_runtime 必填）

    def new_context(self, request_id: str, session_id: str = "default",
                    *, allow_llm: bool = True, extra: dict | None = None) -> AgentContext:
        return AgentContext(
            request_id=request_id, session_id=session_id, config=self.config,
            providers=self.providers, repos=self.repos, logger=self.logger,
            metrics=self.metrics, allow_llm=allow_llm, registry=self.registry,
            caches=self.caches,
            extra=extra or {},
        )

    def agent(self, name: str):
        return self.registry.get(name)

    async def close(self) -> None:
        await self.providers.close_all()


def build_runtime(config: AppConfig | None = None, *, log_level: str = "INFO",
                  reset_metrics: bool = True, caches: CacheManager | None = None) -> Runtime:
    """Build the full runtime. config=None loads the default YAML files."""
    if config is None:
        config = load_config()
    paths = config.paths()
    logger = configure_logging(log_level, paths["logs_dir"])
    metrics = MetricsCollector()
    if reset_metrics:
        metrics.reset()
    providers = build_providers(config)
    repos = build_repositories(config)
    registry = register_all_agents()
    runner = PipelineRunner(registry, config, logger, metrics)
    if caches is None:
        caches = CacheManager()
    # P30-9-BG：安装『范围化联合胜率』挂钩（默认启用；JOINT_EQUITY_MODE=off 可回退）
    try:
        from ...joint_equity import install_joint_hook
        install_joint_hook()
    except Exception:  # noqa: BLE001
        pass
    # P30-9-DG：**启动预热**——spot 表（HU/6-max）与翻前图表在首次点击前加载完毕，
    # 使实时 fast 路径（300ms 紧超时）内的库查表为热路径；否则 solver 常因
    # 冷启动超时而降级（实测：judge 收到空 solver 源 ⇒ 退回启发式 ⇒ 过度全下）。
    try:
        from ..providers.spot_table import get_spot_table
        get_spot_table()
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..providers.spot_table_6max import get_spot_table_6max
        get_spot_table_6max()
        get_spot_table_6max().count()
    except Exception:  # noqa: BLE001
        pass
    try:
        from ..providers.preflop_chart import PreflopChartProvider
        _chart = str(config.solver_config().get("preflop_chart") or "")
        if _chart:
            PreflopChartProvider(_chart).chart_mode()
    except Exception:  # noqa: BLE001
        pass

    return Runtime(config=config, providers=providers, repos=repos, registry=registry,
                   runner=runner, logger=logger, metrics=metrics, caches=caches)
