"""专用模型/计算缓存（V2 Phase 5）。

每个缓存都继承 TTLDictCache：版本号 + TTL + 失效 + 命中率统计。
版本注册表（CacheManager）保证模型升级时按名称整体失效对应缓存。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .base import TTLDictCache, stable_key

DEFAULT_TTLS = {
    "player": 900.0,        # 15 分钟（学习更新会主动失效）
    "population": 6 * 3600.0,
    "equity": 900.0,
    "range": 900.0,
    "solver": 24 * 3600.0,
    "feature": 300.0,
}


def digest(obj: Any, ndigits: int = 6) -> str:
    """对象的稳定摘要（范围分布等大型结构用）。"""
    def _norm(x: Any):
        if isinstance(x, dict):
            return {str(k): _norm(v) for k, v in sorted(x.items())}
        if isinstance(x, (list, tuple, set)):
            return [_norm(v) for v in sorted(x, key=repr)]
        if isinstance(x, float):
            return round(x, ndigits)
        return x

    return hashlib.sha1(json.dumps(_norm(obj), sort_keys=True).encode()).hexdigest()[:16]


class PlayerModelCache(TTLDictCache):
    """player_id -> 玩家画像 dict。Learning Agent 更新后应 invalidate(pid)。"""

    def __init__(self, version: str = "player_v1", ttl_s: float | None = None):
        super().__init__(name="player_model_cache",
                         default_ttl_s=ttl_s or DEFAULT_TTLS["player"],
                         default_version=version)


class PopulationCache(TTLDictCache):
    """segment_key -> 群体统计 dict。"""

    def __init__(self, version: str = "population_v1", ttl_s: float | None = None):
        super().__init__(name="population_cache",
                         default_ttl_s=ttl_s or DEFAULT_TTLS["population"],
                         default_version=version)


class EquityCache(TTLDictCache):
    """(hero, board, villain_range_digest, iterations, seed) -> EquityResult dict。"""

    @staticmethod
    def key(hero: list, board: list, villain_range_signature: str,
            iterations: int, seed: int) -> str:
        return stable_key(hero, board, villain_range_signature, iterations, seed)

    def __init__(self, version: str = "equity_v1", ttl_s: float | None = None):
        super().__init__(name="equity_cache",
                         default_ttl_s=ttl_s or DEFAULT_TTLS["equity"],
                         default_version=version)


class RangeCache(TTLDictCache):
    """(position, route, player_id, model_version) -> range dict。"""

    @staticmethod
    def key(position: str, route: str, player_id: str, model_version: str) -> str:
        return stable_key(position, route, player_id, model_version)

    def __init__(self, version: str = "range_v1", ttl_s: float | None = None):
        super().__init__(name="range_cache",
                         default_ttl_s=ttl_s or DEFAULT_TTLS["range"],
                         default_version=version)


class FeatureCache(TTLDictCache):
    """通用特征缓存：stable_key(*parts) -> 特征 dict（board/赔率等轻量特征）。"""

    def __init__(self, version: str = "feature_v1", ttl_s: float | None = None):
        super().__init__(name="feature_cache",
                         default_ttl_s=ttl_s or DEFAULT_TTLS["feature"],
                         default_version=version)


class CacheManager:
    """全部缓存的持有者 + 模型版本注册表（V2 §十五）。

    - snapshot(): 各缓存规模/命中率
    - bump_version(name, new_version): 模型版本升级 → 该缓存整体失效
    - invalidate(name, prefix): 定向失效（如 Learning 更新某玩家后失效其画像）
    """

    def __init__(self) -> None:
        self.player = PlayerModelCache()
        self.population = PopulationCache()
        self.equity = EquityCache()
        self.range = RangeCache()
        self.feature = FeatureCache()
        from .solver_cache import SolverCache

        self.solver = SolverCache()
        self.model_versions = {
            "population": "population_v1",
            "player": "player_v1",
            "range": "range_v1",
            "bluff": "bluff_v1",
            "exploit": "exploit_v1",
            "solver": "solver_v1",
            "equity": "equity_v1",
            "feature": "feature_v1",
        }

    def version(self, name: str) -> str:
        return self.model_versions.get(name, "v1")

    def all(self) -> dict:
        return {
            "player": self.player, "population": self.population,
            "equity": self.equity, "range": self.range,
            "feature": self.feature, "solver": self.solver,
        }

    def bump_version(self, name: str, new_version: str) -> None:
        """模型版本升级：登记新版本并清空对应缓存。"""
        self.model_versions[name] = new_version
        cache = self.all().get(name)
        if cache is not None:
            cache.invalidate()
            cache.default_version = new_version

    def invalidate(self, name: str, prefix: str = "") -> None:
        cache = self.all().get(name)
        if cache is not None:
            cache.invalidate(prefix)

    def invalidate_all(self) -> None:
        for cache in self.all().values():
            cache.invalidate()

    def snapshot(self) -> dict:
        out = {"model_versions": dict(self.model_versions)}
        for name, cache in self.all().items():
            snap = cache.snapshot()
            snap["version"] = cache.default_version
            out[name] = snap
        return out
