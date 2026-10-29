"""SolverCache（V2 Phase 4）：缓存常见局面 → GTO 策略，避免重复求解。

键 = 归一化局面（街/公共牌/手牌/底池与有效筹码按 bb 归一/下注尺寸/待跟注），
值 = SolverResult.to_dict()。TTL 默认 24h；版本由外部模型版本决定。
"""
from __future__ import annotations

from typing import Any

from .base import TTLDictCache, stable_key

DEFAULT_TTL_S = 24 * 3600.0


def solver_cache_key(query: dict) -> str:
    """归一化查询键：金额全部按 bb 归一，避免同一局面因金额缩放重复求解。"""
    pot = float(query.get("pot") or 0.0)
    eff = float(query.get("effective_stack") or 0.0)
    to_call = float(query.get("to_call") or 0.0)
    # 用 pot/effective 的相对量而非绝对量（比值在小数点后 3 位稳定）
    rel_eff = round(eff / pot, 3) if pot > 0 else round(eff, 1)
    rel_call = round(to_call / pot, 3) if pot > 0 else round(to_call, 1)
    return stable_key(
        query.get("street", ""),
        query.get("board") or [],
        query.get("hole_cards") or [],
        rel_eff,
        rel_call,
        query.get("bet_sizings") or [],
    )


class SolverCache:
    """Solver 结果缓存（版本 + TTL + 失效，线程安全）。"""

    def __init__(self, ttl_s: float = DEFAULT_TTL_S, version: str = "solver_v1",
                 max_entries_hint: int = 20000) -> None:
        self._cache = TTLDictCache(name="solver_cache", default_ttl_s=ttl_s,
                                   default_version=version)
        self.version = version

    @property
    def default_version(self) -> str:
        return self._cache.default_version

    @default_version.setter
    def default_version(self, value: str) -> None:
        self._cache.default_version = value
        self.version = value

    def get(self, query: dict) -> dict | None:
        return self._cache.get(solver_cache_key(query))

    def put(self, query: dict, result: dict) -> None:
        self._cache.put(solver_cache_key(query), result)

    async def get_or_solve(self, query: dict, provider) -> tuple[dict, bool]:
        """缓存命中返回 (result_dict, True)；否则调用 provider 求解并缓存。

        provider 必须实现 async solve(query) -> SolverResult。
        """
        cached = self.get(query)
        if cached is not None:
            return cached, True
        result = await provider.solve(query)
        data = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        self.put(query, data)
        return data, False

    def invalidate(self, prefix: str = "") -> None:
        self._cache.invalidate(prefix)

    def snapshot(self) -> dict:
        return self._cache.snapshot()
