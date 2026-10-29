"""Cache 基础设施（V2 Phase 4/5）。

所有 Cache 共用的原语：
  - 版本号（version）：模型/数据版本变化时失效
  - 更新时间（updated_at）
  - TTL / 失效策略（expires_at + invalidate）
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class CacheEntry:
    value: Any
    version: str = "v1"
    created_at: float = 0.0
    expires_at: float = 0.0

    def is_expired(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self.expires_at > 0 and now > self.expires_at


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    puts: int = 0
    invalidations: int = 0
    expirations: int = 0

    def snapshot(self) -> dict:
        return {
            "hits": self.hits, "misses": self.misses, "puts": self.puts,
            "invalidations": self.invalidations, "expirations": self.expirations,
            "hit_rate": round(self.hits / max(1, self.hits + self.misses), 4),
        }


class TTLDictCache:
    """线程安全的带 TTL / 版本 / 失效的字典缓存。

    - put(key, value, ttl_s, version)
    - get(key) -> (value, version) | None（过期自动删除并计 expiration）
    - invalidate(prefix) / clear()
    """

    def __init__(self, name: str = "cache", default_ttl_s: float = 300.0,
                 default_version: str = "v1") -> None:
        self.name = name
        self.default_ttl_s = default_ttl_s
        self.default_version = default_version
        self._lock = threading.Lock()
        self._entries: dict[str, CacheEntry] = {}
        self.stats = CacheStats()

    # ---------------------------------------------------------------- 基本操作
    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            if entry.is_expired(now):
                del self._entries[key]
                self.stats.expirations += 1
                self.stats.misses += 1
                return None
            self.stats.hits += 1
            return entry.value

    def put(self, key: str, value: Any, ttl_s: float | None = None,
            version: str | None = None) -> None:
        now = time.time()
        with self._lock:
            self._entries[key] = CacheEntry(
                value=value,
                version=version or self.default_version,
                created_at=now,
                expires_at=now + (self.default_ttl_s if ttl_s is None else ttl_s),
            )
            self.stats.puts += 1

    def invalidate(self, prefix: str = "") -> None:
        with self._lock:
            if not prefix:
                self._entries.clear()
                self.stats.invalidations += 1
                return
            for key in [k for k in self._entries if k.startswith(prefix)]:
                del self._entries[key]
            self.stats.invalidations += 1

    def clear(self) -> None:
        self.invalidate("")

    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._entries)

    def snapshot(self) -> dict:
        return {"name": self.name, "size": self.size(), "stats": self.stats.snapshot()}


def stable_key(*parts: Any) -> str:
    """稳定缓存键：把各部分归一化为可排序字符串再哈希。"""
    norm: list[str] = []
    for p in parts:
        if isinstance(p, (list, tuple, set)):
            norm.append("[" + ",".join(str(x) for x in sorted(p, key=str)) + "]")
        elif isinstance(p, dict):
            norm.append("{" + ",".join(f"{k}:{v}" for k, v in sorted(p.items())) + "}")
        else:
            norm.append(str(p))
    return "|".join(norm)
