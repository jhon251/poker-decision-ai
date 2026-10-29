"""延迟统计（V2 Phase 7/§二十六）：P50/P90/P95/P99 + 分量延迟。"""
from __future__ import annotations

import statistics
import threading
from collections import defaultdict


class LatencyTracker:
    """线程安全的延迟样本收集与百分位统计。

    每个路径（realtime/deep/replay）一个 tracker；record() 记录总延迟，
    record_part() 记录分量（state/vision/equity/.../judge）。
    """

    def __init__(self, name: str, max_samples: int = 5000) -> None:
        self.name = name
        self.max_samples = max_samples
        self._lock = threading.Lock()
        self._total: list[float] = []
        self._parts: dict[str, list[float]] = defaultdict(list)

    def record(self, total_ms: float) -> None:
        with self._lock:
            self._total.append(float(total_ms))
            if len(self._total) > self.max_samples:
                self._total.pop(0)

    def record_part(self, part: str, ms: float) -> None:
        with self._lock:
            self._parts[part].append(float(ms))
            if len(self._parts[part]) > self.max_samples:
                self._parts[part].pop(0)

    def _percentiles(self, values: list[float]) -> dict:
        if not values:
            return {"p50": None, "p90": None, "p95": None, "p99": None,
                    "min": None, "max": None, "avg": None, "n": 0}
        s = sorted(values)

        def pct(p: float) -> float:
            import math

            idx = max(0, math.ceil(p / 100.0 * len(s)) - 1)   # 最近秩法
            return round(s[idx], 2)

        return {"p50": pct(50), "p90": pct(90), "p95": pct(95), "p99": pct(99),
                "min": round(min(s), 2), "max": round(max(s), 2),
                "avg": round(statistics.fmean(s), 2), "n": len(s)}

    def snapshot(self) -> dict:
        with self._lock:
            total = self._percentiles(list(self._total))
            parts = {k: self._percentiles(list(v)) for k, v in sorted(self._parts.items())}
        return {"name": self.name, "total": total, "parts": parts}

    def reset(self) -> None:
        with self._lock:
            self._total.clear()
            self._parts.clear()
