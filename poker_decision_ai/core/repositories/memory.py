"""In-memory repository implementations (tests, transient sessions)."""
from __future__ import annotations

import itertools
import threading
from typing import Any

from .interfaces import (
    DecisionRepository,
    HandRepository,
    PlayerRepository,
    PopulationRepository,
    RepositorySet,
    ResearchRepository,
    _iso_now,
)


class _Seq:
    def __init__(self) -> None:
        self._n = itertools.count(1)

    def next(self, prefix: str) -> str:
        return f"{prefix}-{next(self._n)}"


class InMemoryHandRepository(HandRepository):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = _Seq()
        self._items: dict[str, dict] = {}

    def save(self, hand: dict) -> str:
        with self._lock:
            # record_id（如 "hand:player"）优先作唯一键；普通手牌用 hand_id
            hid = str(hand.get("record_id") or hand.get("hand_id") or self._seq.next("hand"))
            rec = dict(hand)
            rec["hand_id"] = str(hand.get("hand_id") or hid)
            rec.setdefault("stored_at", _iso_now())
            self._items[hid] = rec
            return hid

    def get(self, hand_id: str) -> dict | None:
        return self._items.get(hand_id)

    def list(self, limit: int = 1000, offset: int = 0) -> list:
        items = list(self._items.values())[offset : offset + limit]
        return items

    def count(self) -> int:
        return len(self._items)


class InMemoryPlayerRepository(PlayerRepository):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}

    def upsert(self, player: dict) -> None:
        with self._lock:
            pid = str(player["player_id"])
            cur = self._items.get(pid, {})
            merged = dict(cur)
            merged.update(player)
            merged["player_id"] = pid
            merged["updated_at"] = _iso_now()
            self._items[pid] = merged

    def get(self, player_id: str) -> dict | None:
        return self._items.get(player_id)

    def all(self) -> list:
        return list(self._items.values())


class InMemoryPopulationRepository(PopulationRepository):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}

    def upsert_segment(self, segment_key: str, stats: dict) -> None:
        with self._lock:
            cur = self._items.get(segment_key, {})
            merged = dict(cur)
            merged.update(stats)
            merged["segment"] = segment_key
            merged["updated_at"] = _iso_now()
            self._items[segment_key] = merged

    def get_segment(self, segment_key: str) -> dict | None:
        return self._items.get(segment_key)

    def all_segments(self) -> list:
        return list(self._items.values())


class InMemoryDecisionRepository(DecisionRepository):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, dict] = {}

    def save(self, decision: dict) -> str:
        with self._lock:
            rid = str(decision["request_id"])
            self._items[rid] = dict(decision)
            self._items[rid]["stored_at"] = _iso_now()
            return rid

    def load_many(self, records) -> int:
        """P30-9-ID：**批量装载** ✓（冷启动提速 ✓）。

        语义：**保留记录自带的 `stored_at`** ✓（那才是它的入库时间 ✓），
        缺失时用**同一个**时间戳补齐 ✓。
        动机：原先逐条 `save()` ⇒ 每条都重算一次 `_iso_now()` ✗
        （实测 106,373 次 ≈ **4.7 秒**白花 ✓，且把原始入库时间冲掉 ✗）。
        """
        ts = _iso_now()
        n = 0
        with self._lock:
            for rec in records or ():
                rid = rec.get("request_id")
                if not rid:
                    continue
                item = dict(rec)
                item.setdefault("stored_at", ts)
                self._items[str(rid)] = item
                n += 1
        return n

    def get(self, request_id: str) -> dict | None:
        return self._items.get(request_id)

    def list(self, limit: int = 1000) -> list:
        return list(self._items.values())[-limit:]


class InMemoryResearchRepository(ResearchRepository):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = _Seq()
        self._items: dict[str, dict] = {}

    def add(self, kind: str, record: dict) -> str:
        with self._lock:
            rid = str(record.get("id") or self._seq.next(kind))
            rec = dict(record)
            rec["id"] = rid
            rec["kind"] = kind
            rec.setdefault("created_at", _iso_now())
            self._items[rid] = rec
            return rid

    def list(self, kind: str | None = None, limit: int = 500) -> list:
        items = list(self._items.values())
        if kind:
            items = [r for r in items if r.get("kind") == kind]
        return items[-limit:]

    def get(self, record_id: str) -> dict | None:
        return self._items.get(record_id)


def build_in_memory() -> RepositorySet:
    return RepositorySet(
        hands=InMemoryHandRepository(),
        players=InMemoryPlayerRepository(),
        populations=InMemoryPopulationRepository(),
        decisions=InMemoryDecisionRepository(),
        research=InMemoryResearchRepository(),
    )
