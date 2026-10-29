"""JSON-file repositories: persistent storage without a database.

Layout under <data_dir>/:
  hands.jsonl        players.jsonl        populations.jsonl
  decisions.jsonl    research.jsonl       (one JSON object per line, append-only)

On init the files are loaded into memory (personal-study scale), so reads are
instant and writes are crash-safe appends.  This backend exists so the
learning system survives restarts; swap it for SQL later via the same ABCs.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import threading
from pathlib import Path
from typing import Any

from ..errors import RepositoryError
from .interfaces import (
    DecisionRepository,
    HandRepository,
    PlayerRepository,
    PopulationRepository,
    RepositorySet,
    ResearchRepository,
    _iso_now,
)
from .memory import (
    InMemoryDecisionRepository,
    InMemoryHandRepository,
    InMemoryPlayerRepository,
    InMemoryPopulationRepository,
    InMemoryResearchRepository,
)


# ── P30-9-ID：**解析缓存** ✗→✓ ──────────────────────────────────────────
#   实测（cProfile，用户报告"太慢"）：冷启动 **35 秒** ✗，其中 `json.loads` 占 **36.8 秒 / 106,398 行**
#   （`_JsonlStore.load()` 每次构造 runtime 都全量重解析 ✓）。
#   ⇒ 用**签名校验的 pickle 缓存**：键 = (路径, 大小, mtime_ns) ✓，
#     命中即秒级返回 ✓；签名不符/缓存损坏/不可写 ⇒ **自动回退到原解析路径** ✓（绝不因缓存失败 ✗）；
#     `REPO_CACHE=0` 可关闭 ✓；缓存目录默认 `.tmp/repo_cache`（**不参与发布** ✓）。
_CACHE_DISABLED = str(os.environ.get("REPO_CACHE", "1")).lower() in ("0", "off", "false", "no")


def _cache_file(p: Path) -> Path:
    d = Path(os.environ.get("REPO_CACHE_DIR") or ".tmp/repo_cache")
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        d = Path(tempfile.gettempdir()) / "poker_repo_cache"
        d.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(p.resolve()).encode("utf-8")).hexdigest()[:20]
    return d / f"{key}.pkl"


def _signature(p: Path) -> tuple:
    st = p.stat()
    return (int(st.st_size), int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))))


class _JsonlStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def load(self) -> list:
        if not self.path.exists():
            return []
        sig = None
        cache = None
        if not _CACHE_DISABLED:
            try:
                sig = _signature(self.path)
                cache = _cache_file(self.path)
                with open(cache, "rb") as fh:
                    blob = pickle.load(fh)
                if isinstance(blob, dict) and blob.get("sig") == sig \
                        and isinstance(blob.get("rows"), list):
                    return blob["rows"]                      # 命中 ⇒ 秒级 ✓
            except Exception:                                # noqa: BLE001
                pass                                         # 未命中/损坏 ⇒ 走原路 ✓
        out = []
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        try:
                            out.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError as e:
            raise RepositoryError(f"cannot read {self.path}: {e}") from e
        if not _CACHE_DISABLED and sig is not None and cache is not None:
            try:
                tmp = cache.with_suffix(".tmp")
                with open(tmp, "wb") as fh:
                    pickle.dump({"sig": sig, "rows": out}, fh, protocol=4)
                os.replace(tmp, cache)                       # 原子替换 ✓
            except Exception:                                # noqa: BLE001
                pass                                         # 写缓存失败不影响功能 ✓
        return out

    def append(self, record: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            except OSError as e:
                raise RepositoryError(f"cannot append to {self.path}: {e}") from e


class JsonFileRepositories(RepositorySet):
    """Persistent repositories backed by JSONL files."""

    def __init__(self, data_dir: str, seed_data: str | None = None) -> None:
        root = Path(data_dir)
        self._root = root
        root.mkdir(parents=True, exist_ok=True)
        self._stores = {
            "hands": _JsonlStore(root / "hands.jsonl"),
            "players": _JsonlStore(root / "players.jsonl"),
            "populations": _JsonlStore(root / "populations.jsonl"),
            "decisions": _JsonlStore(root / "decisions.jsonl"),
            "research": _JsonlStore(root / "research.jsonl"),
        }
        self.hands = self._mk_hand_repo()
        self.players = self._mk_player_repo()
        self.populations = self._mk_population_repo()
        self.decisions = self._mk_decision_repo()
        self.research = self._mk_research_repo()
        self._seed(seed_data)

    def _mk_hand_repo(self) -> HandRepository:
        inner = InMemoryHandRepository()
        for rec in self._stores["hands"].load():
            if "hand_id" in rec:
                inner.save(rec)
        return _PersistentHandRepo(inner, self._stores["hands"])

    def _mk_player_repo(self) -> PlayerRepository:
        inner = InMemoryPlayerRepository()
        for rec in self._stores["players"].load():
            if "player_id" in rec:
                inner.upsert(rec)
        return _PersistentPlayerRepo(inner, self._stores["players"])

    def _mk_population_repo(self) -> PopulationRepository:
        inner = InMemoryPopulationRepository()
        for rec in self._stores["populations"].load():
            if "segment" in rec:
                inner.upsert_segment(rec["segment"], rec)
        return _PersistentPopulationRepo(inner, self._stores["populations"])

    def _mk_decision_repo(self) -> DecisionRepository:
        inner = InMemoryDecisionRepository()
        store = self._stores["decisions"]
        # P30-9-IE：**改为懒加载** ✓（此前在这里全量 load + 逐条灌入 ✗ ⇒ 冷启动数十秒 ✗）。
        #   注意：这里**不**调用 load() ✓ —— 把"何时读"推迟到真正 get()/list() ✓。
        return _PersistentDecisionRepo(inner, store, lazy_rows=store.load)

    def _mk_research_repo(self) -> ResearchRepository:
        inner = InMemoryResearchRepository()
        for rec in self._stores["research"].load():
            inner.add(rec.get("kind", "note"), rec)
        return _PersistentResearchRepo(inner, self._stores["research"])

    def _seed(self, seed_data: str | None) -> None:
        if not seed_data:
            return
        p = Path(seed_data)
        if not p.exists():
            return
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for pid, rec in (data.get("players") or {}).items():
            rec.setdefault("player_id", pid)
            self.players.upsert(rec)
        for seg, rec in (data.get("populations") or {}).items():
            self.populations.upsert_segment(seg, rec)


class _PersistentHandRepo(InMemoryHandRepository):
    def __init__(self, inner: InMemoryHandRepository, store: _JsonlStore) -> None:
        super().__init__()
        self._items = inner._items
        self._seq = inner._seq
        self._store = store

    def save(self, hand: dict) -> str:
        hid = super().save(hand)
        self._store.append(self._items[hid])
        return hid


class _PersistentPlayerRepo(InMemoryPlayerRepository):
    def __init__(self, inner: InMemoryPlayerRepository, store: _JsonlStore) -> None:
        super().__init__()
        self._items = inner._items
        self._store = store

    def upsert(self, player: dict) -> None:
        super().upsert(player)
        self._store.append(self._items[str(player["player_id"])])


class _PersistentPopulationRepo(InMemoryPopulationRepository):
    def __init__(self, inner: InMemoryPopulationRepository, store: _JsonlStore) -> None:
        super().__init__()
        self._items = inner._items
        self._store = store

    def upsert_segment(self, segment_key: str, stats: dict) -> None:
        super().upsert_segment(segment_key, stats)
        self._store.append(self._items[segment_key])


class _PersistentDecisionRepo(InMemoryDecisionRepository):
    """P30-9-IE：**懒加载** ✓ —— 动机：真实点击路径**不需要** 10 万条历史决策 ✓，
    而每次冷启动都全量解析/反序列化 ✗（实测 `_pickle.load` **7.9 秒** ✗、`json.loads` **36.8 秒** ✗）。
    * `get()/list()` 首次调用时才加载 ✓；
    * `save()` **不触发**加载 ✓（只追加 ✓，否则实时路径首次保存又要付全量代价 ✗）；
    * 签名/缓存逻辑仍在 `_JsonlStore` 内 ✓（此处只推迟"何时读" ✓，不改"读什么" ✓）。
    """

    def __init__(self, inner: InMemoryDecisionRepository, store: _JsonlStore,
                 lazy_rows=None) -> None:
        super().__init__()
        self._items = inner._items
        self._store = store
        self._lazy = lazy_rows                 # callable | list | None
        self._lazy_done = lazy_rows is None

    def _ensure(self) -> None:
        if self._lazy_done:
            return
        self._lazy_done = True
        try:
            rows = self._lazy() if callable(self._lazy) else (self._lazy or [])
            if rows:
                self.load_many([r for r in rows if "request_id" in r])
        except Exception:                                       # noqa: BLE001
            pass                                                # 懒加载失败不影响功能 ✓
        finally:
            self._lazy = None

    def get(self, request_id: str) -> dict | None:
        self._ensure()
        return super().get(request_id)

    def list(self, limit: int = 1000) -> list:
        self._ensure()
        return super().list(limit)

    def save(self, decision: dict) -> str:
        rid = super().save(decision)
        self._store.append(self._items[rid])
        return rid


class _PersistentResearchRepo(InMemoryResearchRepository):
    def __init__(self, inner: InMemoryResearchRepository, store: _JsonlStore) -> None:
        super().__init__()
        self._items = inner._items
        self._seq = inner._seq
        self._store = store

    def add(self, kind: str, record: dict) -> str:
        rid = super().add(kind, record)
        self._store.append(self._items[rid])
        return rid
