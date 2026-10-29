# -*- coding: utf-8 -*-
"""P30-9-HT：**多人（3-way+）桶表** —— 管道就位（**默认关闭** ✓，不混入生产库 ✓）。

背景（第 8 轮决策 B）：多人求解器**暂不采购**（可度量价值 ≈2% ✗，3-way 覆盖不了 4 人 ✓）。
但"管道就位"可以**免费**完成 ✓：只要 schema/键/转换器齐备，
将来买到数据源（Simple 3-way / Monker 等）就能**直接导入** ✓。

设计（与 6-max 表保持一致的分层，但**独立目录 + 独立开关** ✓）：
  * 目录：`data/solver/spots_multiway/`（**不参与 dist 打包**，避免与生产库混淆 ✓）；
  * 记录 schema：`{spot_id, street, board, pot_bb, effective_stack_bb, n_active,
                  positions: {oop, ip, ...}, actor_position, node, actions, solved,
                  source, source_note, synthetic?}` ✓；
  * 查询键：`(street, board, pot 档, node, actor, n_active)` ✓ —— **必须含 `n_active`** ✗
    （多人局与两人局的策略不可互换 ✓）；
  * 默认 `MULTIWAY_TABLE=0` ⇒ 永不返回结果（诚实 miss ✓）；
   显式 `MULTIWAY_TABLE=1` 才启用（供将来接线与测试 ✓）。
"""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any

DEFAULT_DIR = "data/solver/spots_multiway"
_CACHE: dict = {}


def _enabled() -> bool:
    return str(os.environ.get("MULTIWAY_TABLE", "0")).lower() not in (
        "0", "", "false", "no")


def _pot_class(pot_bb: float) -> str:
    return "small" if pot_bb < 8 else ("mid" if pot_bb < 15 else "big")


class MultiwaySpotTable:
    """多人桶表（惰性加载；**默认关闭** ⇒ `lookup()` 恒为 None ✓）。"""

    def __init__(self, dir_path: str | None = None) -> None:
        p = pathlib.Path(dir_path or os.environ.get("MULTIWAY_DIR") or DEFAULT_DIR)
        base = pathlib.Path(__file__).resolve().parents[3]
        self.dir = p if p.is_absolute() else (base / p)
        self._rows: list[dict] = []
        self._loaded = False

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return bool(self._rows)
        self._loaded = True
        if not self.dir.is_dir():
            return False
        for f in sorted(self.dir.glob("*.json")):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:                                  # noqa: BLE001
                continue
            if not d.get("solved") or not d.get("actions"):
                continue
            if int(d.get("n_active") or 0) < 3:
                continue                                       # 只收 3 人以上 ✓
            self._rows.append(d)
        return bool(self._rows)

    def lookup(self, query: dict) -> dict | None:
        """查多人节点；**开关关闭或未命中 ⇒ None** ✓（诚实 miss，不猜测 ✓）。"""
        if not _enabled():
            return None
        if not self._ensure_loaded():
            return None
        n = int(query.get("n_active") or 0)
        if n < 3:
            return None
        street = str(query.get("street") or "")
        board = tuple(sorted(str(c) for c in (query.get("board") or [])))
        pcls = _pot_class(float(query.get("pot_bb") or 0.0))
        node = str(query.get("node") or "first_in")
        actor = str(query.get("actor") or "").upper()
        for r in self._rows:
            if str(r.get("street") or "") != street:
                continue
            if tuple(sorted(str(c) for c in (r.get("board") or []))) != board:
                continue
            if int(r.get("n_active") or 0) != n:
                continue
            if _pot_class(float(r.get("pot_bb") or 0.0)) != pcls:
                continue
            if str(r.get("node") or "first_in") != node:
                continue
            if str(r.get("actor_position") or "").upper() != actor:
                continue
            return {
                "solved": True, "solver_name": "multiway",
                "model": f"multiway:{r.get('spot_id')}",
                "actions": [dict(a) for a in (r.get("actions") or [])],
                "n_active": n, "is_approximate": False,
                "source": "multiway_spot", "source_note": r.get("source_note"),
                "raw": {"spot_id": r.get("spot_id"), "n_active": n,
                         "synthetic": bool(r.get("synthetic"))},
            }
        return None

    def stats(self) -> dict:
        ok = self._ensure_loaded()
        return {"enabled": _enabled(), "dir": str(self.dir),
                 "n_rows": len(self._rows), "loaded": ok}


def get_multiway_table(dir_path: str | None = None) -> MultiwaySpotTable:
    key = str(dir_path or os.environ.get("MULTIWAY_DIR") or DEFAULT_DIR)
    if key not in _CACHE:
        _CACHE[key] = MultiwaySpotTable(dir_path)
    return _CACHE[key]
