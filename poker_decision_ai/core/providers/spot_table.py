"""SOLVER-1 — spot 库查表（realtime 路径的 solver 供给）。

spot 套件由 `scripts/solve_spots.py` 离线求解（真实 postflop CFR），
落盘 `data/solver/spots_r1/{spot_id}.json` + manifest。本模块把 spot 库
装成毫秒级查表：

  * exact 命中：street+board+ranges+pot/stack 全同 ⇒ 直接复用求解结果
    （is_approximate=False，equilibrium=True）；
  * nearest 命中：street+ranges+pot 档+board 纹理类相同的**代表 spot**
    （is_approximate=True，equilibrium=True——策略来自均衡解，但套用到了
    相邻纹理上，如实降级标注）；
  * miss → None（调用方诚实降级到 mock，绝不伪造）。

查表延迟：首次加载 manifest+索引（~百 ms，进程内缓存），单次查询为内存
字典命中（≪50ms）。
"""
from __future__ import annotations

import json
import pathlib
import threading
from typing import Any

SPOT_TABLE_VERSION = "solver1-spot-table-v2-m4"
DEFAULT_SPOT_DIR = "data/solver/spots_r1"
#: M4：facing-bet spot 库（node=facing_bet 记录；与首入库同 CLI/同范围声明）
DEFAULT_FACING_SPOT_DIR = "data/solver/spots_fb"

_LOCK = threading.Lock()
_CACHE: dict[str, "SpotTable"] = {}


def _canon_board(cards) -> tuple:
    """牌面**规范化**（排序）——修"顺序敏感"导致的 exact 漏配。

    真实客户端报的公共牌顺序是任意的，若索引/比较用原始顺序，
    3 张牌只有 1/6 排列能命中（实测均匀覆盖 0.055% vs 理论 0.217%）。
    """
    return tuple(sorted(str(c) for c in (cards or [])))


def _texture_class(board: list[str]) -> tuple:
    """声明式纹理类：(是否对子, 是否单调, 花色数)。"""
    ranks = [c[0] for c in board]
    suits = [c[1] for c in board if len(c) > 1]
    return (len(set(ranks)) < len(ranks), len(set(suits)) == 1, len(set(suits)))


def _pot_class(pot_bb: float) -> str:
    if pot_bb < 8:
        return "SRP"
    if pot_bb < 15:
        return "3BP"
    return "big"


def _bet_bucket(bet_ratio: float) -> str | None:
    """M4：facing-bet 尺寸桶（bet_ratio = to_call/(pot−to_call)，声明式）。

    mid∈[0.45,0.80]（对手 0.5–0.8 pot 下注）；large∈(0.80,3.0]；shove>3.0
    （有效全下/超池 jam）。桶外（<0.45 的 min-raise 类）→ None=诚实 miss
    （套件未解该尺寸，绝不跨桶套用）。
    """
    if 0.45 <= bet_ratio <= 0.80:
        return "mid"
    if 0.80 < bet_ratio <= 3.0:
        return "large"
    if bet_ratio > 3.0:
        return "shove"
    return None


class SpotTable:
    def __init__(self, spot_dir: str = DEFAULT_SPOT_DIR,
                 facing_spot_dir: str | None = DEFAULT_FACING_SPOT_DIR):
        self.dir = pathlib.Path(spot_dir)
        self.facing_dir = (pathlib.Path(facing_spot_dir)
                           if facing_spot_dir else None)
        self._index: dict[tuple, dict] = {}
        self._by_bucket: dict[tuple, list[dict]] = {}
        # M4：facing-bet 索引（键含尺寸桶；与首入索引严格分离）
        self._index_facing: dict[tuple, dict] = {}
        self._by_bucket_facing: dict[tuple, list[dict]] = {}
        self._loaded = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    @staticmethod
    def _actor_of(rec: dict) -> str:
        """记录的行动者（oop/ip）。r1 记录自带 `actor`；缺省 oop（该库为 OOP 首入解）。"""
        return str(rec.get("actor") or "oop").lower()

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        with self._lock:
            if self._loaded:
                return True
            manifest_path = self.dir / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for sid in manifest.get("solved_ids") or []:
                    p = self.dir / f"{sid}.json"
                    try:
                        rec = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if not rec.get("solved"):
                        continue
                    q = rec.get("query") or {}
                    rng = q.get("ranges") or {}
                    oop = str(rng.get("oop") or q.get("oop_range") or "")
                    ip = str(rng.get("ip") or q.get("ip_range") or "")
                    key = (str(q.get("street")),
                           _canon_board(q.get("board")),
                           oop, ip,
                           _pot_class(float(q.get("pot_bb") or 0.0)))
                    self._index[key] = rec
                    bkey = (str(q.get("street")), oop, ip,
                            _pot_class(float(q.get("pot_bb") or 0.0)),
                            _texture_class([str(c) for c in (q.get("board") or [])]))
                    self._by_bucket.setdefault(bkey, []).append(rec)
            # M4：facing-bet 库（可选目录；缺失 = 无 facing 锚点，诚实 miss）
            if self.facing_dir is not None:
                fmanifest = self.facing_dir / "manifest.json"
                if fmanifest.is_file():
                    manifest = json.loads(
                        fmanifest.read_text(encoding="utf-8"))
                    for sid in manifest.get("solved_ids") or []:
                        p = self.facing_dir / f"{sid}.json"
                        try:
                            rec = json.loads(p.read_text(encoding="utf-8"))
                        except Exception:
                            continue
                        if not rec.get("solved"):
                            continue
                        self._ingest_facing(rec)
            self._loaded = True
            return True

    # ------------------------------------------------------------------
    @staticmethod
    def _export_to_rows(export: dict) -> list[dict]:
        """节点导出 → 归一动作行 [{name,size_bb,frequency,ev}]（bb 口径）。

        CLI 输出动作名是 Debug 形态（Fold/Call/Raise/AllIn）——这里做与
        judge 消费端一致的归一（小写；AllIn→allin），保证 facing 索引的
        动作行与 `strategy_anchor.ACTIONS` 同词汇表。
        """
        rows = []
        for a in export.get("actions") or []:
            if not isinstance(a, dict):
                continue
            name = str(a.get("name") or "").strip().lower() \
                .replace("-", "").replace("_", "")
            if name == "allin":
                name = "allin"
            rows.append({"name": name,
                         "size_bb": a.get("amount_bb"),
                         "frequency": float(a.get("frequency") or 0.0),
                         "ev": a.get("ev_bb"),
                         # P30-9-DC：逐手数组（与 private_cards_actor 对齐）
                         "ev_per_hand": a.get("ev_per_hand_bb"),
                         "freq_per_hand": a.get("freq_per_hand")})
        return rows

    def _ingest_facing(self, rec: dict) -> None:
        """facing-bet 记录入索引：每个合法导出节点一条（键含尺寸桶）。"""
        q = rec.get("query") or {}
        rng = q.get("ranges") or {}
        oop = str(rng.get("oop") or q.get("oop_range") or "")
        ip = str(rng.get("ip") or q.get("ip_range") or "")
        street = str(rec.get("street") or q.get("street") or "")
        exports = rec.get("exports") or []
        decls = {d.get("export_id"): d
                 for d in (rec.get("export_decl") or [])}
        # 声明路径（canonical_canon / canonical_large / inverse_donk）→ 导出行
        # （按 path 对齐，防实现返回顺序漂移；error 导出诚实跳过）
        by_path = {tuple(e.get("history") or []): e for e in exports
                   if isinstance(e, dict) and not e.get("error")}
        for export_id, d in decls.items():
            e = by_path.get(tuple(d.get("path") or []))
            if e is None:
                continue                                  # 该节点导出失败（诚实）
            rows = self._export_to_rows(e)
            rows = [r for r in rows if r["frequency"] > 0.0
                    and r["name"] in ("fold", "call", "raise", "allin")]
            if not rows:
                continue
            ratio = float(e.get("bet_ratio") or 0.0)
            bucket = _bet_bucket(ratio)
            if bucket is None:
                continue
            pot_bb = float(e.get("pot_bb") or 0.0)
            pot_cls = _pot_class(pot_bb)
            board = [str(c) for c in (q.get("board") or [])]
            actor = str(e.get("actor") or "oop")
            family = str(rec.get("family") or "")
            node_rec = {
                "spot_id": f"{rec.get('spot_id')}#{export_id}",
                "node": "facing_bet",
                "family": family,         # LIMP / SRP / 3BP（pot 族）
                "actor_role": actor,      # 节点行动者（=面对下注的一方）
                "street": street,
                "board": board,
                "pot_bb": pot_bb,
                "to_call_bb": float(e.get("to_call_bb") or 0.0),
                "bet_ratio": ratio,
                "bet_bucket": bucket,
                "ranges": {"oop": oop, "ip": ip},
                "actions": rows,
                "root_spot_id": rec.get("spot_id"),
                "exploitability_bb": rec.get("exploitability_bb"),
                "simplifications": list(rec.get("simplifications") or []),
            }
            # 键含 actor_role + pot 族：同一 (街/牌面/桶) 的 oop-facing 与
            # ip-facing 是不同节点（donk vs c-bet 面）；LIMP 与 SRP 同 pot
            # 档但范围/上下文不同——族不同绝不互相覆盖。
            key = (street, tuple(board), oop, ip, pot_cls, bucket, actor,
                   family)
            self._index_facing[key] = node_rec
            bkey = (street, oop, ip, pot_cls, bucket, actor, family,
                    _texture_class(board))
            self._by_bucket_facing.setdefault(bkey, []).append(node_rec)

    # ------------------------------------------------------------------
    def lookup(self, query: dict) -> dict | None:
        """查询 → SolverResult dict（可直接喂 SolverAgent）或 None=miss。

        M4：query["node"]="first_in"（缺省，兼容旧调用）| "facing_bet"。
        facing_bet 需要 query["bet_ratio"]=to_call/(pot−to_call)；尺寸桶
        不匹配 → 诚实 miss（绝不跨尺寸桶套用策略）。
        """
        if not self._ensure_loaded():
            return None
        node_type = str(query.get("node") or "first_in")
        if node_type == "facing_bet":
            return self._lookup_facing(query)
        street = str(query.get("street") or "")
        board = _canon_board(query.get("board"))
        texture = _texture_class(list(board))
        pot_bb = float(query.get("pot_bb") or 0.0)
        pot_cls = _pot_class(pot_bb)
        ranges = query.get("ranges") or {}
        oop = str(ranges.get("oop") or "")
        ip = str(ranges.get("ip") or "")

        if oop and ip:
            want_actor = str(query.get("actor") or "").lower()
            exact = self._index.get((street, board, oop, ip, pot_cls))
            if exact is not None and (not want_actor
                                       or self._actor_of(exact) == want_actor):
                return self._to_result(exact, mode="exact", approx=False)
            bucket = self._by_bucket.get((street, oop, ip, pot_cls, texture))
            # P30-9-ES：actor 过滤（缺省不约束，保持向后兼容）
            if bucket and want_actor:
                bucket = [r for r in bucket if self._actor_of(r) == want_actor] or None
            if bucket:
                nearest = bucket[0]             # 声明式：桶内代表 spot
                return self._to_result(nearest, mode="nearest", approx=True)
            return None
        # 无具体 ranges → 按 (street, pot 档, 纹理类) 找桶（range-set 路径：
        # 桶键自带套件声明的 oop/ip 范围字符串）。
        # 桶内 board 与查询完全一致 ⇒ exact（非近似）；否则最近（近似，降级标注）。
        for bkey, rows in sorted(self._by_bucket.items()):
            if bkey[0] == street and bkey[3] == pot_cls and bkey[4] == texture:
                for row in rows:
                    rq = row.get("query") or {}
                    if _canon_board(rq.get("board")) == board:
                        return self._to_result(row, mode="range_set_exact",
                                               approx=False)
                nearest = rows[0]
                return self._to_result(nearest, mode="range_set_nearest",
                                       approx=True)
        return None

    # ------------------------------------------------------------------
    def _lookup_facing(self, query: dict) -> dict | None:
        """M4：facing-bet 节点查表（HU）。

        键 = 街+牌面+范围+pot 档+尺寸桶+**hero 角色**（hero 是套件定义的
        oop 方还是 ip 方——SRP 的 oop=跟注方/ip=加注方、3BP 的 oop=3bet 方/
        ip=跟注方；query["facing_actor"] 由 solver agent 从翻前动作推导，
        推导不出 → 诚实 miss）。
        """
        street = str(query.get("street") or "")
        board = _canon_board(query.get("board"))
        texture = _texture_class(list(board))
        pot_bb = float(query.get("pot_bb") or 0.0)
        pot_cls = _pot_class(pot_bb)
        bucket = _bet_bucket(float(query.get("bet_ratio") or 0.0))
        if bucket is None:
            return None                          # 尺寸桶外 → 诚实 miss
        role = str(query.get("facing_actor") or "")
        if role not in ("oop", "ip"):
            return None                          # 角色未知 → 诚实 miss
        family = str(query.get("pot_family") or "")
        ranges = query.get("ranges") or {}
        oop = str(ranges.get("oop") or "")
        ip = str(ranges.get("ip") or "")
        if oop and ip:
            exact = self._index_facing.get(
                (street, board, oop, ip, pot_cls, bucket, role, family))
            if exact is not None:
                return self._to_result(exact, mode="exact", approx=False)
            fbucket = self._by_bucket_facing.get(
                (street, oop, ip, pot_cls, bucket, role, family, texture))
            if fbucket:
                nearest = fbucket[0]
                return self._to_result(nearest, mode="nearest", approx=True)
            return None
        # 先全桶扫 exact board（精确优先），无 exact 才取首个匹配桶的
        # 代表节点（nearest，近似降级标注）。
        # EXE-1 修复（R6 回归）：_by_bucket_facing 键自 R6 加入 family 后为
        # 8 元组 (street, oop, ip, pot_cls, bucket, role, family, texture)——
        # 本扫描原比较 bkey[6]==texture 是 family 插入前的 7 元组索引，导致
        # range-set 路径全部 miss（生产显式 ranges 走 exact 索引路径不受
        # 影响）。恢复声明语义：按 (street, pot 档, 尺寸桶, 角色, 纹理类)
        # 匹配，family 不作为桶过滤（族隔离只作用于索引键，防覆盖）。
        exact = None
        for bkey, rows in sorted(self._by_bucket_facing.items()):
            if bkey[0] == street and bkey[3] == pot_cls \
                    and bkey[4] == bucket and bkey[5] == role \
                    and bkey[7] == texture:
                for row in rows:
                    if _canon_board(row.get("board")) == board:
                        exact = row
                        break
            if exact is not None:
                break
        if exact is not None:
            return self._to_result(exact, mode="range_set_exact", approx=False)
        for bkey, rows in sorted(self._by_bucket_facing.items()):
            if bkey[0] == street and bkey[3] == pot_cls \
                    and bkey[4] == bucket and bkey[5] == role \
                    and bkey[7] == texture:
                return self._to_result(rows[0], mode="range_set_nearest",
                                       approx=True)
        return None

    @staticmethod
    def _to_result(rec: dict, *, mode: str, approx: bool) -> dict:
        q = rec.get("query") or {}
        node = str(rec.get("node") or "first_in")
        if node == "facing_bet":
            # M4：facing-bet 节点记录（actions/ev 已是节点级均衡聚合）
            return {
                "solved": True,
                "solver_name": "postflop",
                "model": f"spot_table:{rec.get('spot_id')} mode={mode}",
                "actions": [
                    {"name": a["name"], "size_bb": a.get("size_bb"),
                     "frequency": a["frequency"], "ev": a.get("ev")}
                    for a in (rec.get("actions") or [])],
                "bet_sizings_bb": sorted({a.get("size_bb")
                                          for a in (rec.get("actions") or [])
                                          if a.get("size_bb") is not None}),
                "notes": ("facing-bet spot 库查表（真实 CFR 均衡解，节点级导出；"
                          "nearest 模式=代表节点策略套用到相邻纹理，已降级标注）"),
                "is_approximate": approx,
                "raw": {
                    "source": "postflop_spot_lookup",
                    "near_nash_equilibrium": True,
                    "is_mock": False,
                    "lookup_mode": mode,
                    "node": "facing_bet",
                    "spot_id": rec.get("spot_id"),
                    "root_spot_id": rec.get("root_spot_id"),
                    "family": rec.get("family"),
                    "actor_role": rec.get("actor_role"),
                    "bet_ratio": rec.get("bet_ratio"),
                    "bet_bucket": rec.get("bet_bucket"),
                    "exploitability_bb": rec.get("exploitability_bb"),
                    "simplifications": list(rec.get("simplifications") or []),
                },
            }
        return {
            "solved": True,
            "solver_name": "postflop",
            "model": (f"spot_table:{rec.get('spot_id')} mode={mode}"),
            "actions": [
                {"name": a["name"], "size_bb": a["amount_bb"],
                 "frequency": a["frequency"], "ev": a["ev_bb"],
                 # P30-9-DC：逐手数组透传（与 private_cards_actor 对齐）
                 "ev_per_hand": a.get("ev_per_hand_bb"),
                 "freq_per_hand": a.get("freq_per_hand")}
                for a in (rec.get("actions") or [])],
            # 逐手对齐用的手牌列表（无 ⇒ 空列表，消费端诚实降级）
            "hands": list(rec.get("private_cards_actor") or []),
            "bet_sizings_bb": sorted({a["amount_bb"] for a in
                                      (rec.get("actions") or [])
                                      if a.get("amount_bb") is not None}),
            "notes": ("postflop spot 库查表（真实 CFR 均衡解；nearest 模式="
                      "代表 spot 策略套用到相邻纹理，已降级标注）"),
            "is_approximate": approx,
            "raw": {
                "source": "postflop_spot_lookup",
                "near_nash_equilibrium": True,
                "is_mock": False,
                "lookup_mode": mode,
                "node": "first_in",
                "spot_id": rec.get("spot_id"),
                "spot_street": q.get("street"),
                "exploitability_bb": rec.get("exploitability_bb"),
                "iterations": rec.get("iterations"),
                "simplifications": list(rec.get("simplifications") or []),
            },
        }

    def declared_ranges(self, street: str, pot_cls: str) -> tuple[str, str] | None:
        """该 (street, pot 档) 的套件声明范围（供 DEEP 现场求解注入）。"""
        if not self._ensure_loaded():
            return None
        for (kind, oop, ip, pcls, _tex) in sorted(self._by_bucket):
            if kind == street and pcls == pot_cls:
                return (oop, ip)
        return None

    def stats(self) -> dict:
        ok = self._ensure_loaded()
        return {"loaded": ok, "n_spots": len(self._index),
                "n_buckets": len(self._by_bucket),
                "n_facing_nodes": len(self._index_facing),
                "n_facing_buckets": len(self._by_bucket_facing),
                "version": SPOT_TABLE_VERSION,
                "dir": str(self.dir),
                "facing_dir": (str(self.facing_dir)
                               if self.facing_dir else None)}


def get_spot_table(spot_dir: str = DEFAULT_SPOT_DIR) -> SpotTable:
    with _LOCK:
        if spot_dir not in _CACHE:
            _CACHE[spot_dir] = SpotTable(spot_dir)
        return _CACHE[spot_dir]
