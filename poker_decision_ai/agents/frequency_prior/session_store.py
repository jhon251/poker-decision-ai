"""M3（R5）— 会话级个体 villain 先验：realtime 链路逐手累积 + 持久化。

任务书（round5_M3 任务 1）：realtime 链路内**逐手**累积 villain 观测
（actions by street/position），生成会话内个体频率（Wilson 区间 + 时间衰减 +
样本量）；持久化 per-player 存储（沿用 R5 identity 契约），会话恢复不丢。

数据流（生产接线）：
  * 喂入：宿主进程（bench 回路 / 连续决策 loop / UI runner）在每手结束后
    调 `feed_session_hand(session_id, street_actions, ...)`——逐街动作行经
    `domain/poker/frequency_prior.observations_from_street_actions` 重放推导
    facing_bet（绝不猜测），只保留**非 hero** 观测；
  * 读出：`FrequencyPriorAgent`（两级先验的个体层）；strategy_anchor 的
    villain_frequency_prior 偏离通道因此升级为 个体 > 群体 两级
    （session_individual > corpus_individual > corpus_population）；
  * 持久化：per-session JSON 文件（per-player 分桶），启动时载入即会话恢复；
    损坏/未知 schema → fail-closed 空库 + warning（不编造）。

身份契约（沿用 R5 `player_history.PlayerIdentity` 语义）：
  * player_id = 稳定身份键；display_name 可同（identity_from_name）；
  * 空/缺失 player_id → identity_status=UNKNOWN：**不入库、不参与先验**
    （fail-closed + IDENTITY_UNKNOWN warning）；同名玩家切换座位/回归 →
    同一 player_id 桶自然延续（会话恢复语义）。

诚实边界：本模块是确定性簿记 + 查询，绝不调用 LLM；不产生任何编造频率
（不足 min_sample_size 的查询由 domain 层回退语义负责）。
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from poker_decision_ai.domain.poker import frequency_prior as fp

SESSION_STORE_VERSION = 1

#: 街顺序（确定性持久化/回放顺序）
STREET_ORDER: tuple[str, ...] = ("preflop", "flop", "turn", "river")

#: 观测行的持久化字段（FrequencyObservation 的 JSON 形态）
_OBS_FIELDS = ("player_id", "street", "facing_bet", "action", "position", "ts")

#: 进程级注册表（同进程多会话互不混桶；显式注入可完全绕开）
_REGISTRY: dict[str, "VillainSessionStore"] = {}


class VillainSessionStore:
    """单会话的 per-player 观测簿记 + 个体先验查询 + 磁盘持久化。"""

    def __init__(self, *, session_id: str,
                 persist_dir: str | None = None,
                 min_sample_size: int = fp.DEFAULT_MIN_SAMPLE_SIZE,
                 half_life_days: float = fp.DEFAULT_HALF_LIFE_DAYS,
                 wilson_z: float = fp.DEFAULT_WILSON_Z) -> None:
        self.session_id = str(session_id or "realtime")
        self.min_sample_size = max(1, int(min_sample_size))
        self.cfg = fp.FrequencyPriorConfig(
            min_sample_size=self.min_sample_size,
            half_life_days=float(half_life_days),
            wilson_z=float(wilson_z))
        self.persist_path: Path | None = None
        if persist_dir:
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.session_id)[:80] \
                or "session"
            self.persist_path = Path(str(persist_dir)) / f"{safe}.json"
        self._obs: dict[str, list[fp.FrequencyObservation]] = {}
        self._ident: dict[str, dict] = {}
        self.warnings: list[dict] = []
        if self.persist_path is not None:
            self._load()

    # ---------------------------------------------------------- 身份契约
    @staticmethod
    def identity_of(player_id: Any) -> dict:
        """R5 identity 契约：有名 → RECOGNIZED(player_id=名)；无名 → UNKNOWN。"""
        pid = str(player_id or "").strip()
        if not pid:
            return {"player_id": "", "display_name": "",
                    "identity_status": "UNKNOWN", "confidence": 0.0}
        return {"player_id": pid, "display_name": pid,
                "identity_status": "RECOGNIZED", "confidence": 1.0}

    # -------------------------------------------------------------- 喂入
    def record(self, obs: fp.FrequencyObservation) -> bool:
        """记录一条观测；身份 UNKNOWN → 拒绝（fail-closed）+ warning。"""
        ident = self.identity_of(obs.player_id)
        if ident["identity_status"] != "RECOGNIZED":
            self.warnings.append({
                "code": "IDENTITY_UNKNOWN",
                "message": "observation without usable player_id; not stored "
                           "(identity contract: UNKNOWN never persists)"})
            return False
        self._obs.setdefault(ident["player_id"], []).append(obs)
        self._ident[ident["player_id"]] = ident
        return True

    def record_hand(self, street_actions: dict, *,
                    positions_by_player: dict | None = None,
                    hero_id: str | None = None,
                    ts: float | None = None) -> int:
        """一手逐街动作行 → 观测（重放推导 facing_bet；跳过 hero）。"""
        n = 0
        for street in STREET_ORDER:
            rows = (street_actions or {}).get(street) or []
            if not rows:
                continue
            for o in fp.observations_from_street_actions(
                    rows, street=street,
                    positions_by_player=positions_by_player, ts=ts):
                if hero_id and o.player_id == str(hero_id):
                    continue
                if self.record(o):
                    n += 1
        return n

    # -------------------------------------------------------------- 读出
    def observations(self, player_id: str) -> list[fp.FrequencyObservation]:
        return list(self._obs.get(str(player_id or "")) or [])

    def players(self) -> list[str]:
        return sorted(self._obs.keys())

    def counts(self, player_id: str) -> dict:
        """该玩家分 context 的观测数（诚实样本量口径，n=原始计数）。"""
        obs = self.observations(player_id)
        facing = sum(1 for o in obs if o.facing_bet is True)
        no_bet = sum(1 for o in obs if o.facing_bet is False)
        return {"facing_bet": facing, "no_bet": no_bet, "total": len(obs)}

    def prior(self, player_id: str, *, position: str | None = None,
              street: str | None = None,
              facing_bet: bool | None) -> fp.FrequencyPrior:
        """个体先验查询（域层 action_frequencies：Wilson + 衰减 + 回退链）。"""
        return fp.action_frequencies(
            self.observations(player_id),
            player_id=str(player_id or ""), facing_bet=facing_bet,
            position=position, street=street, config=self.cfg)

    def summary(self) -> dict:
        return {pid: self.counts(pid) for pid in self.players()}

    # ---------------------------------------------------------- 持久化
    def to_dict(self) -> dict:
        players: dict[str, dict] = {}
        for pid in self.players():
            ident = dict(self._ident.get(pid)
                         or self.identity_of(pid))
            players[pid] = {
                "identity": ident,
                "updated_at": time.time(),
                "observations": [self._obs_row(o) for o in self._obs[pid]],
            }
        return {"version": SESSION_STORE_VERSION,
                "session_id": self.session_id,
                "saved_at": time.time(),
                "config": dict(self.cfg.to_dict()),
                "players": players}

    @classmethod
    def _obs_row(cls, o: fp.FrequencyObservation) -> list:
        return [o.player_id, o.street, o.facing_bet, o.action, o.position, o.ts]

    @classmethod
    def _obs_from_row(cls, row: Any) -> fp.FrequencyObservation | None:
        if not isinstance(row, (list, tuple)) or len(row) != len(_OBS_FIELDS):
            return None
        pid, street, facing, action, position, ts = row
        if action not in ("check", "call", "fold", "bet", "raise") \
                or not isinstance(facing, bool):
            return None                      # 未知行 → 丢弃（不编造）
        return fp.FrequencyObservation(
            player_id=str(pid), street=str(street), facing_bet=bool(facing),
            action=str(action),
            position=(str(position) if position is not None else None),
            ts=(float(ts) if isinstance(ts, (int, float)) else None))

    def save(self) -> dict:
        """原子落盘；无 persist_path → 跳过（进程内语义，显式声明）。"""
        if self.persist_path is None:
            return {"saved": False, "reason": "no_persist_path"}
        blob = json.dumps(self.to_dict(), ensure_ascii=False)
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.persist_path.with_suffix(".json.tmp")
            tmp.write_text(blob, encoding="utf-8")
            tmp.replace(self.persist_path)
            return {"saved": True, "path": str(self.persist_path),
                    "players": len(self._obs)}
        except OSError as exc:
            self.warnings.append({"code": "SESSION_PERSIST_WRITE_FAILED",
                                  "message": f"session store save failed: {exc}"})
            return {"saved": False, "reason": str(exc)}

    def _load(self) -> None:
        """会话恢复：载入 per-player 桶；损坏/未知版本 → 空库 + warning。"""
        assert self.persist_path is not None
        if not self.persist_path.exists():
            return
        try:
            blob = json.loads(
                self.persist_path.read_text(encoding="utf-8"))
        except Exception as exc:             # noqa: BLE001（损坏文件如实降级）
            self.warnings.append({
                "code": "SESSION_PERSIST_READ_FAILED",
                "message": (f"session store unreadable "
                            f"({self.persist_path}): {exc}; starting empty")})
            return
        if not isinstance(blob, dict) \
                or int(blob.get("version") or 0) != SESSION_STORE_VERSION:
            self.warnings.append({
                "code": "SESSION_PERSIST_SCHEMA_MISMATCH",
                "message": "session store version mismatch; starting empty"})
            return
        players = blob.get("players") or {}
        for pid, entry in players.items():
            if not isinstance(entry, dict):
                continue
            ident = entry.get("identity") or {}
            if str(ident.get("identity_status") or "") != "RECOGNIZED":
                continue                     # UNKNOWN 身份不入库（fail-closed）
            rows = entry.get("observations") or []
            restored = 0
            for row in rows:
                o = self._obs_from_row(row)
                if o is not None and self.identity_of(o.player_id)[
                        "identity_status"] == "RECOGNIZED":
                    self._obs.setdefault(str(pid), []).append(o)
                    restored += 1
            if restored:
                self._ident[str(pid)] = {
                    "player_id": str(pid),
                    "display_name": str(ident.get("display_name") or pid),
                    "identity_status": "RECOGNIZED",
                    "confidence": float(ident.get("confidence") or 1.0)}

    def reset(self) -> None:
        """清空进程内观测（测试/换桌显式调用；不动磁盘文件）。"""
        self._obs.clear()
        self._ident.clear()


# ----------------------------------------------------------------------
# 进程级注册表 + 生产喂入 API
# ----------------------------------------------------------------------

def session_store_config(cfg_map: dict) -> dict:
    """config.yaml::frequency_prior 的会话层键（缺项=声明缺省）。"""
    cfg_map = dict(cfg_map or {})
    return {
        "enabled": bool(cfg_map.get("session_enabled", True)),
        "min_sample_size": int(cfg_map.get("session_min_sample_size",
                                           fp.DEFAULT_MIN_SAMPLE_SIZE)),
        "persist_dir": (str(cfg_map.get("session_persist_dir"))
                        if cfg_map.get("session_persist_dir")
                        else "data/cache/villain_sessions"),
        "half_life_days": float(cfg_map.get("half_life_days",
                                            fp.DEFAULT_HALF_LIFE_DAYS)),
        "wilson_z": float(cfg_map.get("wilson_z", fp.DEFAULT_WILSON_Z)),
    }


def get_or_create_session_store(session_id: str, cfg_map: dict | None = None,
                                warnings: list | None = None
                                ) -> VillainSessionStore:
    """按 session_id 取（或懒建）会话 store（进程级注册表）。"""
    scfg = session_store_config(cfg_map)
    sid = str(session_id or "realtime")
    store = _REGISTRY.get(sid)
    if store is None:
        store = VillainSessionStore(
            session_id=sid,
            persist_dir=scfg["persist_dir"] if scfg["enabled"] else None,
            min_sample_size=scfg["min_sample_size"],
            half_life_days=scfg["half_life_days"],
            wilson_z=scfg["wilson_z"])
        _REGISTRY[sid] = store
    if warnings is not None and store.warnings:
        warnings.extend(store.warnings)
        store.warnings = []
    return store


def feed_session_hand(session_id: str, street_actions: dict, *,
                      positions_by_player: dict | None = None,
                      hero_id: str | None = None,
                      ts: float | None = None,
                      cfg_map: dict | None = None) -> dict:
    """生产喂入口：每手结束后由宿主进程调用（bench/loop/UI runner）。

    返回 {"recorded": n, "players": [...]}；喂入只增不删（确定性簿记）。
    """
    store = get_or_create_session_store(session_id, cfg_map)
    n = store.record_hand(street_actions,
                          positions_by_player=positions_by_player,
                          hero_id=hero_id, ts=ts)
    return {"recorded": n, "players": store.players(),
            "counts": store.summary()}
