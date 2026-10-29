"""EXE-1 — play_history 训练数据落盘（JSONL；每手一行）。

产品数据底座（"越打越懂你"）：每手对局结束后**立即**追加一行结构化 JSON：
  手牌/公共牌/桌面阵容/全部 hero 决策（含用户动作 + AI 参考建议 + 决策
  provenance + 一致性）/净收益三口径/终局原因。文件可被离线 replay 与
  metrics 消费（本模块 load_session 为最小读取器）。

诚实边界：只记录真实发生的对局事实与真实决策溯源；中止手同样落盘
（aborted 字段声明原因，绝不丢弃/替换）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

PLAY_HISTORY_SCHEMA = "poer-play-history-v1"


def session_filename(session_id: str) -> str:
    return f"play_session_{session_id}.jsonl"


class PlayHistoryWriter:
    """逐手 JSONL 追加写（每行一手；UTF-8；崩溃安全——行级 flush）。"""

    def __init__(self, history_dir: str | os.PathLike, session_id: str,
                 *, stake: str, label: str, seed_base: int,
                 lineup: list[dict], deterministic: bool,
                 table_size: int = 6, stack_bb: int = 100,
                 focus_position: str = "auto", source: str = "play"):
        self.dir = Path(history_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session_id = str(session_id)
        self.path = self.dir / session_filename(self.session_id)
        self._n_hands = 0
        # 会话头（第一行；schema 声明 + 桌面静态信息）
        self._write_line({
            "record": "session_head",
            "schema": PLAY_HISTORY_SCHEMA,
            "session_id": self.session_id,
            "created_at": _utc_now(),
            "stake": str(stake),
            "stake_label": str(label),
            "seed_base": int(seed_base),
            "deterministic_seed": bool(deterministic),
            "lineup": list(lineup),
            "table_size": int(table_size),
            "stack_bb": int(stack_bb),
            "focus_position": str(focus_position),
            "source": str(source),          # play=对打 | external=外部牌谱导入
        })

    # ------------------------------------------------------------------
    def _write_line(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False, default=str)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    # ------------------------------------------------------------------
    def write_hand(self, *, hand_index: int, deal_seed: int, button: int,
                   hero_hole: list[str], board: list[str],
                   street_reached: str, n_players_dealt: int,
                   hero_decisions: list[dict], net_bb: float | None,
                   net_bb_raked: float | None, net_bb_raked_full: float | None,
                   rake_chips: float, pot_chips: float,
                   terminal_reason: str | None, aborted: str | None,
                   hand_wall_ms: float,
                   timeline: list[dict] | None = None,
                   winners: list[int] | None = None,
                   revealed: list[dict] | None = None) -> None:
        self._write_line({
            "record": "hand",
            "schema": PLAY_HISTORY_SCHEMA,
            "session_id": self.session_id,
            "hand_index": int(hand_index),
            "deal_seed": int(deal_seed),
            "button": int(button),
            "hero_hole": list(hero_hole),
            "board": [str(c) for c in (board or [])],
            "street_reached": str(street_reached),
            "n_players_dealt": int(n_players_dealt),
            "hero_decisions": list(hero_decisions),
            "net_bb": net_bb,
            "net_bb_raked": net_bb_raked,
            "net_bb_raked_full": net_bb_raked_full,
            "rake_chips": round(float(rake_chips or 0.0), 2),
            "pot_chips": float(pot_chips),
            "terminal_reason": terminal_reason,
            "aborted": aborted,
            "hand_wall_ms": round(float(hand_wall_ms), 1),
            # 回放数据（v1 附加字段；旧文件缺失时回放退化为只看结果）
            "timeline": [dict(e) for e in (timeline or [])],
            "winners": [int(s) for s in (winners or [])],
            "revealed": [dict(r) for r in (revealed or [])],
            "written_at": _utc_now(),
        })
        self._n_hands += 1

    # ------------------------------------------------------------------
    def write_session_tail(self, summary: dict) -> None:
        self._write_line({"record": "session_tail", "schema": PLAY_HISTORY_SCHEMA,
                          "session_id": self.session_id,
                          "n_hands_records": self._n_hands,
                          "summary": dict(summary),
                          "written_at": _utc_now()})

    @property
    def n_hands_written(self) -> int:
        return self._n_hands


# ═══════════════════════ 读取端（replay/metrics 最小消费契约） ═══════════
def load_session(path: str | os.PathLike) -> dict:
    """读一个 play_history 会话文件 → {head, hands[], tail|None}。

    校验：schema 必须匹配 PLAY_HISTORY_SCHEMA（v1）；不匹配 → ValueError
    （消费方显式失败，绝不静默猜格式）。
    """
    head: dict | None = None
    hands: list[dict] = []
    tail: dict | None = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("schema") != PLAY_HISTORY_SCHEMA:
                raise ValueError(f"unsupported play-history schema: "
                                 f"{row.get('schema')!r} in {path}")
            rec = row.get("record")
            if rec == "session_head":
                head = row
            elif rec == "hand":
                hands.append(row)
            elif rec == "session_tail":
                tail = row
    if head is None:
        raise ValueError(f"play-history file missing session_head: {path}")
    return {"head": head, "hands": hands, "tail": tail}


def iter_hand_lines(path: str | os.PathLike) -> Iterator[dict]:
    """流式逐手读取（大文件友好）。"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("record") == "hand":
                yield row


def session_metrics(path: str | os.PathLike) -> dict:
    """最小离线指标：手数/净 bb 三口径/decisions/advisor 一致率。"""
    hands = list(iter_hand_lines(path))
    done = [h for h in hands if h.get("net_bb") is not None]
    n = max(1, len(done))

    def bb100(key: str) -> float | None:
        if not done:
            return None
        return round(100.0 * sum(h.get(key) or 0.0 for h in done) / len(done), 3)

    n_dec = n_agree = 0
    n_solver_anchor = 0
    for h in hands:
        for d in h.get("hero_decisions") or []:
            n_dec += 1
            adv = d.get("advisor") or {}
            prov_src = str((adv.get("provenance") or {}).get("source") or "")
            if prov_src in ("solver_hit", "preflop_chart"):
                n_solver_anchor += 1
            if adv.get("available") and adv.get("action") == d.get("action"):
                n_agree += 1
    return {
        "file": str(path),
        "n_hands": len(hands),
        "n_hands_completed": len(done),
        "bb_per_100_rakefree": bb100("net_bb"),
        "bb_per_100_net": bb100("net_bb_raked"),
        "bb_per_100_net_fullrake": bb100("net_bb_raked_full"),
        "hero_decisions": n_dec,
        "advisor_available": sum(
            1 for h in hands for d in (h.get("hero_decisions") or [])
            if (d.get("advisor") or {}).get("available")),
        "advisor_agreement": (round(n_agree / n_dec, 4) if n_dec else None),
        "advisor_solver_anchored_share": (round(n_solver_anchor / n_dec, 4)
                                          if n_dec else None),
        "generated_at": _utc_now(),
    }


def _utc_now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")
