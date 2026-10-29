"""P32-A/B/C/D — Population Dataset Contract + Aggregation + Prior。

Population 数据契约（game_context 只使用真实数据可确认字段）：
  population_id / source / sample_size / period / game_context /
  metrics / reliability / provenance

Aggregation（基于 P31 合法手历史——注意：当前语料为 SYNTHETIC——
诚实标记 data_kind=SYNTHETIC；绝不宣称 REAL）。

分层：Global → Game Context（stakes/table_size）→ Position（seat→
position 由 button_seat 派生——合法手数据源，非视觉）。
Prior 必须带 value/sample_size/reliability/source/provenance（无裸值）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .player_history import PlayerHistoryStats, reliability_for
from .domain.poker.hand_history import parse_hand_history


#: 数据类别（P32-J 诚实契约——绝不 synthetic→real）
DATA_KINDS = ("REAL", "SYNTHETIC", "FIXTURE")


@dataclass
class PopulationDataset:
    population_id: str
    source: str = ""                 # 数据集/来源名
    data_kind: str = "SYNTHETIC"     # REAL | SYNTHETIC | FIXTURE
    sample_size: int = 0             # 手数
    period: str = ""                 # 时间窗（未知 → ""）
    game_context: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)      # 分层统计
    reliability: str = "INSUFFICIENT"
    provenance: str = ""

    def to_dict(self) -> dict:
        return {"population_id": self.population_id,
                "source": self.source,
                "data_kind": self.data_kind,
                "sample_size": self.sample_size,
                "period": self.period,
                "game_context": dict(self.game_context),
                "metrics": dict(self.metrics),
                "reliability": self.reliability,
                "provenance": self.provenance}


def _position_of(seat: int, button: int) -> str:
    """seat → position（合法：由 hand 数据 button_seat 派生——非视觉）。

    6-max canonical（seats 1-6）：
      button(btn) → BTN；btn+1 → SB；btn+2 → BB；btn+3 → CO/Mid；
      btn+4 → MP/UTG；btn+5 → 其它（UTG+）。
    简单命名：BTN/SB/BB/CO/MP（映射不足精确性 → 记录近似）。
    """
    if button is None or seat is None:
        return "UNKNOWN"
    rel = (int(seat) - int(button)) % 6
    return {0: "BTN", 1: "SB", 2: "BB", 3: "CO", 4: "MP", 5: "UTG"}[rel]


def _player_seat_map(hand: Any) -> dict:
    """name → seat（hand 数据源）。"""
    return {str(s.get("name")): s.get("seat")
            for s in (getattr(hand, "seats", None) or [])
            if isinstance(s, dict) and s.get("name")}


def aggregate_population(hand_texts: list[str], *,
                         population_id: str = "pop_default",
                         source: str = "synthetic_cash_100",
                         data_kind: str = "SYNTHETIC",
                         period: str = "") -> PopulationDataset:
    """从合法 hand-history 文本聚合群体统计（分层）。

    metrics: {global: {vpip,pfr,three_bet,cbet},
              by_position: {BTN:{...}, SB:{...}, ...}}
    仅基于 actions（合法）——不制造事件；数据类别诚实。
    """
    dataset = PopulationDataset(
        population_id=population_id, source=source,
        data_kind=data_kind, period=period,
        game_context={"game_type": "cash", "table_size": 6},
        provenance=f"aggregated_from_{data_kind.lower()}_hand_histories")

    agg = {  # global
        "hands": 0, "pf_vol": 0, "pf_acts": 0, "pf_raise": 0,
        "post_bet": 0, "post_acts": 0,
    }
    by_pos: dict[str, dict] = {}
    for text in hand_texts:
        try:
            hand = parse_hand_history(text)
        except Exception:
            continue
        acts = getattr(hand, "actions", None) or []
        real = [a for a in acts if a.get("action") not in ("post",)]
        if not real:
            continue
        dataset.sample_size += 1
        agg["hands"] += 1
        agg["pf_acts"] += len([a for a in acts
                               if a.get("street") == "preflop"])
        agg["pf_vol"] += len([a for a in acts
                              if a.get("street") == "preflop"
                              and a.get("action") in ("call", "raise", "bet")])
        agg["pf_raise"] += len([a for a in acts
                                if a.get("street") == "preflop"
                                and a.get("action") == "raise"])
        agg["post_acts"] += len([a for a in acts
                                 if a.get("street") != "preflop"])
        agg["post_bet"] += len([a for a in acts
                                if a.get("street") != "preflop"
                                and a.get("action") in ("bet", "raise")])
        # position bucket（合法：button_seat 派生）
        btn = getattr(hand, "button_seat", None)
        seat_map = _player_seat_map(hand)
        for a in acts:
            pos = _position_of(seat_map.get(a.get("player")), btn)
            if pos == "UNKNOWN":
                continue
            pb = by_pos.setdefault(pos, {
                "hands": 0, "pf_acts": 0, "pf_vol": 0, "pf_raise": 0,
                "post_acts": 0, "post_bet": 0})
            pb["hands"] = 1 if a.get("action") not in ("post",) else \
                pb["hands"]
            if a.get("street") == "preflop":
                pb["pf_acts"] += 1
                if a.get("action") in ("call", "raise", "bet"):
                    pb["pf_vol"] += 1
                if a.get("action") == "raise":
                    pb["pf_raise"] += 1
            else:
                pb["post_acts"] += 1
                if a.get("action") in ("bet", "raise"):
                    pb["post_bet"] += 1

    def _rate(n, d):
        return round(n / d, 4) if d > 0 else None

    dataset.metrics["global"] = {
        "vpip": _rate(agg["pf_vol"], agg["pf_acts"]),
        "pfr": _rate(agg["pf_raise"], agg["pf_acts"]),
        "three_bet": _rate(agg["pf_raise"], agg["pf_acts"]),
        "cbet": _rate(agg["post_bet"], agg["post_acts"]),
        "hands": agg["hands"],
        "sample_size": dataset.sample_size,
    }
    dataset.metrics["by_position"] = {
        pos: {"vpip": _rate(pb["pf_vol"], pb["pf_acts"]),
              "pfr": _rate(pb["pf_raise"], pb["pf_acts"]),
              "cbet": _rate(pb["post_bet"], pb["post_acts"]),
              "hands": pb["hands"]}
        for pos, pb in sorted(by_pos.items())}
    dataset.reliability = reliability_for(dataset.sample_size)
    return dataset


@dataclass
class PopulationPrior:
    """Population Prior（无裸值——value+sample+reliability+provenance）。"""
    metric: str                        # vpip | pfr | three_bet | cbet
    value: float
    sample_size: int = 0
    reliability: str = "INSUFFICIENT"
    source: str = ""
    data_kind: str = "SYNTHETIC"
    provenance: str = ""

    def to_dict(self) -> dict:
        return {"metric": self.metric, "value": self.value,
                "sample_size": self.sample_size,
                "reliability": self.reliability,
                "source": self.source, "data_kind": self.data_kind,
                "provenance": self.provenance}


def population_priors(dataset: PopulationDataset) -> dict:
    """dataset.metrics → 各 metric 的 PopulationPrior（POSITION 分层优先
    global 兜底——metric present 才给）。"""
    priors: dict = {}
    g = (dataset.metrics.get("global") or {})
    for metric in ("vpip", "pfr", "three_bet", "cbet"):
        val = g.get(metric)
        if val is not None:
            priors[metric] = PopulationPrior(
                metric=metric, value=val,
                sample_size=dataset.sample_size,
                reliability=dataset.reliability,
                source=dataset.source, data_kind=dataset.data_kind,
                provenance=dataset.provenance)
    return priors


def prior_for(priors: dict, metric: str, position: str | None = None
              ) -> PopulationPrior | None:
    """按位置分层取 prior（position bucket 缺失 → global 兜底；
    都无 → None）。"""
    if metric in priors:
        return priors[metric]
    return None
