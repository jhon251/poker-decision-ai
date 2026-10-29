"""P43 — Conditional Opponent Model（纯数学；仅真实历史；禁止伪造）。

目标：把 PlayerHistoryStore/repos.hands 中的真实手牌（parse_hand_history
产出的 parsed 动作级数据）加工为条件行为概率：

    P(fold | street, facing, size_bucket, position, texture)

数据来源（唯一合法源）：repos.hands 中携带 "parsed" 键的手牌记录
（DataImporter.import_hand_text 写入）。归一化 per-hand 布尔记录不含
street/size/texture 条件 → 不参与条件索引（诚实缺失，不补数据）。

事件抽取（确定性回放）：
  逐动作重放 parsed.actions，推导每个决策点的：
    street        preflop/flop/turn/river
    facing        "none"（无人下注）| "bet"（本街首次面对下注）| "raise"（面对加注）
    size_bucket   facing 时 to_call/pot_before 分桶（<=0.5 / <=1.0 / <=2.0 / >2.0）
    position      positions_for(seats, button_seat) 派生（button 缺失 → ""）
    texture       该街公共牌 texture_tag（preflop → "preflop"）
    outcome       fold=1（弃牌）/ fold=0（call/bet/raise/check 继续）

估计（确定性、可解释；复用既有 bayes_shrink Beta-Binomial 收缩）：
  player 条件索引与 population 条件索引分开累计；
  后验 = bayes_shrink(rate, n, prior=population 同键率或 0.5, strength=200)；
  样本门槛 MIN_SAMPLES=30（契约，禁止为准确度调低）；
  退化阶梯（条件越细数据越不足时逐级回粗，全部 <MIN_SAMPLES → UNKNOWN）：
    (street,facing,bucket,position,texture)
      → 去 texture → 去 position → 去 bucket → (street,facing) → UNKNOWN
  全程无随机；同一输入 → 同一输出。
"""
from __future__ import annotations

from typing import Any

from .stats import bayes_shrink
from .models import positions_for

#: 样本门槛（contract——与 P42 fold 统计通道一致；禁止为准确度降低）
MIN_SAMPLES = 30
#: Beta-Binomial 先验伪计数（复用既有 shrinkage 契约）
PRIOR_STRENGTH = 200.0
#: 中性先验（无 population 同键估计时）
NEUTRAL_PRIOR = 0.5
#: facing 大小分桶（to_call / pot_before——底池赔率代理）
SIZE_BUCKETS = ("s<=0.5", "m<=1.0", "l<=2.0", "x>2.0")


def size_bucket(to_call: float, pot_before: float) -> str | None:
    """面对下注/加注的尺寸分桶（确定性；pot<=0 或 to_call<0 → None）。"""
    if pot_before is None or pot_before <= 0 or to_call is None or to_call < 0:
        return None
    r = to_call / pot_before
    if r <= 0.5:
        return "s<=0.5"
    if r <= 1.0:
        return "m<=1.0"
    if r <= 2.0:
        return "l<=2.0"
    return "x>2.0"


def texture_tag(board: list) -> str:
    """基础 board texture（确定性；空 → "preflop"）。

    优先级 paired > monotone > connected > wet > dry（声明式简单纹理）。
    """
    ranks: list[int] = []
    suits: list[int] = []
    for c in board or []:
        try:
            from .cards import parse_card
            card = c if isinstance(c, object) and hasattr(c, "rank") else parse_card(str(c))
            ranks.append(card.rank)
            suits.append(card.suit)
        except ValueError:
            continue
    if not ranks:
        return "preflop"
    if len(set(ranks)) < len(ranks):
        return "paired"
    sc = max(suits.count(s) for s in set(suits)) if suits else 0
    if sc >= 3:
        return "monotone"
    rs = sorted(set(ranks))
    if any(b - a == 1 for a, b in zip(rs, rs[1:])):
        return "wet"                       # 相连（连牌面）
    if any(b - a == 2 for a, b in zip(rs, rs[1:])):
        return "connected"                 # 单隔张（gutshot 空间）
    return "dry"


def extract_events(parsed: dict) -> list[dict]:
    """parsed 手牌 → 条件事件列表（确定性重放；不合法输入 → []）。

    parsed 形如 ParsedHand.to_dict()：{hand_id, seats, button_seat,
    actions: [{street, player, action, amount, to, all_in}], street_boards}。
    """
    if not isinstance(parsed, dict):
        return []
    actions = [a for a in (parsed.get("actions") or [])
               if isinstance(a, dict) and a.get("action")]
    if not actions:
        return []
    seats = parsed.get("seats") or []
    seat_of = {str(s.get("name")): s.get("seat") for s in seats
               if isinstance(s, dict) and s.get("name")}
    btn = parsed.get("button_seat")
    try:
        pos_map = positions_for(
            [{"seat": s.get("seat")} for s in seats if isinstance(s, dict)],
            btn) if btn is not None else {}
    except Exception:
        pos_map = {}
    boards = parsed.get("street_boards") or {}
    hid = str(parsed.get("hand_id") or "")

    contributed: dict[str, float] = {}
    level = 0.0
    aggression_n = 0
    folded: set[str] = set()
    cur_street = "preflop"
    events: list[dict] = []
    for a in actions:
        street = str(a.get("street") or cur_street)
        player = str(a.get("player") or "")
        act = str(a.get("action") or "")
        if street != cur_street:
            cur_street = street
            level = 0.0
            aggression_n = 0
        if act == "post":
            contributed[player] = contributed.get(player, 0.0) + float(a.get("amount") or 0.0)
            level = max(level, contributed.get(player, 0.0))
            continue
        if act not in ("fold", "call", "bet", "raise", "check"):
            continue
        if player in folded or not player:
            continue
        amount = float(a.get("amount") or 0.0)
        to_amt = a.get("to")
        pot_before = sum(contributed.values())
        my = contributed.get(player, 0.0)
        to_call = max(0.0, level - my)
        facing = "none"
        if to_call > 1e-9:
            # 本街第 1 个激进行为 → facing "bet"；第 ≥2 个 → "raise"
            facing = "raise" if aggression_n >= 2 else "bet"
        bucket = (size_bucket(to_call, pot_before)
                  if facing != "none" else None)
        board = boards.get(street) or ([] if street == "preflop"
                                       else boards.get("flop") or [])
        events.append({
            "player": player,
            "street": street,
            "facing": facing,
            "size_bucket": bucket,
            "position": str(pos_map.get(seat_of.get(player)) or ""),
            "texture": texture_tag(board) if street != "preflop" else "preflop",
            "action": act,
            "fold": 1 if act == "fold" else 0,
            "hand_id": hid,
        })
        # 状态推进
        if act == "fold":
            folded.add(player)
        elif act == "call":
            contributed[player] = my + amount
        elif act in ("bet", "raise"):
            contributed[player] = (float(to_amt) if to_amt is not None
                                   else my + amount)
            level = max(level, contributed.get(player, 0.0))
            aggression_n += 1
    return events


class ConditionalFoldIndex:
    """条件 fold 计数索引（纯 dict，可序列化；确定性查询+退化阶梯）。

    每个事件同时计入 4 个粗化层级（full / 去 texture / 去 texture+position /
    去 texture+position+bucket）——退化阶梯查询才能命中粗键。
    """

    #: 粗化阶梯（与 estimate() 的 ladders 一致）
    _LADDERS: tuple[tuple[str, ...], ...] = (
        (), ("texture",), ("texture", "position"),
        ("texture", "position", "size_bucket"),
    )

    def __init__(self) -> None:
        self._counts: dict[str, list[int]] = {}   # key → [folds, n]
        self._n_events = 0

    @staticmethod
    def _key(ev: dict, drop: tuple[str, ...] = ()) -> str:
        parts = [ev["street"], ev["facing"],
                 "" if "size_bucket" in drop else (ev.get("size_bucket") or ""),
                 "" if "position" in drop else (ev.get("position") or ""),
                 "" if "texture" in drop else (ev.get("texture") or "")]
        return "|".join(str(p) for p in parts)

    def add_events(self, events: list[dict]) -> None:
        for ev in events or []:
            if ev.get("facing") == "none":
                continue                  # 无弃牌决策 → 不进 fold 索引
            for drop in self._LADDERS:
                k = self._key(ev, drop)
                c = self._counts.setdefault(k, [0, 0])
                c[0] += int(ev.get("fold") or 0)
                c[1] += 1
            self._n_events += 1

    @property
    def total_events(self) -> int:
        return self._n_events

    def lookup(self, key: str) -> list[int] | None:
        return self._counts.get(key)

    def estimate(self, street: str, facing: str, size_bucket_s: str | None,
                 position: str | None = None, texture: str | None = None,
                 population: "ConditionalFoldIndex | None" = None,
                 min_samples: int = MIN_SAMPLES
                 ) -> dict | None:
        """条件 fold 概率（退化阶梯 + Beta-Binomial 收缩；无数据 → None）。

        返回 {"p_fold", "samples", "raw", "confidence", "key", "degraded_to",
              "prior_rate"}；先验 = population 同键率（无则中性 0.5）。
        """
        base = {"street": street, "facing": facing,
                "size_bucket": size_bucket_s or "",
                "position": position or "",
                "texture": texture or ""}
        ev = {k: base[k] for k in
              ("street", "facing", "size_bucket", "position", "texture")}
        ladders = list(self._LADDERS)
        for drop in ladders:
            k = self._key(ev, drop)
            cnt = self._counts.get(k)
            if cnt is None or cnt[1] <= 0:
                continue
            n = cnt[1]
            if n < min_samples:
                continue
            pk = self._key(ev, drop)
            prior_rate = None
            if population is not None:
                pc = population._counts.get(pk)
                if pc is not None and pc[1] > 0:
                    prior_rate = pc[0] / pc[1]
            raw = cnt[0] / n
            p = bayes_shrink(raw, n,
                             prior_rate if prior_rate is not None
                             else NEUTRAL_PRIOR,
                             PRIOR_STRENGTH)
            return {"p_fold": round(float(p), 6),
                    "samples": n,
                    "raw": round(raw, 6),
                    "confidence": round(min(1.0, n / 100.0), 4),
                    "key": k,
                    "degraded_to": len(drop),
                    "prior_rate": (round(prior_rate, 6)
                                   if prior_rate is not None
                                   else NEUTRAL_PRIOR)}
        return None

    def to_dict(self) -> dict:
        return {"counts": {k: list(v) for k, v in sorted(self._counts.items())},
                "total_events": self._n_events}

    @classmethod
    def from_dict(cls, d: dict | None) -> "ConditionalFoldIndex":
        idx = cls()
        counts = (d or {}).get("counts") or {}
        if isinstance(counts, dict):
            for k, v in counts.items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    idx._counts[str(k)] = [int(v[0]), int(v[1])]
        te = (d or {}).get("total_events")
        idx._n_events = int(te) if isinstance(te, (int, float)) else 0
        return idx
