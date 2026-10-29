"""Fold Equity v1 (P42-C2): 确定性弃牌权益估计 —— range × 条件行为统计。

模型（全部显式声明，禁止伪造）：
  villain 面对下注/加注时的继续概率由两条独立证据通道融合：

  1) 统计通道（真实行为数据）：
     metric 按 facing 映射 —— hero bet → fold_to_cbet（任何街，注明口径
     局限）；hero raise(preflop) → fold_to_3bet；hero raise(postflop) →
     无对应统计（诚实跳过）。样本数 ≥ min_samples 才可用。

  2) 范围结构通道（确定性数学）：
     对手"按底池赔率理性继续"的组合权重占比。对每个 villain 范围组合
     （去冲突后）以 seeded 共享采样评估 vs hero 手牌的 equity（每组合
     runouts 个 runout，river 精确一手牌），得到 eq_c；
     req_v = s_v / (P + 2·s_v)（s_v = villain 需再跟注额），
     f_continue = Σ w_c·[eq_c ≥ req_v] / Σ w_c，p_range_fold = 1 − f_continue。
     尺寸条件化由此通道天然成立（下注越大 → req_v 越高 → 继续越少）。

  融合（经验贝叶斯收缩，与 opponent 画像同思路）：
     w_stat = n / (n + N0)（N0=100 平滑常数）
     p_continue = w_stat·(1−p_stat_fold) + (1−w_stat)·f_continue
     p_fold = 1 − p_continue（最终 clip 到 [0.01, 0.99]，声明于 provenance）

  仅统计可用 → 纯统计（provenance 标明"未按尺寸调整"）；仅范围可用 →
  纯结构（provenance 标明"理性继续假设"）；两者皆缺 → UNKNOWN（不伪造）。

  确定性：相同输入 → 相同输出（固定种子、固定迭代顺序、无全局随机）。
  纯数学：无 LLM、无网络、无 IO。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .cards import Card, parse_card
from .ranges import parse_range
from .equity import evaluate7, remaining_deck  # noqa: F401  (evaluate7 re-export)
from .conditional_model import ConditionalFoldIndex, size_bucket, texture_tag

_N0 = 100.0          # 统计通道平滑常数（经验贝叶斯收缩）
_FOLD_CLIP = (0.01, 0.99)


def _f(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


@dataclass
class FoldEquityResult:
    available: bool = False
    unavailable_reason: str = ""
    p_fold: float | None = None
    p_continue: float | None = None
    mode: str = "unknown"            # stat+range | stat | range | unknown
    provenance: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"available": self.available,
                "unavailable_reason": self.unavailable_reason,
                "p_fold": round(self.p_fold, 6) if self.p_fold is not None else None,
                "p_continue": round(self.p_continue, 6) if self.p_continue is not None else None,
                "mode": self.mode,
                "provenance": self.provenance}


class FoldEstimator:
    """单 villain 弃牌权益估计器：每决策实例化一次，按尺寸查询。

    组合级 equity 只计算一次（与尺寸无关），不同下注尺寸仅换 req_v 阈值。
    """

    def __init__(self, pot_bb: float, facing: str, street: str,
                 villain_dist: dict | None = None,
                 behavior: dict | None = None,
                 hero_cards: list[str] | None = None,
                 board: list[str] | None = None,
                 seed: int = 42, runouts: int = 12,
                 min_samples: int = 30, n0: float = _N0) -> None:
        self.pot_bb = float(pot_bb)
        self.facing = str(facing)                 # "bet" | "raise"
        self.street = str(street)
        self.villain_dist = villain_dist or {}
        self.behavior = behavior or {}
        self.hero_cards = [str(c) for c in (hero_cards or [])]
        self.board = [str(c) for c in (board or [])]
        self.seed = int(seed)
        self.runouts = int(runouts)
        self.min_samples = int(min_samples)
        self.n0 = float(n0)
        # P43：条件行为模型（来自 Opponent Agent 画像——真实历史加工）
        cond_p = behavior.get("conditional_model") if isinstance(behavior, dict) else None
        cond_g = behavior.get("conditional_population") if isinstance(behavior, dict) else None
        self._cond_player = ConditionalFoldIndex.from_dict(cond_p) \
            if isinstance(cond_p, dict) and cond_p.get("counts") else None
        self._cond_pop = ConditionalFoldIndex.from_dict(cond_g) \
            if isinstance(cond_g, dict) and cond_g.get("counts") else None
        self._texture = texture_tag(self.board)
        self._combo_eq: list[tuple[float, float]] | None = None   # [(eq_c, w_c)]
        self._range_error: str | None = None

    # ---- 统计通道 ----------------------------------------------------
    def _metric_name(self) -> str | None:
        if self.facing == "bet":
            return "fold_to_cbet"                 # 口径局限：cbet 统计被用于任意下注
        if self.facing == "raise" and self.street == "preflop":
            return "fold_to_3bet"
        return None                               # postflop raise：无对应统计

    def _stat_fold(self, villain_call_bb: float = 0.0
                   ) -> tuple[float | None, dict]:
        prov: dict = {"channel": "stat",
                      "metric": self._metric_name(),
                      "size_adjusted": False}
        # ---- P43 优先级 1：条件玩家模型（真实历史条件化） ----
        if self._cond_player is not None:
            bucket = size_bucket(villain_call_bb, self.pot_bb)
            est = self._cond_player.estimate(
                self.street, self.facing, bucket, texture=self._texture,
                population=self._cond_pop, min_samples=self.min_samples)
            if est is not None:
                prov.update({"source": "conditional_player",
                             "size_adjusted": True,
                             "conditional": est})
                return float(est["p_fold"]), prov
        # ---- P43 优先级 2：条件群体模型（同键 population 条件率） ----
        if self._cond_pop is not None:
            bucket = size_bucket(villain_call_bb, self.pot_bb)
            est = self._cond_pop.estimate(
                self.street, self.facing, bucket, texture=self._texture,
                min_samples=self.min_samples)
            if est is not None:
                prov.update({"source": "conditional_population",
                             "size_adjusted": True,
                             "conditional": est})
                return float(est["p_fold"]), prov
        # ---- P43 优先级 3：全局统计（P42 语义，fallback 保留） ----
        prov["source"] = "global"
        metric = self._metric_name()
        if metric is None:
            prov["reason"] = f"no behavior metric for facing={self.facing} postflop"
            return None, prov
        stats = (self.behavior.get("behavior_statistics")
                 if isinstance(self.behavior, dict) else None) or {}
        m = stats.get(metric)
        if not isinstance(m, dict):
            prov["reason"] = f"no behavior stat {metric}"
            return None, prov
        n = int(_f(m.get("samples")) or 0)
        val = _f(m.get("value"))
        prov.update({"samples": n, "raw_value": val})
        if val is None or not (0.0 <= val <= 1.0):
            prov["reason"] = "stat value invalid"
            return None, prov
        if n < self.min_samples:
            prov["reason"] = f"samples {n} < {self.min_samples}"
            return None, prov
        return min(max(val, 0.0), 1.0), prov

    # ---- 范围结构通道 --------------------------------------------------
    def _combo_equities(self) -> list[tuple[float, float]] | None:
        """每范围组合 vs hero 的 equity（seeded；river 用现成牌面精确）。"""
        if self._combo_eq is not None:
            return self._combo_eq
        if not self.villain_dist or len(self.hero_cards) != 2:
            self._range_error = "requires villain distribution and hero hole cards"
            return None
        try:
            hero_c = [parse_card(c) for c in self.hero_cards]
            board_c = [parse_card(c) for c in self.board]
        except ValueError as e:
            self._range_error = f"invalid card: {e}"
            return None
        dead = {c.code for c in hero_c + board_c}
        entries: list[tuple[tuple, float]] = []
        for cat, w in self.villain_dist.items():
            try:
                sub = parse_range(str(cat))
            except ValueError:
                continue
            fw = _f(w)
            if fw is None or fw <= 0:
                continue
            for (hi, lo, suited), _ in sub.weights.items():
                entries.append(((hi, lo, suited), fw))
        if not entries:
            self._range_error = "villain distribution unparseable/empty"
            return None
        needed = 5 - len(board_c)
        if needed < 0:
            self._range_error = "board exceeds 5 cards"
            return None
        rng = random.Random(self.seed)
        hs_cache = evaluate7(hero_c + board_c) if needed == 0 else None
        out: list[tuple[float, float]] = []
        for (hi, lo, suited), w in entries:
            # 具体组合（与 hero/board 去冲突；全冲突则跳过该组合）
            if hi == lo:
                pairs = [(s1, s2) for i, s1 in enumerate(range(4))
                         for s2 in range(i + 1, 4)]
            elif suited:
                pairs = [(s, s) for s in range(4)]
            else:
                pairs = [(s1, s2) for s1 in range(4) for s2 in range(4)
                         if s1 != s2]

            def _cards(s1: int, s2: int) -> tuple[Card, Card]:
                return (Card((hi - 2) * 4 + s1), Card((lo - 2) * 4 + s2))

            live = [(a, b) for a, b in pairs
                    if _cards(a, b)[0].code not in dead
                    and _cards(a, b)[1].code not in dead]
            if not live:
                continue
            wins = ties = 0
            for _ in range(self.runouts):
                if needed == 0:
                    c1, c2 = _cards(*live[rng.randrange(len(live))])
                    vs = evaluate7([c1, c2] + board_c)
                    hs = hs_cache
                else:
                    c1, c2 = _cards(*live[rng.randrange(len(live))])
                    rem = remaining_deck(hero_c + board_c + [c1, c2])
                    full = board_c + rng.sample(rem, needed)
                    hs = evaluate7(hero_c + full)
                    vs = evaluate7([c1, c2] + full)
                if vs > hs:
                    wins += 1                    # combo 赢 → eq_c 为组合视角
                elif vs == hs:
                    ties += 1
            out.append(((wins + 0.5 * ties) / self.runouts, w))
        if not out:
            self._range_error = "no live combos after card removal"
            return None
        self._combo_eq = out
        return out

    def _range_continue(self, villain_call_bb: float) -> tuple[float | None, dict]:
        prov: dict = {"channel": "range", "model": "pot_odds_continue_threshold"}
        if villain_call_bb <= 0 or self.pot_bb <= 0:
            prov["reason"] = "non-positive pot or call size"
            return None, prov
        combos = self._combo_equities()
        if combos is None:
            prov["reason"] = self._range_error or "range unavailable"
            return None, prov
        req_v = villain_call_bb / (self.pot_bb + 2.0 * villain_call_bb)
        total_w = sum(w for _, w in combos)
        if total_w <= 0:
            prov["reason"] = "zero range weight"
            return None, prov
        f_continue = sum(w for eq, w in combos if eq >= req_v) / total_w
        prov.update({"required_equity_v": round(req_v, 6),
                     "f_continue": round(f_continue, 6),
                     "combos": len(combos), "runouts": self.runouts,
                     "seed": self.seed})
        return f_continue, prov

    def continuing_equity(self, villain_call_bb: float) -> tuple[float | None, dict]:
        """P56 A3-v2：villain **继续范围**（eq_c ≥ req_v）下 hero 的
        加权平均 equity（hero 视角）。

        语义：bet/raise 被跟分支的 p(win|called) 应相对"继续范围"而非
        全范围（跟注范围更强——A3 细化）。req_v 与 _range_continue 同式；
        缓存的 eq_c 为 villain 组合视角 → hero 视角 = 1 − eq_c（tie 已按
        半权重计入）。复用缓存；无范围/无组合 → (None, prov)（不伪造）。
        """
        prov: dict = {"channel": "range", "model": "continuing_range_equity"}
        if villain_call_bb <= 0 or self.pot_bb <= 0:
            prov["reason"] = "non-positive pot or call size"
            return None, prov
        combos = self._combo_equities()
        if combos is None:
            prov["reason"] = self._range_error or "range unavailable"
            return None, prov
        req_v = villain_call_bb / (self.pot_bb + 2.0 * villain_call_bb)
        w_cont = sum(w for eq, w in combos if eq >= req_v)
        if w_cont <= 0:
            prov["reason"] = "empty continuing range at req_v"
            return None, prov
        eq_cont = sum((1.0 - eq) * w for eq, w in combos
                      if eq >= req_v) / w_cont
        prov.update({"required_equity_v": round(req_v, 6),
                     "eq_continuing_hero": round(eq_cont, 6),
                     "w_continue": round(w_cont, 6),
                     "combos": len(combos)})
        return eq_cont, prov

    # ---- 融合 ----------------------------------------------------------
    def for_size(self, villain_call_bb: float) -> FoldEquityResult:
        """villain 需再跟 villain_call_bb（bet 场景 = s；raise 场景 = R−B）。"""
        res = FoldEquityResult()
        p_stat, prov_stat = self._stat_fold(villain_call_bb)
        f_cont, prov_range = self._range_continue(villain_call_bb)
        res.provenance = {"stat": prov_stat, "range": prov_range,
                          "n0": self.n0, "facing": self.facing,
                          "street": self.street,
                          "source": (str(prov_stat.get("source"))
                                     if p_stat is not None else
                                     ("range_only" if f_cont is not None
                                      else "unknown"))}
        if p_stat is None and f_cont is None:
            res.unavailable_reason = (
                prov_stat.get("reason") or prov_range.get("reason") or
                "no fold evidence channel available")
            return res
        if p_stat is not None and f_cont is not None:
            # P50-1（F1 修复）：P43 conditional 路径的 samples 嵌套在
            # prov["conditional"]["samples"]，P42 全局统计路径在顶层——
            # 两种 provenance 形态都合法；此前只读顶层导致 conditional
            # 证据 w_stat 恒为 0（被静默丢弃）。融合公式/N0 不变。
            n = float(prov_stat.get("samples")
                      or (prov_stat.get("conditional") or {}).get("samples")
                      or 0)
            w_stat = n / (n + self.n0)
            p_continue = w_stat * (1.0 - p_stat) + (1.0 - w_stat) * f_cont
            res.mode = "stat+range"
        elif p_stat is not None:
            w_stat = 1.0
            p_continue = 1.0 - p_stat
            res.mode = "stat"
            res.warnings.append({"code": "STAT_ONLY",
                                 "message": "fold stat not size/range adjusted"})
        else:
            w_stat = 0.0
            p_continue = f_cont
            res.mode = "range"
            res.warnings.append({"code": "RANGE_ONLY",
                                 "message": "rational-continue assumption (no behavior stat)"})
        res.provenance["w_stat"] = round(w_stat, 6)
        p_fold = min(max(1.0 - p_continue, _FOLD_CLIP[0]), _FOLD_CLIP[1])
        res.available = True
        res.p_fold = p_fold
        res.p_continue = 1.0 - p_fold
        return res
