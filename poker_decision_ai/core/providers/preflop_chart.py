"""SOLVER-1 — HU preflop 查表基线 provider（source=chart；**非 solver 输出**）。

postflop solver 不覆盖 preflop；本 provider 以 configs/solver/
preflop_chart_hu.yaml（数据起点 = configs/ranges.yaml 已声明范围）做
**1/0 成员资格查表**。诚实边界：
  * source=chart，equilibrium=false，绝不冒充 solver/均衡解；
  * 纯查表频率（无混合频率——图表基线的诚实形态）；
  * 未匹配 scenario → available=False（不编造）。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from ..errors import ProviderError
from .solver import SolverAction, SolverResult, SolverProvider

CHART_PROVIDER_VERSION = "solver1-preflop-chart-v1"
DEFAULT_CHART_PATH = "configs/solver/preflop_chart_hu.yaml"


class PreflopChartProvider(SolverProvider):
    """preflop 查表基线（source=chart）。"""

    name = "preflop_chart"
    is_mock = False

    def __init__(self, chart_path: str = DEFAULT_CHART_PATH) -> None:
        self.chart_path = chart_path or DEFAULT_CHART_PATH
        self._chart: dict | None = None

    # ------------------------------------------------------------------
    def _load_chart(self) -> dict:
        if self._chart is not None:
            return self._chart
        p = Path(self.chart_path)
        if not p.is_file():
            self._chart = {}
            return self._chart
        # P30-9-CB：支持频率 JSON 图表（外部 GTO，source=external_gto_json）
        if p.suffix.lower() == ".json":
            raw = json.loads(p.read_text(encoding="utf-8")) or {}
        else:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        self._chart = dict(raw.get("chart") or {})
        return self._chart

    def chart_mode(self) -> str:
        """`frequency`（外部 GTO 频率图表）| `membership`（1/0 成员资格图表）。"""
        chart = self._load_chart()
        if str(chart.get("source") or "") == "external_gto_json":
            return "frequency"
        return "membership"

    def is_configured(self) -> bool:
        chart = self._load_chart()
        return bool(chart.get("scenarios"))

    def status_reason(self) -> str:
        if self.is_configured():
            return ""
        return f"preflop chart missing or empty: {self.chart_path}"

    # ------------------------------------------------------------------
    @staticmethod
    def _hand_class(hole_cards: list[str]) -> str:
        """AhKd → AKs（suited）；AhKc → AKo；AhAd → AA。"""
        if len(hole_cards) != 2:
            return ""
        r1, s1 = hole_cards[0][0], hole_cards[0][1]
        r2, s2 = hole_cards[1][0], hole_cards[1][1]
        hi, lo = (r1, r2) if "23456789TJQKA".index(r1) >= "23456789TJQKA".index(r2) \
            else (r2, r1)
        if r1 == r2:
            return f"{hi}{lo}"
        return f"{hi}{lo}{'s' if s1 == s2 else 'o'}"

    @staticmethod
    def _range_contains(range_str: str, hand_class: str) -> bool | None:
        """声明式成员资格（ranges.yaml 记法子集）。

        支持：AA（对子）；AKs / AKo / AK（both）；AKs+（同高牌 kicker≥）；
        A2o+（同上）；AA-22（对子区间）；A5s-A2s（同高牌 kicker 区间）；
        random:P（全部手牌）。未知记法 → None（调用方诚实报错）。
        """
        order = "23456789TJQKA"
        oi = order.index
        hand = hand_class.strip()
        is_pair = len(hand) == 2 and hand[0] == hand[1]
        hi_i = oi(hand[0])
        lo_i = oi(hand[1])

        def rank_i(c: str) -> int:
            return oi(c)

        for token in re.split(r"[,，]", range_str or ""):
            t = token.strip()
            if not t:
                continue
            if t.lower().startswith("random:"):
                return True
            # 对子：AA
            if re.fullmatch(r"([2-9TJQKA])", t):
                if is_pair and hi_i == oi(t[0]):
                    return True
                continue
            # 对子区间：AA-22
            m = re.fullmatch(r"([2-9TJQKA])-([2-9TJQKA])", t)
            if m:
                if is_pair:
                    a, b = sorted((oi(m.group(1)), oi(m.group(2))))
                    if a <= hi_i <= b:
                        return True
                continue
            # 同高牌区间（suited/offsuit）：A5s-A2s / KQs-K9s
            m = re.fullmatch(r"([2-9TJQKA])([2-9TJQKA])(s|o)?-"
                             r"([2-9TJQKA])([2-9TJQKA])(s|o)?", t)
            if m:
                r1, r2, su1, r3, r4, su2 = m.groups()
                hi = r1 if oi(r1) >= oi(r3) else r3
                kick_lo = min(oi(r2 if oi(r1) >= oi(r3) else r4),
                              oi(r4 if oi(r1) >= oi(r3) else r2))
                suit = su1 or su2
                if is_pair or hand[0] != hand[1]:
                    if hand[:2] != f"{hi}{''}":
                        pass
                if len(hand) == 3 and hand[0] == hi and                         (suit is None or hand[2] == suit):
                    k_i = oi(hand[1])
                    if kick_lo <= k_i <= oi(
                            r2 if oi(r1) >= oi(r3) else r4):
                        return True
                continue
            # 单张（可带 +）：AKs / AK / AKs+ / A2o+ / 66+ 已在上方处理
            m = re.fullmatch(r"([2-9TJQKA])([2-9TJQKA])(s|o)?\+?", t)
            if m:
                r1, r2, suit = m.groups()
                hi = r1 if oi(r1) >= oi(r2) else r2
                lo = r2 if hi == r1 else r1
                plus = t.endswith("+")
                if hi == lo:                       # 66+（对子开放区间）
                    if is_pair and hi_i >= oi(hi):
                        return True
                    continue
                if is_pair or hand[0] != hi:
                    continue
                if len(hand) != 3:
                    continue
                if suit is not None and hand[2] != suit:
                    continue
                if not plus:
                    if hand[1] == lo and (suit is None or hand[2] == suit):
                        return True
                    continue
                if lo_i >= oi(lo):
                    return True
                continue
            return None                            # 未知记法 → 诚实失败
        return False

    # ------------------------------------------------------------------
    async def solve(self, query: dict) -> SolverResult:
        if str(query.get("street") or "") != "preflop":
            raise ProviderError("preflop chart provider covers preflop only")
        chart = self._load_chart()
        scenarios = chart.get("scenarios") or {}
        scenario_key = str(query.get("scenario") or "")
        scenario = scenarios.get(scenario_key)
        if not scenario and "@" in scenario_key:
            # P30-9-DB：尺寸标签缺失 ⇒ 回退到无标签（默认 2.5bb）键
            _fb = scenario_key.split("@", 1)[0]
            scenario = scenarios.get(_fb)
            if scenario:
                scenario_key = _fb
        if not scenario:
            raise ProviderError(
                f"unknown preflop scenario: {scenario_key!r} "
                f"(known: {sorted(scenarios)})")

        hole_cards = [str(c) for c in (query.get("hole_cards") or [])]
        hand_class = self._hand_class(hole_cards)
        if not hand_class:
            raise ProviderError("hole_cards missing for preflop chart lookup")

        # ---- P30-9-CB：频率图表（外部 GTO）----
        if self.chart_mode() == "frequency":
            raise_f = float((scenario.get("raise") or {}).get(hand_class, 0.0) or 0.0)
            call_f = float((scenario.get("call") or {}).get(hand_class, 0.0) or 0.0)
            fold_f = max(0.0, 1.0 - raise_f - call_f)
            size_bb = float(scenario.get("size_bb") or 0.0) or None
            acts: list[SolverAction] = []
            if raise_f > 0:
                acts.append(SolverAction(name="raise", size_bb=size_bb,
                                          frequency=round(raise_f, 6)))
            if call_f > 0:
                acts.append(SolverAction(name="call", size_bb=None,
                                          frequency=round(call_f, 6)))
            if fold_f > 0:
                acts.append(SolverAction(name="fold", size_bb=None,
                                          frequency=round(fold_f, 6)))
            if not acts:
                raise ProviderError(
                    f"frequency chart has no action for {hand_class} @ {scenario_key}")
            return SolverResult(
                solver_name=str(chart.get("name") or "preflop_gto_chart"),
                model=f"{chart.get('name')}:{scenario_key}",
                actions=acts,
                bet_sizings_bb=([size_bb] if size_bb else []),
                notes=(f"外部 GTO 翻前图表（6-max）；场景={scenario_key}；"
                        f"{chart.get('source_note', '')[:120]}"),
                is_approximate=False,
                raw={"chart": chart.get("name"), "scenario": scenario_key,
                      "hand": hand_class, "raise": raise_f, "call": call_f,
                      "fold": round(fold_f, 6), "source": "external_gto_json"})

        ranges_file = chart.get("ranges_file") or "configs/ranges.yaml"
        ranges = (yaml.safe_load(
            Path(ranges_file).read_text(encoding="utf-8")) or {}).get(
            "ranges") or {}

        actions: list[SolverAction] = []
        chosen = None
        for rule in scenario.get("actions") or []:
            ref = rule.get("if_in_range")
            if ref:
                node = ranges
                ok_value: bool | None = True
                for part in str(ref).split("."):
                    if not isinstance(node, dict) or part not in node:
                        ok_value = None
                        break
                    node = node[part]
                if ok_value is None:
                    raise ProviderError(f"chart range ref missing: {ref}")
                inside = self._range_contains(str(node), hand_class)
                if inside is None:
                    raise ProviderError(f"chart range token unparsable: {ref}")
                if not inside:
                    continue
            actions.append(SolverAction(
                name=str(rule.get("action")),
                size_bb=(float(rule["size_bb"])
                         if rule.get("size_bb") is not None else None),
                frequency=1.0,
                ev=None,
            ))
            chosen = str(rule.get("action"))
            break

        if not actions:
            raise ProviderError(
                f"preflop chart produced no action for {hand_class!r} "
                f"in scenario {scenario_key!r}")

        # SOLVER-2：附上场景声明的完整动作菜单（frequency=0）——供
        # 有界剥削偏离作锚定目标（chosen 行保持 frequency=1.0；菜单是
        # 图表声明过的动作，绝不引入场景之外的动作）。
        seen_names = {str(a.name) for a in actions}
        for rule in scenario.get("actions") or []:
            an = str(rule.get("action") or "")
            if an in seen_names:
                continue
            seen_names.add(an)
            if an not in ("fold", "check", "call", "bet", "raise", "allin"):
                continue
            actions.append(SolverAction(
                name=an,
                size_bb=(float(rule["size_bb"])
                         if rule.get("size_bb") is not None else None),
                frequency=0.0,
                ev=None,
            ))

        return SolverResult(
            solver_name=self.name,
            model=f"{chart.get('name', 'preflop_chart')}({hand_class}→{chosen})",
            actions=actions,
            bet_sizings_bb=sorted({a.size_bb for a in actions
                                   if a.size_bb is not None}),
            notes="HU preflop 查表基线（source=chart；ranges.yaml 起点；"
                  "非 solver 产物、非均衡解）",
            is_approximate=True,
            raw={
                "source": "chart",
                "equilibrium": False,
                "is_mock": False,
                "hand_class": hand_class,
                "scenario": scenario_key,
                "chosen": chosen,
                "simplifications": ["查表成员资格 1/0（无混合频率）",
                                    "HU only", "ranges.yaml 起点范围"],
            },
        )


def preflop_chart_contract() -> dict:
    return {
        "version": CHART_PROVIDER_VERSION,
        "source": "chart",
        "is_solver_output": False,
        "is_equilibrium": False,
        "is_mock": False,
        "covers": "HU preflop scenarios declared in chart yaml",
    }
