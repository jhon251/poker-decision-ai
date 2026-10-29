"""Human Style Agent (风格标签): soft-label players from their feature vector.

Rules (deterministic scoring, 0..1 soft scores; labels are explanatory only,
the underlying continuous feature vector stays the source of truth):

  Nit             : tightness < 0.25 and aggression < 0.3
  Maniac          : tightness > 0.75 and aggression > 0.6
  LAG             : tightness > 0.6  and aggression > 0.45
  TAG             : tightness > 0.55 and aggression > 0.4
  Calling Station : calling_tendency > 0.7
  Whale           : vpip > 0.5 with small sample (<200 hands), or clearly losing
                    (net/bb_per_100 negative)
  Recreational    : small sample with off-center stats (vpip > 0.4 or
                    aggression < 0.2)
  Regular         : larger sample with near-neutral stats
  Short Stack     : stack/bb < 20

labels = scores > 0.4; primary_label = argmax.  No opponent input -> default
"Regular" at low confidence + warning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from poker_decision_ai.core.orchestration.base_agent import BaseAgent
from poker_decision_ai.core.orchestration.context import AgentContext

_LABELS = ["Nit", "TAG", "LAG", "Maniac", "Calling Station",
           "Recreational", "Regular", "Whale", "Short Stack"]


@dataclass
class HumanStyleOutput:
    players: dict = field(default_factory=dict)   # pid -> {labels, primary_label, label_scores, is_recreational}
    confidence: float = 0.0
    evidence: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def _f(fv: dict, key: str, default: float = 0.5) -> float:
    v = fv.get(key)
    return float(v) if isinstance(v, (int, float)) else default


def _score_labels(fv: dict, stack_bb: float | None, sample_size: int,
                  vpip: float | None, net: float | None) -> dict[str, float]:
    tight = _f(fv, "tightness")
    agg = _f(fv, "aggression")
    call = _f(fv, "calling_tendency")

    scores = {label: 0.0 for label in _LABELS}

    # Nit: very tight + passive
    if tight < 0.25 and agg < 0.3:
        scores["Nit"] = 0.8
    elif tight < 0.35:
        scores["Nit"] = 0.5

    # Maniac: very loose + very aggressive
    if tight > 0.75 and agg > 0.6:
        scores["Maniac"] = 0.85
    elif tight > 0.65 and agg > 0.5:
        scores["Maniac"] = 0.5

    # LAG: loose-aggressive
    if tight > 0.6 and agg > 0.45:
        scores["LAG"] = 0.7

    # TAG: tight-aggressive
    if tight > 0.55 and agg > 0.4:
        scores["TAG"] = 0.6

    # Calling Station
    if call > 0.7:
        scores["Calling Station"] = 0.8
    elif call > 0.6:
        scores["Calling Station"] = 0.5

    # Whale: very loose + few hands, or clearly losing money
    if vpip is not None and vpip > 0.5 and sample_size < 200:
        scores["Whale"] = 0.7
    elif net is not None and net < -30:
        scores["Whale"] = 0.6

    # Recreational: few hands + off-center stats
    if sample_size < 200 and ((vpip is not None and vpip > 0.4) or agg < 0.2):
        scores["Recreational"] = 0.7
    elif sample_size < 50:
        scores["Recreational"] = 0.4

    # Regular: large, near-neutral sample
    if sample_size >= 200:
        scores["Regular"] = 0.6
    elif sample_size >= 100:
        scores["Regular"] = 0.4

    # Short Stack
    if stack_bb is not None and stack_bb < 20:
        scores["Short Stack"] = 0.9

    return scores


class HumanStyleAgent(BaseAgent):
    name: ClassVar[str] = "human_style"
    version: ClassVar[str] = "1.0.0"
    zh_name: ClassVar[str] = "风格标签 Agent"
    layer: ClassVar[str] = "human"
    depends: ClassVar[list[str]] = ["state", "opponent"]
    Input: ClassVar[type] = dict
    Output: ClassVar[type] = HumanStyleOutput

    async def applicable(self, raw_input: dict, ctx: AgentContext) -> tuple[bool, str]:
        if not raw_input.get("state"):
            return False, "state input is required for human-style labelling"
        return True, ""

    async def handle(self, inp: dict, ctx: AgentContext) -> HumanStyleOutput:
        state = inp.get("state") or {}
        opponent = inp.get("opponent") or {}
        out = HumanStyleOutput()
        warnings: list[dict] = []

        bb = float(state.get("bb") or 0.0) or 1.0
        players = state.get("players", []) or []
        opp_players = (opponent or {}).get("players") or {}

        targets = [p for p in players if isinstance(p, dict) and not p.get("is_hero")
                   and p.get("status") in ("active", "allin")]
        if not targets:
            targets = [p for p in players if isinstance(p, dict) and not p.get("is_hero")]

        if not targets:
            warnings.append({"code": "NO_VILLAINS", "message": "no non-hero players in state"})
            out.warnings = warnings
            out.confidence = 0.0
            return out

        if not opp_players:
            warnings.append({"code": "NO_OPPONENT",
                             "message": "no opponent input; defaulting to 'Regular' at low confidence"})

        for p in targets:
            pid = str(p.get("player_id", ""))
            if not pid:
                continue
            stack_bb = float(p.get("stack", 0.0) or 0.0) / bb

            opp = opp_players.get(pid) if isinstance(opp_players, dict) else None
            fv = (opp or {}).get("feature_vector") if isinstance(opp, dict) else None
            if not isinstance(fv, dict):
                fv = {}
            sample_size = int((opp or {}).get("sample_size", 0) or 0) if isinstance(opp, dict) else 0
            bs = (opp or {}).get("behavior_statistics") if isinstance(opp, dict) else {}
            bs = bs if isinstance(bs, dict) else {}

            vpip = None
            vm = bs.get("vpip")
            if isinstance(vm, dict) and vm.get("value") is not None:
                vpip = float(vm.get("value", 0.0) or 0.0)

            net = None
            bbm = bs.get("bb_per_100")
            if isinstance(bbm, dict) and bbm.get("value") is not None:
                net = float(bbm.get("value", 0.0) or 0.0)

            if not opp_players:
                # no profile at all -> conservative default
                scores = {label: 0.0 for label in _LABELS}
                scores["Regular"] = 0.3
                labels = ["Regular"]
                primary = "Regular"
                is_rec = False
                conf = 0.2
            else:
                scores = _score_labels(fv, stack_bb, sample_size, vpip, net)
                labels = [l for l in _LABELS if scores[l] > 0.4]
                primary = max(scores, key=lambda l: scores[l]) if scores else "Regular"
                is_rec = scores.get("Recreational", 0.0) > 0.4
                conf = min(1.0, 0.4 + sample_size / 500)

            out.players[pid] = {
                "labels": labels,
                "primary_label": primary,
                "label_scores": {l: round(scores[l], 3) for l in _LABELS},
                "is_recreational": is_rec,
            }

        # confidence grows with the largest opponent sample size (low without data)
        if out.players:
            if not opp_players:
                out.confidence = 0.2
            else:
                max_sample = 0
                for pid in out.players:
                    o = opp_players.get(pid) if isinstance(opp_players, dict) else None
                    if isinstance(o, dict):
                        max_sample = max(max_sample, int(o.get("sample_size", 0) or 0))
                out.confidence = round(min(1.0, 0.4 + max_sample / 500), 3)

        out.evidence = [{"kind": "human_style",
                         "text": f"labelled {len(out.players)} player(s): "
                                 + ", ".join(f"{k}={v['primary_label']}" for k, v in out.players.items()),
                         "source": "human_style", "weight": 1.0}]
        out.warnings = warnings
        return out
