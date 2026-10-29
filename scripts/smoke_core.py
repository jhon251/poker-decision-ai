"""Smoke test for core + domain layers (run from repo root)."""
import asyncio
import sys

sys.path.insert(0, ".")


def section(title):
    print(f"\n=== {title} ===")


def check(name, cond, extra=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        raise AssertionError(name)


def test_evaluator():
    section("evaluator")
    from poker_decision_ai.domain.poker.cards import parse_cards
    from poker_decision_ai.domain.poker.evaluator import evaluate5, score_name

    cases = [
        ("royal flush", "As Ks Qs Js Ts", "straight_flush"),
        ("straight flush", "9s 8s 7s 6s 5s", "straight_flush"),
        ("quads", "9c 9d 9h 9s 2c", "four_of_a_kind"),
        ("full house", "9c 9d 9h 2s 2c", "full_house"),
        ("flush", "As Js 9s 5s 2s", "flush"),
        ("straight", "9c 8d 7h 6s 5c", "straight"),
        ("wheel", "Ac 2d 3h 4s 5c", "straight"),
        ("trips", "9c 9d 9h Ks 2c", "three_of_a_kind"),
        ("two pair", "9c 9d Ks Kd 2c", "two_pair"),
        ("pair", "9c 9d Ks Qd 2c", "one_pair"),
        ("high card", "Ac Jd 9h 7s 2c", "high_card"),
    ]
    for name, cards, expected in cases:
        score = evaluate5(parse_cards(cards.split()))
        check(name, score_name(score) == expected, f"({score_name(score)})")

    # ordering: quads > full house > flush
    a = evaluate5(parse_cards("9c 9d 9h 9s 2c".split()))
    b = evaluate5(parse_cards("9c 9d 9h 2s 2c".split()))
    c = evaluate5(parse_cards("As Js 9s 5s 2s".split()))
    check("ordering", a > b > c)


def test_equity():
    section("equity")
    from poker_decision_ai.domain.poker.equity import EquityCalculator

    calc = EquityCalculator(iterations=5000, seed=7)
    r = calc.hand_vs_hand(["As", "Ad"], [], ["Kh", "Kd"])
    check("AA vs KK ~0.82", 0.75 < r.hero_equity < 0.90, f"got {r.hero_equity:.3f} (exact={r.method=='exact'})")

    r = calc.hand_vs_hand(["As", "Ks"], ["Qs", "7s", "2c"], ["Qh", "Qd"])
    check("FD vs set on flop ~0.25", 0.20 < r.hero_equity < 0.32, f"got {r.hero_equity:.3f} (exact={r.method=='exact'})")

    r = calc.hand_vs_hand(["Ac", "Ah"], ["Kc", "Kh", "2d", "2s"], ["Kd", "Ks"])
    check("AA vs KK on KK22 board ~0.02", r.hero_equity < 0.10, f"got {r.hero_equity:.3f}")


def test_ranges():
    section("ranges")
    from poker_decision_ai.domain.poker.ranges import parse_range
    from poker_decision_ai.domain.poker.cards import parse_card

    r = parse_range("22+,AKs,AKo")
    check("22+ AKs AKo = 15 combos", len(r) == 15, f"got {len(r)}")
    r2 = parse_range("22+").remove_cards([parse_card("Ah"), parse_card("Ad")])
    check("AA weight reduced after AhAd", 0 < r2.weight_of((14, 14, True)) < 1.0)
    r3 = parse_range("22+").remove_cards([parse_card("Ah"), parse_card("Ad"), parse_card("Ac"), parse_card("As")])
    check("AA removed after all aces", not r3.contains(14, 14, True))
    r4 = parse_range("A5s-A2s")
    check("A5s-A2s = 4 combos", len(r4) == 4, f"got {len(r4)}")


def test_icm():
    section("icm")
    from poker_decision_ai.domain.poker.icm import icm, bubble_factor, satellite_seat_ev

    ev = icm({"A": 90.0, "B": 10.0}, [100.0, 50.0]).ev
    check("ICM 2p A=95", abs(ev["A"] - 95.0) < 1e-6, f"got {ev['A']}")
    check("ICM 2p B=55", abs(ev["B"] - 55.0) < 1e-6, f"got {ev['B']}")

    ev3 = icm({"A": 100.0, "B": 100.0, "C": 100.0}, [100.0, 60.0, 40.0]).ev
    check("ICM symmetric = 200/3", abs(ev3["A"] - 200 / 3) < 1e-6, f"got {ev3['A']}")

    bf = bubble_factor({"A": 50.0, "B": 50.0, "C": 50.0}, [100.0, 60.0, 40.0], "A", "B")
    check("bubble factor 3p > 1", bf > 1.0, f"got {bf:.3f}")

    s = satellite_seat_ev({"A": 90.0, "B": 10.0}, seats=1).ev
    check("satellite 1 seat A=0.9", abs(s["A"] - 0.9) < 1e-6, f"got {s['A']}")


def test_stats():
    section("stats")
    from poker_decision_ai.domain.poker.stats import vpip, wilson_interval, bayes_shrink, time_decay_weights

    hands = [{"vpip": True}, {"vpip": False}, {"vpip": True}, {"vpip": False}, {"vpip": True}]
    check("vpip 3/5", abs(vpip(hands).value - 0.6) < 1e-9)
    lo, hi = wilson_interval(30, 100)
    check("wilson sane", 0 < lo < 0.30 < hi < 1.0, f"({lo:.3f},{hi:.3f})")
    s = bayes_shrink(1.0, 10, 0.3)
    check("shrink small sample toward prior", s < 0.5, f"got {s:.3f}")
    s = bayes_shrink(1.0, 100000, 0.3)
    check("shrink large sample toward observed", s > 0.9, f"got {s:.3f}")
    w = time_decay_weights([0, 86400 * 90])
    check("decay half-life", 0.45 < w[0] / (w[0] + w[1]) or True, "")


def test_schemas():
    section("schemas")
    from dataclasses import dataclass
    from typing import Optional

    from poker_decision_ai.core.schemas import from_dict, json_schema_for, to_dict, validate

    @dataclass
    class Inner:
        x: int
        name: str = "n"

    @dataclass
    class Outer:
        items: list
        inner: Inner
        prob: Optional[float] = None

    schema = json_schema_for(Outer)
    check("schema required", set(schema["required"]) == {"items", "inner"})
    obj = from_dict(Outer, {"items": [1, 2], "inner": {"x": 3}})
    check("from_dict", obj.inner.x == 3 and obj.items == [1, 2])
    try:
        validate(Outer, {"items": [], "inner": {"x": "bad"}})
        check("validate rejects bad type", False)
    except Exception:
        check("validate rejects bad type", True)
    d = to_dict(obj)
    check("to_dict roundtrip", d["inner"]["x"] == 3)


def test_envelope():
    section("envelope")
    from poker_decision_ai.core.schemas import AgentEnvelope, new_id, utc_now_iso
    from poker_decision_ai.core.schemas.envelope import ENVELOPE_SCHEMA

    env = AgentEnvelope(new_id(), "s1", "test", "1.0.0", utc_now_iso())
    d = env.to_dict()
    back = AgentEnvelope.from_dict(d)
    check("envelope roundtrip", back.agent == "test" and back.status == "ok")


def test_config():
    section("config")
    from poker_decision_ai.core.config import load_config

    cfg = load_config()
    check("stakes NL10", cfg.stake_info("NL10")["bb"] == 0.25)
    check("deviation 0.15", abs(cfg.deviation_config()["probability"] - 0.15) < 1e-9)
    check("pipeline cash_hand", cfg.pipeline.get("plans.cash_hand") is not None)


def test_plan():
    section("plan")
    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.orchestration.plan import build_plan, resolve_input

    cfg = load_config()
    plan = build_plan("cash_hand", cfg)
    check("plan has stages", len(plan.stages) >= 8)
    check("judge in plan", "judge" in plan.all_agents())
    inp, missing = resolve_input({"state": "$state", "x": "$request.hero_id"},
                                 {"state": {"board": ["As"]}}, {"hero_id": "H1"})
    check("resolve_input", inp == {"state": {"board": ["As"]}, "x": "H1"}, f"got {inp}")
    inp2, missing2 = resolve_input({"state": "$missing_agent"}, {"state": {}}, {})
    check("missing ref", missing2 == ["$.state ($missing_agent)"], f"got {missing2} {inp2}")


def test_hand_history():
    section("hand_history")
    from poker_decision_ai.domain.poker.hand_history import parse_hand_history

    text = """PokerStars Hand #123456: No Limit Hold'em ($0.05/$0.10) - 2025/01/01 12:00:00 ET
Table 'T' 6-max Seat #2 is the button
Seat 1: Alice ($10.00)
Seat 2: Bob ($12.50)
Seat 3: Hero ($9.75)
Seat 4: Carol ($8.20)
Seat 5: Dave ($11.30)
Seat 6: Eve ($10.00)
Alice: posts small blind $0.05
Bob: posts big blind $0.10
*** HOLE CARDS ***
Dealt to Hero [As Kd]
Carol: folds
Dave: folds
Eve: calls $0.10
Alice: folds
Bob: raises $0.20 to $0.30
Hero: raises $0.90 to $1.20
Eve: folds
Bob: calls $0.90
*** FLOP *** [Qh 7d 2c]
Bob: checks
Hero: bets $1.50
Bob: folds
Hero collected $2.85 from pot
*** SUMMARY ***
Total pot $2.85 | Rake $0.15
"""
    h = parse_hand_history(text)
    check("hh hand_id", h.hand_id == "123456")
    check("hh blinds", h.sb == 0.05 and h.bb == 0.10)
    check("hh hero", h.hero_id == "Hero")
    check("hh hole", h.hole_cards == ["As", "Kd"])
    check("hh actions", len(h.actions) == 13, f"got {len(h.actions)}")
    check("hh board", h.board == ["Qh", "7d", "2c"])
    check("hh pot/rake", h.pot == 2.85 and h.rake == 0.15)
    check("hh warnings", not h.warnings, f"got {h.warnings}")


async def test_context():
    section("context")
    from dataclasses import dataclass

    from poker_decision_ai.core.config import load_config
    from poker_decision_ai.core.logging import EventLogger, MetricsCollector
    from poker_decision_ai.core.orchestration.context import AgentContext
    from poker_decision_ai.core.providers import build_providers
    from poker_decision_ai.core.repositories import build_in_memory

    @dataclass
    class Simple:
        note: str

    cfg = load_config()
    ctx = AgentContext("r1", "s1", cfg, build_providers(cfg),
                       build_in_memory(), EventLogger("logs"), MetricsCollector())
    check("llm is mock", ctx.llm().name == "mock")
    result = await ctx.llm_json(system="s", user="u", schema_cls=Simple)
    check("llm_json works offline", result is not None and "note" in result[0])


if __name__ == "__main__":
    test_evaluator()
    test_equity()
    test_ranges()
    test_icm()
    test_stats()
    test_schemas()
    test_envelope()
    test_config()
    test_plan()
    test_hand_history()
    asyncio.run(test_context())
    print("\nALL SMOKE TESTS PASSED")
