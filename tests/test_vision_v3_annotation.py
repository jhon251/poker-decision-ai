"""V-3a 抽样器与比对纯逻辑单测（离线，不触语料/不启 GUI）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import v3_compare as vc            # noqa: E402
import v3_annotate as va           # noqa: E402
import v3_sample_frames as vs      # noqa: E402


def _row(file="F1.png", event="TABLE_CHANGED", street=None, obs=None,
         conf=None, reason=None, err=None):
    return {"file": file, "event": event, "ts": 1, "street": street,
            "observed_cardinality": obs, "confidence": conf,
            "reason": reason, "provenance": "x", "err": err}


# ── 抽样器：信号判定 ──────────────────────────────────────────────────
def test_signals_street_card_mismatch():
    rec = vs.normalize_row(_row(street="flop", obs=4))
    assert "street_card_mismatch" in vs.disagreement_signals(rec)
    ok = vs.normalize_row(_row(street="flop", obs=3))
    assert "street_card_mismatch" not in vs.disagreement_signals(ok)


def test_signals_ghost_and_partial_and_overfull():
    ghost = vs.normalize_row(_row(street="preflop", obs=2))
    assert "ghost_cards" in vs.disagreement_signals(ghost)
    assert "partial_cards" in vs.disagreement_signals(ghost)
    over = vs.normalize_row(_row(street="river", obs=6))
    sigs = vs.disagreement_signals(over)
    assert "overfull_board" in sigs and "street_card_mismatch" in sigs


def test_signals_street_unknown_low_conf_illegal_and_err():
    unk = vs.normalize_row(_row(street=None, obs=0))
    assert "street_unknown" in vs.disagreement_signals(unk)
    low = vs.normalize_row(_row(street="flop", obs=3, conf=0.3))
    assert "low_confidence" in vs.disagreement_signals(low)
    ill = vs.normalize_row(_row(reason="illegal_transition_0->4"))
    assert "illegal_transition" in vs.disagreement_signals(ill)
    err = vs.normalize_row(_row(err="decode fail"))
    assert vs.disagreement_signals(err) == ["frame_error"]
    clean = vs.normalize_row(_row(street="preflop", obs=0, conf=0.7,
                                  reason="zero_cards"))
    assert vs.disagreement_signals(clean) == []


# ── 抽样器：选择逻辑 ──────────────────────────────────────────────────
def test_pick_disagreement_rare_first_and_cap():
    rows = []
    # 500 个 preflop+2 卡帧（partial/ghost 大桶），3 个非法转换帧
    for i in range(500):
        rows.append(_row(file=f"P{i}.png", street="preflop", obs=2))
    for i in range(3):
        rows.append(_row(file=f"I{i}.png", reason="illegal_transition_0->4"))
    picked = vs.pick_disagreement([vs.normalize_row(r) for r in rows], 20)
    files = [p["file"] for p in picked]
    assert len(files) == len(set(files)) <= 20
    assert all(f.startswith("I") for f in files[:3])   # 稀有信号先入列
    assert len([f for f in files if f.startswith("I")]) == 3


def test_pick_disagreement_fallback_fills_to_n():
    rows = [_row(file=f"P{i}.png", street="preflop", obs=2)
            for i in range(400)]
    picked = vs.pick_disagreement([vs.normalize_row(r) for r in rows], 250)
    assert len(picked) == 250


def test_stratified_random_deterministic_and_disjoint():
    rows = []
    for i in range(200):
        rows.append(_row(file=f"T{i}.png", event="TABLE_CHANGED",
                         street="preflop", obs=0))
    for i in range(50):
        rows.append(_row(file=f"S{i}.png", event="SETTLED_BOARD_CHECK",
                         street="river", obs=5))
    recs = [vs.normalize_row(r) for r in rows]
    a = vs.stratified_random(recs, 30, set(), seed=7)
    b = vs.stratified_random(recs, 30, set(), seed=7)
    assert [r["file"] for r in a] == [r["file"] for r in b]
    assert len(a) == 30 and len({r["file"] for r in a}) == 30
    evs = {r["event"] for r in a}
    assert evs == {"TABLE_CHANGED", "SETTLED_BOARD_CHECK"}   # 每层至少 2
    excl = {r["file"] for r in a}
    c = vs.stratified_random(recs, 30, excl, seed=7)
    assert not ({r["file"] for r in c} & excl)


def test_select_frames_disjoint_and_counts():
    rows = []
    for i in range(300):
        rows.append(_row(file=f"P{i}.png", event="POT_CHANGED",
                         street="preflop", obs=2))
    for i in range(200):
        rows.append(_row(file=f"T{i}.png", event="TABLE_CHANGED",
                         street="preflop", obs=0))
    recs = [vs.normalize_row(r) for r in rows]
    dis, rnd, stats = vs.select_frames(recs, 250, 100, seed=42)
    assert len(dis) == 250 and len(rnd) == 100
    assert not ({r["file"] for r in dis} & {r["file"] for r in rnd})
    assert stats["disagreement_selected"] == 250


def test_pick_demo_distinct():
    rows = []
    for ev in ("TABLE_CHANGED", "POT_CHANGED", "SCENE_CHANGED",
               "SETTLED_BOARD_CHECK"):
        for i in range(10):
            rows.append(_row(file=f"{ev}_{i}.png", event=ev,
                             street="preflop", obs=0))
    picked = vs.pick_demo([vs.normalize_row(r) for r in rows], 4, seed=1)
    assert len(picked) == 4
    assert len({p["file"] for p in picked}) == 4
    assert len({p["event"] for p in picked}) == 4


# ── 标注工具：纯逻辑 ──────────────────────────────────────────────────
def test_record_complete_rules():
    rec = va.new_record("a.png", "TABLE_CHANGED", 1)
    assert not va.record_complete(rec)               # 无 street
    rec["street"] = "preflop"
    assert va.record_complete(rec)                   # hero 不可见即可
    rec["hero_status"] = "visible"
    assert not va.record_complete(rec)               # 可见但牌不齐
    rec["hero_cards"] = ["Ac", "Kd"]
    assert va.record_complete(rec)


def test_build_board_cards_filters_empty():
    slots = [("A", "h"), (va.EMPTY, "s"), ("T", va.EMPTY),
             (va.EMPTY, va.EMPTY), ("2", "c")]
    assert va.build_board_cards(slots) == ["Ah", "2c"]


def test_progress_and_annotations_roundtrip(tmp_path):
    order = ["a.png", "b.png"]
    recs = {"a.png": va.new_record("a.png", None, 1)}
    recs["a.png"].update(street="flop", board_cards=["Ah", "Td", "2s"])
    prog = tmp_path / "p.json"
    va.save_progress(prog, order, recs, "batch.jsonl")
    loaded = va.load_progress(prog)
    assert loaded["order"] == order and loaded["records"]["a.png"]["street"] \
        == "flop"
    out = tmp_path / "ann.jsonl"
    n = va.save_annotations(out, order, recs)
    assert n == 1                                    # 只有 a 完成
    lines = [json.loads(l) for l in
             out.read_text(encoding="utf-8").splitlines() if l]
    assert lines[0]["file"] == "a.png" and lines[0]["annotated_at"]
    # 加 b 完成后重写 → 2 条
    recs["b.png"] = va.new_record("b.png", None, 2)
    recs["b.png"].update(street="unclear")
    assert va.save_annotations(out, order, recs) == 2


# ── 比对：逐帧比较与聚合 ──────────────────────────────────────────────
def _mach(street=None, cards=(), countable=None, hero=None,
          prov="single_frame_machine"):
    return {"file": "F.png", "street": street,
            "observed_cardinality": len(cards) if countable is None
            else countable,
            "board_cards": list(cards), "board_countable": countable
            if countable is not None else len(cards),
            "rank_only_cards": [], "hero_cards": hero,
            "hero_status": "RECOGNIZED" if hero else "UNCERTAIN",
            "street_provenance": prov}


def _human(street="flop", cards=(), hero=None, hero_status="not_visible"):
    return {"file": "F.png", "street": street, "board_cards": list(cards),
            "hero_cards": hero, "hero_status": hero_status}


def test_compare_full_agreement():
    h = _human(street="flop", cards=["Ah", "Td", "2s"])
    m = _mach(street="flop", cards=["Td", "Ah", "2s"], countable=3)
    c = vc.compare_frame(h, m)
    assert c["street"] and c["count"] and c["cards"]
    assert c["hero"] is None and c["hero_order"] is None  # 不可见→剔除


def test_compare_machine_unresolved_and_mismatch():
    h = _human(street="flop", cards=["Ah", "Td", "2s"])
    m = _mach(street=None, cards=[], countable=0)
    c = vc.compare_frame(h, m)
    assert c["street"] is False and c["count"] is False
    assert c["machine"]["street"] is None
    m2 = _mach(street="flop", cards=["Ah", "Kd", "2s"], countable=3)
    c2 = vc.compare_frame(h, m2)
    assert c2["cards"] is False and c2["count"] and c2["street"]


def test_compare_unclear_excluded_and_hero_order():
    c = vc.compare_frame(_human(street="unclear"), _mach(street="flop"))
    assert c["street"] is None and c["count"] is None
    h = _human(street="preflop", hero=["6c", "3h"],
               hero_status="visible")
    m = _mach(street="preflop", hero=["3h", "6c"])
    c2 = vc.compare_frame(h, m)
    assert c2["hero"] is True and c2["hero_order"] is False


def test_aggregate_rates():
    comps = []
    for i, ok in enumerate([True, True, False, None]):
        comps.append({"file": f"F{i}", "street": ok, "count": None,
                      "rank": None, "suit": None, "board3": None,
                      "cards": None, "hero": None, "hero_order": None,
                      "human": {}, "machine": {"street": "preflop"}})
    agg = vc.aggregate(comps)
    assert agg["street"] == (2, 3, 2 / 3)


def test_systematic_errors_confusion():
    # V-2b 口径：多重集差（machine countable rank 多 vs human 多配对）；
    # suit 假阳性 = FULL 声明整牌不在 human 牌面。
    comps = [
        {"file": "A", "cards": False, "rank": True, "suit": False,
         "board3": False,
         "human": {"board_cards": ["Ah", "Kd"]},
         "machine": {"board_cards": ["As", "Kd"],
                     "rank_only_cards": []}},
        {"file": "B", "cards": False, "rank": False, "suit": False,
         "board3": False,
         "human": {"board_cards": ["2c"]},
         "machine": {"board_cards": ["3c"],
                     "rank_only_cards": []}},
    ]
    se = vc.systematic_errors(comps)
    assert se["rank_confusion"]["3->2"] == 1
    assert se["suit_false_positive"]["As"] == 1
    assert se["rank_extra_machine"] == {"3": 1}
    assert se["rank_extra_human"] == {"2": 1}


def test_render_report_headline_format():
    comps = [
        {"file": "F1.png", "street": True, "count": True, "cards": True,
         "rank": True, "suit": True, "board3": True,
         "hero": None, "hero_order": None,
         "human": {"street": "flop", "board_cards": ["Ah", "Td", "2s"],
                   "hero_cards": None, "hero_status": "not_visible"},
         "machine": {"street": "flop", "observed_cardinality": 3,
                     "board_cards": ["Ah", "Td", "2s"], "board_countable": 3,
                     "rank_only_cards": [], "hero_cards": None,
                     "hero_status": "UNCERTAIN",
                     "street_provenance": "single_frame_exemption"}},
    ]
    comps.append(dict(comps[0], file="F2.png", cards=False, rank=False,
                      suit=False, board3=False,
                      human=dict(comps[0]["human"],
                                 board_cards=["Ah", "Kd", "2s"])))
    agg = vc.aggregate(comps)
    se = vc.systematic_errors(comps)
    meta = {"batch": "b.jsonl", "seed": 1, "n_annotated": 2, "n_batch": 2,
            "n_compared": 2, "systematic": se}
    text = vc.render_report(agg, comps, meta, adjudications=None)
    k, n, r = agg["board3"]
    assert f"board 三层全过一致率 {r * 100:.2f}%（n={n}）" in text
    assert "→ 达标" in text or "→ 未达标" in text
    assert "分歧 1" in text and "F2.png" in text
    assert "（待复核）" in text
    text2 = vc.render_report(agg, comps, meta,
                             adjudications={"F2.png": "机器错"})
    assert "机器错" in text2
