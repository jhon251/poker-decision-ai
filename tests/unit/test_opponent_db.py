# -*- coding: utf-8 -*-
"""P30-9-HY 测试：**对手档案库**（每次点击记录逐座位证据 ✓；只记真实可见字段 ✓）。

覆盖：一次遭遇写 6 行（含 hero 标记 ✓）；缺名字 ⇒ `seat:N` + 标志 ✓；
未知字段写 NULL（**不编造** ✓）；同一手牌相邻遭遇的迁移推断 ✓；
关闭开关 ⇒ 不写库 ✓；报表带样本量 ✓。
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from poker_decision_ai.core import opponents as op  # noqa: E402


def _obs(street, stacks, in_hand, bets=None, board=None, names=None):
    names = names or {i: f"P{i}" for i in range(1, 7)}
    seats = [{"seat": i, "name": names.get(i), "stack": stacks[i],
               "in_hand": in_hand[i]} for i in range(1, 7)]
    return {"street": street, "board": board or ["As", "Kd", "7h"], "pot": 12.0,
             "to_call": 4.0, "seats": seats, "button_seat": 5, "hero_seat": 6,
             "blinds": {"sb": 0.5, "bb": 1.0}, "warnings": [{"code": "X"}],
             "field_confidences": 0.9, "extra": {"v3f_bets": bets or {}}}


def _db(tmp_path) -> pathlib.Path:
    return tmp_path / "opp.db"


def test_one_encounter_writes_six_rows_and_flags_hero(tmp_path, monkeypatch):
    db = _db(tmp_path)
    r = op.record_encounter(_obs("flop", {i: 100 for i in range(1, 7)},
                                  {i: 1 for i in range(1, 7)}, {"S1": 4.0}),
                             path=db)
    assert r["ok"] and r["observations"] == 6 and r["players"] == 6
    con = sqlite3.connect(str(db))
    n = con.execute("SELECT COUNT(*) FROM observations WHERE click_id=?", (r["click_id"],)).fetchone()[0]
    hero = con.execute("SELECT COUNT(*) FROM observations WHERE is_hero=1").fetchone()[0]
    bet = con.execute("SELECT bet_front FROM observations WHERE seat=1").fetchone()[0]
    nobet = con.execute("SELECT bet_front FROM observations WHERE seat=2").fetchone()[0]
    con.close()
    assert n == 6 and hero == 1, "一次遭遇应写 6 行且 hero 仅 1 行 ✓"
    assert bet == 4.0 and nobet is None, "前置下注额应如实写入；未提供者写 NULL ✓（不编造 ✗）"


def test_missing_name_falls_back_to_seat_key(tmp_path):
    db = _db(tmp_path)
    r = op.record_encounter(_obs("flop", {i: 100 for i in range(1, 7)},
                                  {i: 1 for i in range(1, 7)},
                                  names={1: None, 2: "  "}), path=db)
    assert r["ok"]
    con = sqlite3.connect(str(db))
    rows = dict(con.execute("SELECT seat, player_key FROM observations").fetchall())
    miss = dict(con.execute("SELECT seat, name_missing FROM observations").fetchall())
    con.close()
    assert rows[1] == "seat:1" and rows[2] == "seat:2", "缺名字 ⇒ seat:N ✓（不编造名字 ✗）"
    assert miss[1] == 1 and miss[2] == 1


def test_unavailable_fields_are_null_not_guessed(tmp_path):
    """实测：单帧拿不到动作序列 ✗ ⇒ 库里不得出现任何 fold/check/raise/call 标签字段 ✓。"""
    db = _db(tmp_path)
    op.record_encounter(_obs("flop", {i: 100 for i in range(1, 7)},
                              {i: 1 for i in range(1, 7)}), path=db)
    con = sqlite3.connect(str(db))
    cols = [r[1] for r in con.execute("PRAGMA table_info(observations)")]
    con.close()
    assert not any(c in cols for c in ("action", "verb", "vpip", "pfr")), \
        "不得凭空记录动作标签/VPIP/PFR ✗"


def test_same_hand_transitions_detect_invest_and_fold(tmp_path):
    db = _db(tmp_path)
    op.record_encounter(_obs("flop", {i: 100 for i in range(1, 7)},
                              {i: 1 for i in range(1, 7)}, {"S1": 4.0}), path=db)
    r2 = op.record_encounter(_obs("turn", {1: 96, 2: 100, 3: 100, 4: 100, 5: 100, 6: 100},
                                   {1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 1}, {"S1": 12.0},
                                   board=["As", "Kd", "7h", "2c"]), path=db)
    assert r2["transitions"] >= 2
    con = sqlite3.connect(str(db))
    kinds = dict(con.execute("SELECT basis, COUNT(*) FROM transitions GROUP BY basis"))
    confs = [r[0] for r in con.execute("SELECT confidence FROM transitions")]
    con.close()
    assert "stack_decrease" in kinds, "筹码减少 ⇒ 投入证据 ✓"
    assert "in_hand_to_out" in kinds, "在池→消失 ⇒ 弃牌证据 ✓"
    assert all(0.0 <= c <= 1.0 for c in confs), "每条推断必须带置信 ✓"


def test_different_hand_no_transition(tmp_path):
    """**换手**（庄家位变化 ✓）⇒ 不得跨手推断 ✓。"""
    db = _db(tmp_path)
    op.record_encounter(_obs("flop", {i: 100 for i in range(1, 7)},
                              {i: 1 for i in range(1, 7)}), path=db)
    o2 = _obs("flop", {1: 50, 2: 100, 3: 100, 4: 100, 5: 100, 6: 100},
               {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1})
    o2["button_seat"] = 2                      # 换手 ✓
    r = op.record_encounter(o2, path=db)
    assert r["ok"] and r["transitions"] == 0, "换手不得推断迁移 ✓"


def test_disabled_switch_writes_nothing(tmp_path, monkeypatch):
    db = _db(tmp_path)
    monkeypatch.setenv("OPPONENT_DB", "0")
    r = op.record_encounter(_obs("flop", {i: 100 for i in range(1, 7)},
                                  {i: 1 for i in range(1, 7)}), path=db)
    assert r["ok"] is False and "disabled" in str(r.get("reason"))
    assert not db.exists(), "关闭开关 ⇒ 不得建库 ✓"


def test_stats_carry_sample_size(tmp_path, monkeypatch):
    db = _db(tmp_path)
    op.record_encounter(_obs("flop", {i: 100 for i in range(1, 7)},
                              {i: 1 for i in range(1, 7)}), path=db)
    s = op.stats(db, min_sample=20)
    assert s["exists"] and s["encounters"] == 1 and s["observations"] == 6
    assert all("enough_sample" in r for r in s["rows"]), "报表必须带样本量判定 ✓"
    assert s["by_source"] in ({"runtime": 1}, {"test": 1}), "来源须如实标记 ✓"


def test_report_includes_position_and_street_distributions(tmp_path):
    """P30-9-HZ：⑤ 要求"各位置/街分布" ✓ —— 分母必须等于该对手遭遇数 ✓。"""
    db = _db(tmp_path)
    op.record_encounter(_obs("preflop", {i: 100 for i in range(1, 7)},
                              {i: 1 for i in range(1, 7)}), path=db)
    op.record_encounter(_obs("flop", {i: 100 for i in range(1, 7)},
                              {i: 1 for i in range(1, 7)}, board=["As", "Kd", "7h"]), path=db)
    s = op.stats(db, min_sample=20)
    for r in s["rows"]:
        assert "by_position" in r and "by_street" in r and "in_hand_by_position" in r
        assert sum(r["by_position"].values()) == r["encounters"], "位置分布分母=遭遇数 ✓"
        assert sum(r["by_street"].values()) == r["encounters"], "街分布分母=遭遇数 ✓"
        assert set(r["by_street"]) <= {"preflop", "flop", "turn", "river"}
        assert all(v >= 0 for v in r["in_hand_by_position"].values())
    # 两次遭遇 ⇒ 位置分布合计 2，街分布含 preflop 与 flop ✓
    one = s["rows"][0]
    assert sum(one["by_position"].values()) == 2 and "flop" in one["by_street"]


def test_db_write_does_not_change_decision(tmp_path, monkeypatch):
    """⑥ 硬要求：开关**不得改变识别/决策结果** ✓。

    做法：同一张真实帧分别以 `OPPONENT_DB=0` / `=1` 跑完整点击路径 ⇒ 决策字段必须逐项相同 ✓。
    （无真实帧时跳过 ✓——**不伪造帧** ✗。）
    """
    import json as _j

    from PIL import Image
    frames = sorted((ROOT / "dist" / "logs" / "frames").glob("*.png"))
    if not frames:
        import pytest
        pytest.skip("无真实帧（dist/logs/frames 为空）⇒ 跳过 ✓（不伪造帧 ✗）")
    sys.path.insert(0, str(ROOT))
    from cash_game_ui.runner import CashGameRunner

    def _run(db_on: bool, frame):
        monkeypatch.setenv("OPPONENT_DB", "1" if db_on else "0")
        monkeypatch.setenv("OPPONENT_DB_PATH", str(tmp_path / "x.db"))
        monkeypatch.setenv("DSH_EVIDENCE_SOURCE", "test")
        r = CashGameRunner()
        im = Image.open(frame).convert("RGBA")
        res = r.analyze(im.tobytes(), im.width, im.height, allow_recapture=False)
        d = res.to_dict() if hasattr(res, "to_dict") else dict(res)
        return ({k: d.get(k) for k in ("status", "decision_action", "final_decision",
                                        "reasoning", "solver_name", "bridge")},
                 getattr(r, "_last_opponents", None))

    # 有些帧本身读不出座位（例如整屏不可用帧 ✓）⇒ 取第一张**能写出观测**的帧 ✓
    probe = {"ok": False}
    frame = None
    for f in frames[:6]:
        _, probe = _run(True, f)
        if isinstance(probe, dict) and probe.get("ok"):
            frame = f
            break
    if frame is None:
        import pytest
        pytest.skip("可用真实帧未产生座位观测 ⇒ 跳过 ✓（不伪造 ✓）")
    (tmp_path / "x.db").unlink(missing_ok=True)
    off, _ = _run(False, frame)
    on, wrote = _run(True, frame)
    assert _j.dumps(off, sort_keys=True, default=str) == _j.dumps(on, sort_keys=True, default=str), \
        "对手库开关**不得改变**决策结果 ✓"
    assert isinstance(wrote, dict) and wrote.get("ok"), "开启时应写库 ✓"
    assert (tmp_path / "x.db").is_file(), "开启时应落盘 ✓"


# ── P30-9-IA：**动作标签**（只用可见筹码证据 ✓；模糊宁可不标 ✓；**永不** check ✗）──
def _ca(prev, cur, **kw):
    from poker_decision_ai.core.opponents import classify_action
    return classify_action(prev, cur, **kw)


def _labels(res):
    return [r["action"] for r in res]


def test_action_fold_and_no_check_ever():
    got = _ca({"stack": 100, "in_hand": 1, "bet_front": 0},
               {"stack": 100, "in_hand": 0, "bet_front": 0, "street": "turn"}, bb=1.0)
    assert _labels(got) == ["fold"] and got[0]["confidence"] >= 0.5
    # 未投入且仍在池 ⇒ **不得**标 check ✗（实测：两帧稀疏序列无法可靠判定过牌 ✓）
    quiet = _ca({"stack": 100, "in_hand": 1, "bet_front": 0},
                 {"stack": 100, "in_hand": 1, "bet_front": 0, "street": "flop"}, bb=1.0,
                 cur_max_bet_others=0.0)
    assert _labels(quiet) == [], "过牌不可推断 ⇒ 必须不标 ✓（宁可漏，不编造 ✗）"


def test_action_bet_call_raise_by_prior_bet():
    prev0 = {"stack": 100, "in_hand": 1, "bet_front": 0}
    bet = _ca(prev0, {"stack": 90, "in_hand": 1, "bet_front": 10, "street": "flop"},
               prev_max_bet_others=0.0, pot_delta=10.0, bb=1.0)
    assert _labels(bet) == ["bet"] and bet[0]["confidence"] > 0.7, "首个下注 + 底池同向 ⇒ bet ✓"
    call = _ca({"stack": 100, "in_hand": 1, "bet_front": 0},
                {"stack": 90, "in_hand": 1, "bet_front": 10, "street": "flop"},
                prev_max_bet_others=10.0, bb=1.0)
    assert _labels(call) == ["call"], "与上一帧他人下注持平 ⇒ call ✓"
    raised = _ca({"stack": 100, "in_hand": 1, "bet_front": 0},
                  {"stack": 75, "in_hand": 1, "bet_front": 25, "street": "flop"},
                  prev_max_bet_others=10.0, bb=1.0)
    assert _labels(raised) == ["raise"], "超过他人下注 ⇒ raise ✓"


def test_action_no_label_without_meaningful_change():
    """**诚实性检查** ✓：投入变化在容差内、或数据反而减少（不一致 ✗）⇒ **不标** ✓。

    （说明：一旦"增加超过容差"成立，bet/call/raise 三分支是**完备**的 ✓，
      不存在"模糊区间"——此前的假设有误 ✗，故测试改为检查这两类**真不该标**的情形 ✓。）
    """
    tiny = _ca({"stack": 100, "in_hand": 1, "bet_front": 10},
                {"stack": 99.5, "in_hand": 1, "bet_front": 10.5, "street": "flop"},
                prev_max_bet_others=10.0, bb=1.0)
    assert _labels(tiny) == [], "变化在容差内 ⇒ 不标 ✓"
    decrease = _ca({"stack": 100, "in_hand": 1, "bet_front": 25},
                    {"stack": 100, "in_hand": 1, "bet_front": 10, "street": "flop"},
                    prev_max_bet_others=25.0, bb=1.0)
    assert _labels(decrease) == [], "前置下注反而减少（数据不一致 ✗）⇒ 不标 ✓（不臆测 ✓）"


def test_action_allin_two_bases():
    a1 = _ca({"stack": 100, "in_hand": 1, "bet_front": 0},
              {"stack": 0.4, "in_hand": 1, "bet_front": 0, "street": "river"}, bb=1.0)
    assert "allin" in _labels(a1) and a1[0]["confidence"] >= 0.7
    a2 = _ca({"stack": 30, "in_hand": 1, "bet_front": 0},
              {"stack": 0, "in_hand": 1, "bet_front": 30, "street": "turn"}, bb=1.0)
    assert "allin" in _labels(a2)


def test_action_preflop_blind_is_not_call():
    """翻前 SB/BB 的强制投入 ⇒ `blind` ✓，**不得**误记成 call ✗。"""
    got = _ca({"stack": 100, "in_hand": 1, "bet_front": None},
               {"stack": 99, "in_hand": 1, "bet_front": 1, "street": "preflop",
                "position": "BB"}, prev_max_bet_others=0.5, bb=1.0)
    labs = _labels(got)
    assert "blind" in labs and "call" not in labs, f"盲注不得记成 call ✗（实得 {labs}）"


def test_action_street_change_uses_current_frame_max():
    """换街 ⇒ 上一街数值不可比 ✗：本街用**本帧他人最大下注**判断 ✓。"""
    open_bet = _ca({"stack": 100, "in_hand": 1, "bet_front": 25},
                    {"stack": 75, "in_hand": 1, "bet_front": 25, "street": "turn"},
                    prev_max_bet_others=25.0, cur_max_bet_others=0.0,
                    street_changed=True, bb=1.0)
    assert _labels(open_bet) == ["bet"], "换街后本帧无更高下注 ⇒ bet（置信 0.6 ✓）"
    assert open_bet[0]["confidence"] <= 0.6
    call = _ca({"stack": 100, "in_hand": 1, "bet_front": 0},
                {"stack": 75, "in_hand": 1, "bet_front": 25, "street": "turn"},
                prev_max_bet_others=0.0, cur_max_bet_others=25.0,
                street_changed=True, bb=1.0)
    assert _labels(call) == ["call"]


def test_actions_written_with_basis_and_evidence(tmp_path):
    """入库时动作必须带 basis + confidence + evidence（可审计 ✓）。"""
    db = _db(tmp_path)
    op.record_encounter(_obs("flop", {i: 100 for i in range(1, 7)},
                              {i: 1 for i in range(1, 7)}, {}), path=db)
    op.record_encounter(_obs("flop", {1: 90, 2: 100, 3: 100, 4: 100, 5: 100, 6: 100},
                              {i: 1 for i in range(1, 7)}, {"S1": 10.0}), path=db)
    con = sqlite3.connect(str(db))
    rows = list(con.execute("SELECT action, basis, confidence, evidence FROM actions"))
    con.close()
    assert rows, "应有动作行 ✓"
    for action, basis, conf, ev in rows:
        assert action in ("fold", "allin", "bet", "raise", "call", "blind")
        assert basis and 0.0 < conf <= 1.0 and ev and ev.startswith("{"), "必须带依据/置信/证据 ✓"
