# -*- coding: utf-8 -*-
"""P30-9-N：规则下限尺寸（具体尺度）+ JSON 建议输出 + 统计 + 重试策略。"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cash_game_ui.runner as R                                            # noqa: E402
import poker_decision_ai.vision2.pixel.realclient.production_bridge as pb  # noqa: E402
from poker_decision_ai import action_semantics as asem                     # noqa: E402
from cash_game_ui.runner import CashGameRunner                             # noqa: E402

OBS_25 = {"blinds": {"sb": 2.0, "bb": 5.0}, "street": "flop",
          "hole_cards": ["As", "Kd"], "pot": 12.0}


# ── 规则下限尺寸（具体尺度） ─────────────────────────────────────────
def test_rule_size_raise_is_2bb_and_converted_to_chips():
    fd = {"action": "raise", "recommended_size": None, "reasoning": "x"}
    out = asem.apply_rule_size(fd, OBS_25)
    su = out["rule_size_suggestion"]
    assert su["bb"] == 2.0 and su["chips"] == 10.0 and su["bb_chips"] == 5.0
    assert su["source"] == "rule_floor" and su["is_strategy_size"] is False
    assert "规则下限参考" in out["reasoning"] and "10 chip" in out["reasoning"]


def test_rule_size_bet_is_1bb():
    out = asem.apply_rule_size(
        {"action": "bet", "recommended_size": None, "reasoning": "x"}, OBS_25)
    assert out["rule_size_suggestion"]["bb"] == 1.0
    assert out["rule_size_suggestion"]["chips"] == 5.0


def test_rule_size_not_added_when_strategy_size_exists():
    out = asem.apply_rule_size(
        {"action": "raise", "recommended_size": 30.0, "reasoning": "x"}, OBS_25)
    assert "rule_size_suggestion" not in out
    assert out["recommended_size"] == 30.0


def test_rule_size_without_blinds_gives_bb_only():
    out = asem.apply_rule_size(
        {"action": "raise", "recommended_size": None, "reasoning": "x"}, {})
    su = out["rule_size_suggestion"]
    assert su["bb"] == 2.0 and su["chips"] is None


def test_rule_size_not_applicable_for_call_check():
    for act in ("call", "check", "fold"):
        out = asem.apply_rule_size(
            {"action": act, "recommended_size": None, "reasoning": "x"}, OBS_25)
        assert "rule_size_suggestion" not in out


# ── JSON 建议输出 ───────────────────────────────────────────────────
def test_advice_json_written_with_stable_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(CashGameRunner, "_advice_dir",
                        staticmethod(lambda: str(tmp_path)))
    run = CashGameRunner()
    fd = {"action": "raise", "recommended_size": None, "confidence": 0.5,
          "reasoning": "r", "rule_size_suggestion": {"action": "raise", "bb": 2.0,
                                                      "chips": 10.0}}
    p = run._write_advice_json(fd, OBS_25, {"blinds_source": "user_setting"}, "OK")
    assert p and pathlib.Path(p).exists()
    d = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    assert d["schema"] == "cash_game_advice/v1"
    assert d["action"] == "raise" and d["size_bb"] is None
    assert d["size_rule"]["chips"] == 10.0
    assert d["vision"]["hero_cards"] == ["As", "Kd"]
    assert d["vision"]["blinds"] == {"sb": 2.0, "bb": 5.0}
    assert d["provenance"]["blinds_source"] == "user_setting"
    assert "BASELINE" in d["honesty"]["strategy_mode_note"]


def test_advice_json_size_chips_conversion(tmp_path, monkeypatch):
    monkeypatch.setattr(CashGameRunner, "_advice_dir",
                        staticmethod(lambda: str(tmp_path)))
    run = CashGameRunner()
    p = run._write_advice_json(
        {"action": "bet", "recommended_size": 3.0}, OBS_25, {}, "OK")
    d = json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    assert d["size_bb"] == 3.0 and d["size_chips"] == 15.0


# ── 统计埋点 ────────────────────────────────────────────────────────
def test_attempt_log_appends_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = CashGameRunner()
    run._log_attempt("OK", [], 0, "raise", "/x/advice.json")
    run._log_attempt("DECISION_UNAVAILABLE", ["blinds"], 2, None, None)
    lines = (tmp_path / "logs" / "attempts.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    a, b = (json.loads(x) for x in lines)
    assert a["status"] == "OK" and a["action"] == "raise"
    assert b["blocking"] == ["blinds"] and b["recapture"] == 2


def test_stats_script_counts_rate(tmp_path):
    import subprocess
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    rows = [{"ts": "2026-10-09 18:14:04", "status": "OK", "blocking": [], "recapture": 0,
             "action": "raise", "advice_json": "x"},
            {"ts": "2026-10-09 18:20:00", "status": "DECISION_UNAVAILABLE",
             "blocking": ["blinds"], "recapture": 1, "action": None, "advice_json": None}]
    (tmp_path / "logs" / "attempts.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")
    out = subprocess.run([sys.executable, str(ROOT / "scripts/advice_stats.py"),
                          "--dir", str(tmp_path)],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", cwd=str(ROOT))
    assert "logs/attempts.jsonl" in out.stdout     # 记录来源
    assert "50.0%" in out.stdout                    # OK 率（1/2）
    assert "blinds" in out.stdout                   # 阻断字段分布
    assert "OK" in out.stdout and "DECISION_UNAVAILABLE" in out.stdout


# ── 重试策略：瞬时类才重试、预算 2、早退 ─────────────────────────────
def _patch(monkeypatch, blocking, reason=None):
    state = {"captures": 0, "bridges": 0, "status": "UNAVAILABLE"}

    def fake_capture(self):
        state["captures"] += 1
        return b"x" * 16, 100, 100, (0, 0, 100, 100)

    def fake_bridge(rgba, w, h, **kw):
        state["bridges"] += 1
        return {"status": state["status"], "reason": reason,
                "bridge": {"blocking_fields": list(blocking),
                            "observation": {}, "unknown_fields": []},
                "request": {}}

    monkeypatch.setattr(CashGameRunner, "capture_frame", fake_capture)
    monkeypatch.setattr(pb, "frame_to_decision_request", fake_bridge)
    return state


def test_retry_budget_is_two_for_transient(monkeypatch):
    st = _patch(monkeypatch, ["hero_cards", "pot"])
    run = CashGameRunner()
    run.analyze(b"f", 100, 100, allow_recapture=True)
    assert st["bridges"] == 3          # 1 首发 + 2 重试（预算 2）
    assert run._last_recapture == 2
    assert run._last_recapture_classes == ["hero_cards,pot", "hero_cards,pot"]


def test_no_retry_for_blinds_only(monkeypatch):
    st = _patch(monkeypatch, ["blinds"])
    run = CashGameRunner()
    run.analyze(b"f", 100, 100, allow_recapture=True)
    assert st["bridges"] == 1 and run._last_recapture == 0


def test_retry_for_street_unrepresentable(monkeypatch):
    st = _patch(monkeypatch, [], reason="street_unrepresentable:None;occupied_unknown")
    run = CashGameRunner()
    run.analyze(b"f", 100, 100, allow_recapture=True)
    assert st["bridges"] == 3
    assert run._last_recapture_classes[0] == "street_unrepresentable"
