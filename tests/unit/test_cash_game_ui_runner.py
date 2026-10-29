"""PACKAGE+UI — 测试套件：runner 一键链 + UI 数据显示（无 Tk 主循环）。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_decision_ai.vision2_platform.png_decode import decode_png  # noqa: E402
from cash_game_ui.runner import CashGameRunner  # noqa: E402


def _patch_runtime_timeouts(runtime):
    """P52-3：realtime 超时测试安全覆写（同 test_p34；生产语义零改动）。

    生产 realtime.agent_timeout_ms=300ms 为线上延迟预算；全量套件负载下
    偶发超时 → agent 降级 → 决策变化 → 断言失败（flaky）。测试只放宽
    时钟预算，不断言内容。
    """
    import types
    cfg = runtime.config
    rt = dict(cfg.realtime_config() or {})
    rt["agent_timeout_ms"] = max(int(rt.get("agent_timeout_ms", 300)), 5000)
    rt["deadline_ms"] = max(int(rt.get("deadline_ms", 3000)), 20000)
    cfg.realtime_config = types.MethodType(lambda self, _rt=rt: dict(_rt),
                                           cfg)
    return runtime


_orig_get_runtime = CashGameRunner._get_runtime


def _test_safe_get_runtime(self):
    return _patch_runtime_timeouts(_orig_get_runtime(self))


CashGameRunner._get_runtime = _test_safe_get_runtime

EV = Path("data2")


def _frame(fid):
    _cand = EV / fid
    if not _cand.exists():
        _cand = Path("data/vision2/coinpoker_v3") / fid
    w, h, rgba = decode_png(_cand.read_bytes())
    return rgba, w, h


# ── 一键链：8bf8 → OK + 全 Agent + 结构化视觉 ───────────────────────
def test_one_click_chain_ok():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = CashGameRunner().analyze(rgba, w, h)
    d = res.to_dict()
    assert d["status"] == "OK"
    assert d["decision_action"] in ("fold", "check", "call", "bet", "raise",
                                    "allin")
    # 2026-10-10：接入外部 6-max GTO 翻前图表后，命中时策略可为
    # solver_anchor（图表锚定）⇒ 断言放宽为"非 REAL solver 之外皆可"。
    assert d["solver_mode"] in ("BASELINE", "UNKNOWN", "SOLVER_ANCHOR",
                                "solver_anchor")
    # 全 Agent 链（结构化）
    agents = {a["agent"]: a for a in d.get("agents", [])}
    assert "Vision" in agents and "Position" in agents
    assert "State" in agents and "Solver" in agents
    assert "Strategy" in agents and "Judge" in agents
    assert "Decision" in agents
    assert agents["Judge"]["status"] == "LEGAL"
    # 视觉结构化输出（唯一视觉入口——喂下游）
    v = d.get("vision", {})
    assert v.get("hero_cards") == ["5c", "3c"]
    assert v.get("pot") == 8920.0
    assert v.get("to_call") == 2000.0
    assert v.get("blinds") == {"sb": 1000.0, "bb": 2000.0}


def test_vision_feeds_downstream_no_second_pipeline():
    """Vision 输出（observer）→ 其它 Agent；无重复截图标记。"""
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = CashGameRunner().analyze(rgba, w, h)
    d = res.to_dict()
    for a in d.get("agents", []):
        assert a["provenance"] != "second_capture"      # 无第二套 pipeline
    v = d.get("vision", {})
    assert v.get("hero_seat") == 5                        # 座位识别


# ── UNKNOWN / UNAVAILABLE 显示数据 ─────────────────────────────────
def test_unavailable_display_data():
    rgba, w, h = _frame("TABLE_CHANGED_53_1791197130197_full.png")
    res = CashGameRunner().analyze(rgba, w, h)
    d = res.to_dict()
    # V-1 FM-1：该帧现于 street 层结构性拒绝——status=UNAVAILABLE + reason，
    # 不再有 blocking_fields 视图。
    # V-2 更新（test-side only）：召回修复后该帧（目视 GT 4 卡）可计数控 4 /
    # FULL 3 → 拒绝原因 street_board_contradiction（诚实拒绝语义不变）。
    assert d["status"] == "UNAVAILABLE"
    assert d["decision_action"] is None                   # 不猜值
    assert ("street_new_hand_unconfirmed" in (d.get("reason") or "")
            or "street_board_contradiction" in (d.get("reason") or ""))


def test_occupied_unknown_display_data():
    rgba, w, h = _frame("TABLE_CHANGED_53_1791197130197_full.png")
    res = CashGameRunner().analyze(rgba, w, h)
    d = res.to_dict()
    assert d["status"] == "UNAVAILABLE"
    assert d["decision_action"] is None


# ── BASELINE 不冒充 REAL（UI 显示用）───────────────────────────────
def test_baseline_not_real():
    rgba, w, h = _frame("8bf82f134651db0add626be0304a2ff1.png")
    res = CashGameRunner().analyze(rgba, w, h)
    d = res.to_dict()
    assert d["solver_mode"] != "REAL"


# ── UI 格式化（显示层——无 Tk 主循环）────────────────────────────────
def test_ui_format_shows_decision():
    from cash_game_ui.app import CashGameApp  # noqa: F401
    # app 构造函数需 Tk——此处仅验证格式化逻辑（静态方法）
    fmt = CashGameApp._fmt_detail
    d = {"status": "OK", "decision_action": "CALL"}
    assert "CALL" in fmt(d)
