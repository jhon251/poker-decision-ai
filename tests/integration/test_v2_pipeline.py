"""Phase 12 (V2) 集成测试：四模式 Realtime+Deep 全周期 / 回放 / Dashboard API / 诚实状态。"""
from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from poker_decision_ai.core.schemas import new_id
from poker_decision_ai.dashboard import DashboardApp, make_handler
from poker_decision_ai.deep import DeepRunner
from poker_decision_ai.realtime import RealtimeRunner
from poker_decision_ai.replay_engine import ReplayEngine

ROOT = Path(__file__).parent.parent.parent

CASH = (ROOT / "examples/hands/cash_flop_bet.txt").read_text(encoding="utf-8")
MTT = (ROOT / "examples/hands/tournament_bubble.txt").read_text(encoding="utf-8")
SAT = (ROOT / "examples/hands/satellite_final.txt").read_text(encoding="utf-8")
PKO = (ROOT / "examples/hands/pko_flop.txt").read_text(encoding="utf-8")

MODES = [
    ("cash", CASH, {}),
    ("tournament:mtt", MTT, {"game_mode": "tournament", "tournament_type": "mtt"}),
    ("tournament:satellite", SAT, {"game_mode": "tournament", "tournament_type": "satellite"}),
    ("bounty:pko", PKO, {"game_mode": "bounty", "bounty_type": "pko"}),
]


async def test_four_modes_realtime_deep_cycle(runtime):
    """四模式各自跑 Realtime → Deep 全周期，provider_status 始终诚实。"""
    for spec, text, overrides in MODES:
        request = {"kind": "tournament_hand" if ":" in spec else "cash_hand",
                   "hand_history_text": text, "hero_id": "Hero", **overrides}
        rr = RealtimeRunner(runtime)
        rt = await rr.decide(request)
        assert rt.final_decision is not None
        assert rt.game_context["game_mode"] == spec.split(":")[0]
        assert rt.provider_status["solver"]["status"] in ("mock", "real", "unavailable")
        dr = DeepRunner(runtime)
        deep = await dr.deepen(request, rt)
        assert deep.status in ("ok", "degraded")
        # realtime 决策未被 deep 覆盖
        rec = runtime.repos.decisions.get(rt.request_id)
        assert rec["path"] == "realtime"
        if deep.revision:
            assert deep.revision["realtime_action"] and deep.revision["deep_action"]


async def test_replay_engine_on_pko(runtime):
    result = await ReplayEngine(runtime).replay_hand(
        PKO, hero_id="Hero", mode_overrides={"game_mode": "bounty", "bounty_type": "pko"})
    assert len(result.decision_points) >= 1
    for d in result.decision_points:
        assert d.game_context["game_mode"] == "bounty"
        assert d.provider_status["solver"]["status"] in ("mock", "real", "unavailable")


async def test_realtime_deep_revision_cycle(runtime):
    from poker_decision_ai.core.schemas import utc_now_iso

    rr = RealtimeRunner(runtime)
    rt = await rr.decide({"kind": "cash_hand", "hand_history_text": CASH, "hero_id": "Hero"})
    # 构造一个与 realtime 不同的 deep 结果并写 revision（验证记录不覆盖 realtime）
    runtime.repos.decisions.save({
        "request_id": f"{rt.request_id}:revision:deep", "path": "revision",
        "revision": {"revision_id": f"{rt.request_id}:revision:deep",
                     "request_id": rt.request_id,
                     "realtime_action": "call", "deep_action": "fold",
                     "realtime_confidence": 0.8, "deep_confidence": 0.5,
                     "confidence_change": -0.3, "reason": "test",
                     "evidence": [], "model_versions": {}, "created_at": utc_now_iso()},
        "created_at": utc_now_iso(),
    })
    rt_after = runtime.repos.decisions.get(rt.request_id)
    assert rt_after["path"] == "realtime"
    assert rt_after["final_decision"]["action"] == rt.final_decision["action"]


# ---------------------------------------------------------------- Dashboard
@pytest.fixture
def dashboard_server():
    from http.server import ThreadingHTTPServer

    app = DashboardApp()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield app, server
    server.shutdown()


def _http_get(port: int, path: str):
    import urllib.request

    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as resp:
        return resp.status, resp.read().decode("utf-8")


def test_dashboard_status_api(dashboard_server):
    app, server = dashboard_server
    status, body = _http_get(server.server_address[1], "/api/status")
    assert status == 200
    data = json.loads(body)
    assert data["provider_status"]["solver"]["status"] == "mock"
    assert "cash" in data["capabilities"]
    assert "bounty:pko" in data["capabilities"]
    assert "model_versions" in data["caches"]


def test_dashboard_html_page(dashboard_server):
    app, server = dashboard_server
    status, body = _http_get(server.server_address[1], "/")
    assert status == 200
    assert "Provider 状态" in body
    assert "MOCK" in body.upper()
    assert "tournament:satellite" in body


def test_dashboard_analyze_api(dashboard_server):
    app, server = dashboard_server
    import urllib.request

    payload = json.dumps({"kind": "cash_hand", "hand_history_text": CASH,
                          "hero_id": "Hero"}).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}/api/analyze",
                                 data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    assert data["realtime"]["final_decision"] is not None
    assert data["deep"]["final_decision"] is not None
    assert data["realtime"]["provider_status"]["solver"]["status"] == "mock"


def test_dashboard_replay_api(dashboard_server):
    app, server = dashboard_server
    import urllib.request

    payload = json.dumps({"hand_history_text": (ROOT / "examples/hands/cash_full_hand.txt").read_text(encoding="utf-8"),
                          "hero_id": "Hero"}).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}/api/replay",
                                 data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    assert data["hand_id"] == "6001"
    assert len(data["decision_points"]) == 5
