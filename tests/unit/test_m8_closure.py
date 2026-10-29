"""M8 — 两项 M7 已记录缺口的闭环回归（最小集）。

闭环 1（原 `test_m7_b7_unknown_fallback.py::test_dealer_conflict_must_not_yield_position`）：
    dealer/button 冲突不得再经冻结 `positions_for` 的"回退固定顺序"分支
    产生非空 position。

闭环 2（原 `test_m7_b7_mode_dispatch.py::test_mtt_must_also_strip_top_level_bounties_table`）：
    MTT 顶层必须彻底禁止 PKO `bounties` 状态串入。

两项修复均**未修改冻结文件**：`positions_for` 零修改，修复位于调用侧
（`vision2.assembly._positions_for`）与 mode sanitize 的禁止键集合。
"""
from __future__ import annotations

import copy

import pytest

from poker_decision_ai import strategy_mode_switch as sms
from poker_decision_ai.domain.poker.models import positions_for
from poker_decision_ai.vision2.assembly import assemble
from poker_decision_ai.vision2.schemas import RawSeat, RawVisionState

_SEATS = [RawSeat(seat=1, occupied=True), RawSeat(seat=3, occupied=True),
          RawSeat(seat=6, occupied=True)]


def _raw(**kw) -> RawVisionState:
    base = {"source": "mock", "synthetic_test_only": True, "game_type": "cash",
            "poker_variant": "nlhe", "table_size": 6, "street": "preflop",
            "seats": list(_SEATS)}
    base.update(kw)
    return RawVisionState(**base)


# ── 闭环 1：dealer 冲突 → position 全 null ─────────────────────────────
def test_dealer_conflict_yields_no_position_at_all():
    """冲突 button（不在座位表内）→ 全部 position None，且 valid=False。"""
    for bad_button in (99, 0, -1, 7):
        n = assemble(_raw(button_seat=bad_button, hero_seat=1))
        assert all(p["position"] is None for p in n.players), bad_button
        assert n.hero["position"] is None, bad_button
        assert n.validation.valid is False, bad_button


def test_missing_dealer_yields_no_position():
    n = assemble(_raw(button_seat=None, hero_seat=1))
    assert all(p["position"] is None for p in n.players)
    assert n.hero["position"] is None


def test_valid_dealer_still_resolves_position():
    """对照：dealer 合法时位置必须真实推导（防"一律 null"式伪通过）。"""
    n = assemble(_raw(button_seat=3, hero_seat=1))
    assert n.validation.valid is True
    assert n.hero["position"] == "BB"
    assert {p["seat"]: p["position"] for p in n.players} == {
        1: "BB", 3: "BTN", 6: "SB"}


def test_invalid_button_is_never_forwarded_to_frozen_positions_for(monkeypatch):
    """证明修复点：本层绝不把无效 button_seat 传给冻结函数。

    `_positions_for` 内部以函数级 import 取 `positions_for`，故探针打在被
    import 的源模块上（models.positions_for）。
    """
    from poker_decision_ai.domain.poker import models as models_mod

    seen: list = []
    real = models_mod.positions_for

    def probe(players, button_seat):
        seen.append(button_seat)
        return real(players, button_seat)

    monkeypatch.setattr(models_mod, "positions_for", probe)
    n = assemble(_raw(button_seat=99, hero_seat=1))
    assert seen == [], f"无效 button 被转发给冻结函数: {seen}"
    assert n.hero["position"] is None


def test_valid_button_is_forwarded_exactly_once(monkeypatch):
    from poker_decision_ai.domain.poker import models as models_mod

    seen: list = []
    real = models_mod.positions_for

    def probe(players, button_seat):
        seen.append(button_seat)
        return real(players, button_seat)

    monkeypatch.setattr(models_mod, "positions_for", probe)
    n = assemble(_raw(button_seat=3, hero_seat=1))
    assert seen == [3]
    assert n.hero["position"] == "BB"


def test_frozen_positions_for_semantics_unchanged():
    """冻结 positions_for 零修改回归（含其既有 fallback 分支原样保留）。"""
    players = [{"seat": 1}, {"seat": 2}, {"seat": 3}]
    assert positions_for(players, 2) == {1: "BB", 2: "BTN", 3: "SB"}
    # 冻结函数自身的"回退固定顺序"分支未被本任务改动（仍按原语义）
    assert positions_for(players, 99) == {1: "BTN", 2: "SB", 3: "BB"}


def test_positions_are_independent_of_button_validity_paths():
    """无座位证据时亦不得 fallback。"""
    n = assemble(_raw(button_seat=3, hero_seat=1, seats=[]))
    assert n.players == []
    assert n.hero["position"] is None


# ── 闭环 2：MTT 顶层 bounties 不得串入 ─────────────────────────────────
def _pko_state() -> dict:
    return {"game_type": "pko", "pot": 10.0,
            "players": [{"player_id": "a", "stack": 100.0, "bounty": 5.0},
                        {"player_id": "b", "stack": 100.0}],
            "tournament": {"payouts": [{"place": 1, "amount": 100.0}]},
            "bounty_type": "pko", "bounties": {"a": 5.0}}


def test_mtt_strips_every_bounty_carrier():
    clean, stripped = sms.sanitize_state_for_mode(_pko_state(), "mtt")
    assert "bounties" not in clean
    assert "bounty_type" not in clean
    assert all("bounty" not in p for p in clean["players"])
    assert set(stripped) == {"state.bounty_type", "state.bounties",
                             "players.bounty"}


def test_mtt_after_sanitize_reports_bounty_forbidden_not_enabled():
    """MTT 下 bounty 必须 forbidden；即便原 state 带完整 PKO bounty 上下文。"""
    clean, _ = sms.sanitize_state_for_mode(_pko_state(), "mtt")
    rep = sms.mode_capability_report(clean, "mtt")
    assert rep["capabilities"]["bounty"] == {"permission": "forbidden",
                                             "status": "forbidden"}
    assert rep["capabilities"]["icm"]["permission"] == "permitted"


def test_pko_keeps_bounties_and_mtt_does_not_bleed_back():
    """PKO 保留 bounty 上下文；同一 state 走 MTT 后不得残留（双向不串线）。"""
    pko_clean, pko_stripped = sms.sanitize_state_for_mode(_pko_state(), "pko")
    assert pko_stripped == []
    assert pko_clean["bounties"] == {"a": 5.0}
    assert pko_clean["players"][0]["bounty"] == 5.0
    mtt_clean, _ = sms.sanitize_state_for_mode(_pko_state(), "mtt")
    assert sms.mode_capability_report(mtt_clean, "mtt")["capabilities"][
        "bounty"]["status"] == "forbidden"


def test_sanitize_does_not_mutate_input():
    st = _pko_state()
    snap = copy.deepcopy(st)
    sms.sanitize_state_for_mode(st, "mtt")
    assert st == snap


def test_cash_and_mtt_bounty_forbidden_sets_are_consistent():
    """CASH 与 MTT 的 bounty 禁止覆盖面必须一致（防再次漂移）。"""
    cash_keys = set(sms._FORBIDDEN_STATE_KEYS["cash"])
    mtt_keys = set(sms._FORBIDDEN_STATE_KEYS["mtt"])
    for carrier in ("bounty_type", "bounties"):
        assert carrier in cash_keys and carrier in mtt_keys, carrier


# ── 剩余 4 个 sealed 读取路径：显式登记为「未接线」（可审计，不掩盖）────
SEALED_UNGUARDED = {
    "poker_decision_ai/multistreet_v2_benchmark.py":
        "docs/V3F_R21_MANIFEST.json",
    "poker_decision_ai/multistreet_v3_benchmark.py":
        "docs/V3F_R22_MANIFEST.json",
}


def test_sealed_benchmark_modules_still_match_their_seal():
    """这 4 个读取路径所在文件仍与历史封存逐字节一致（未被 M8 改动）。"""
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for mod, man_rel in SEALED_UNGUARDED.items():
        man = json.loads((root / man_rel).read_text(encoding="utf-8"))
        assert mod in man["code_hashes"], mod
        now = hashlib.sha256((root / mod).read_bytes()).hexdigest()
        assert now == man["code_hashes"][mod], (
            f"{mod} 已偏离 {man_rel} 的封存——M8 禁止重写历史封存证据")


def test_sealed_read_paths_are_explicitly_unguarded():
    """如实登记缺口：这 4 处读取**不经过** ArtifactGuard。

    不接线的原因不是疏漏，而是"接线必须改这些字节 → 必然破坏历史封存，
    且无法在不重写历史 manifest 的前提下修复"。本测试把该边界固定下来，
    使缺口无法被无声地"看起来已解决"。
    """
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for mod in SEALED_UNGUARDED:
        src = (root / mod).read_text(encoding="utf-8")
        assert "ArtifactGuard" not in src
        assert "artifact_guard" not in src
        assert "require(" not in src
        # 这 4 处读取仍走模块内自带的 _read()（未校验）
        assert "def _read(" in src


def test_guarded_read_paths_do_use_the_guard():
    """对照：已接线的 5 处必须真的调用 guard（防回退/误删）。"""
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    qb = (root / "poker_decision_ai/quality_benchmark.py").read_text(
        encoding="utf-8")
    mb = (root / "poker_decision_ai/multistreet_benchmark.py").read_text(
        encoding="utf-8")
    assert "_guard().require(" in qb
    assert "_guard().require(" in mb
    assert "ArtifactGuard(ARTIFACT_MANIFEST, root=ROOT)" in qb
    assert "ArtifactGuard(ARTIFACT_MANIFEST, root=ROOT)" in mb
