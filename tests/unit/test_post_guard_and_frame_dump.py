# -*- coding: utf-8 -*-
"""P30-9-AF/AG/AH：不可能投入守卫、hero 强制投入下限、失败存帧、观察者原因透出。"""
from __future__ import annotations

import glob
import pathlib
import sys

ROOT = pathlib.Path(r"<repo>")
sys.path.insert(0, str(ROOT))

import numpy as np                                                        # noqa: E402
from poker_decision_ai import facing_evidence as fe                        # noqa: E402
from cash_game_ui.runner import CashGameRunner                             # noqa: E402

#: 用户截图实测值：blinds 0.05/0.15；S1 下注 0.2；S4(底中=你) 0.05；S5(右下) 0.02
SCREEN = {"street": "preflop", "pot": 0.27,
          "blinds": {"sb": 0.05, "bb": 0.15}, "hero_seat": 5, "button_seat": 1,
          "seats": [{"seat": i} for i in range(1, 7)],
          "extra": {"v3f_bets": {"S1": 0.2, "S2": None, "S3": None,
                                  "S4": 0.05, "S5": 0.02, "S6": None},
                     "v3f_hero_position": "BB"}}


def test_implausible_preflop_post_below_small_blind_excluded():
    """翻前任何人投入不可能小于小盲（0.02 < 0.05）⇒ 判误读并排除。"""
    bad = fe.implausible_preflop_posts(SCREEN)
    assert bad["seats"] == ["S5"]
    hb = fe.hero_opp_bets(SCREEN)
    assert "S5" in (hb["detail"].get("implausible_preflop_posts") or [])
    # 该误读值不得进入 opp_max（否则 0.02 会被当成"对手投入"）
    assert hb["opp_max"] == 0.2


def test_hero_forced_post_floor_uses_blind_when_read_too_low():
    """hero=BB、bb=0.15，却把 hero 区读成 0.05 ⇒ 按盲注计并如实记录。"""
    hb = fe.hero_opp_bets(SCREEN)
    assert hb["hero_bet"] == 0.15
    assert hb["detail"]["hero_post_floor_applied"]["read"] == 0.05
    assert hb["detail"]["hero_post_floor_applied"]["floor"] == 0.15


def test_to_call_matches_client_after_floor():
    """修正后 to_call 应为 0.2 − 0.15 = 0.05（此前会算成 0.15，偏大 3 倍）。"""
    assert fe.to_call_lower_bound({**SCREEN, "to_call": None})["to_call"] == 0.05


def test_floor_not_applied_postflop():
    obs = {**SCREEN, "street": "flop", "board": ["Kh", "7d", "2c"]}
    hb = fe.hero_opp_bets(obs)
    assert "hero_post_floor_applied" not in hb["detail"]


def test_seat_bets_text_uses_geometric_roles_and_flags_misread():
    txt = CashGameRunner._seat_bets_text(SCREEN)
    assert "底中(你)" in txt and "左上" in txt
    assert "误读" in txt and "0.02" in txt


def test_observer_reason_passthrough():
    obs = {"hole_cards": [], "hero_seat": 5, "button_seat": 1,
           "warnings": ["hero_cards 未识别 reason=no_card_pair_present"]}
    d = fe.hero_cards_diagnosis(obs)
    assert d["reason"] == "hero_cards_region_no_match"
    assert d["observer_reason"] == "no_card_pair_present"


def test_failure_frame_dumped(tmp_path, monkeypatch):
    """失败必须自动存帧（不再依赖用户手动截图）。"""
    run = CashGameRunner()
    run._log_dir = str(tmp_path)
    frame = np.zeros((1400, 1876, 4), dtype=np.uint8)
    frame[..., 3] = 255
    res = run.analyze(frame.tobytes(), 1876, 1400, allow_recapture=False)
    assert res.status != "OK"
    files = glob.glob(str(tmp_path / "logs" / "frames" / "*.png"))
    assert files, "失败未存帧"
    assert run._last_frame_path and run._last_frame_path.endswith(".png")


def test_no_frame_dump_on_success_path(monkeypatch, tmp_path):
    """成功路径不写失败帧目录（避免无意义占用）。"""
    run = CashGameRunner()
    run._log_dir = str(tmp_path)
    run._last_frame = None
    assert run._dump_failure_frame("UNAVAILABLE", []) is None
    assert not (tmp_path / "logs" / "frames").exists()
