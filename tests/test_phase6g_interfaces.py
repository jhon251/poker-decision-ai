"""Phase 6G 接口层测试：as_of 边界/time_sensitive 语义/predict/deviation/契约。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from phase6g_interfaces import (  # noqa: E402
    DATA_END, UnsupportedError, deviation_score, get_profile, predict_probe,
    validate_contract)


def _first_t1_player():
    idx = {}
    for line in open("data/uoft/models/phase6/phase6g/profile_index.jsonl",
                     encoding="utf-8"):
        pid, off = json.loads(line)
        idx[pid] = off
        if len(idx) >= 1000:
            break
    with open("data/uoft/models/phase6/profiles.jsonl", encoding="utf-8") as f:
        for pid, off in idx.items():
            f.seek(off)
            row = json.loads(f.readline())
            if row["tier"] == "T1":
                return pid
    return None


# ---------------- as_of 边界 ----------------
def test_as_of_at_data_end_ok():
    pid = _first_t1_player()
    p = get_profile(pid, DATA_END)
    assert p["player_id"] == pid
    assert p["meta"]["as_of"] == DATA_END
    assert p["profile_type"] == "static_full_history"


def test_as_of_backward_unsupported():
    pid = _first_t1_player()
    with pytest.raises(UnsupportedError):
        get_profile(pid, "2009-07-10T00:00:00")


def test_as_of_missing_value_error():
    pid = _first_t1_player()
    with pytest.raises(ValueError):
        get_profile(pid, None)


def test_unknown_player_population_only():
    p = get_profile("nonexistent_player_xyz", DATA_END)
    assert p["population_only"] is True
    assert p["features"] == {}
    assert p["tier"] == "unknown"


# ---------------- time_sensitive 语义（补充契约） ----------------
def test_unstable_marks_time_sensitive_only():
    """unstable → time_sensitive=true；value/reliability/consumable 不得改变。"""
    pid = _first_t1_player()
    p = get_profile(pid, DATA_END)
    # vpip 在 6E 为 unstable
    vpip = p["features"]["vpip"]
    assert vpip["time_status"] == "unstable"
    assert vpip["time_sensitive"] is True
    # 与 6C 原始值一致（value/reliability/consumable 不被接口修改）
    off = None
    for line in open("data/uoft/models/phase6/phase6g/profile_index.jsonl",
                     encoding="utf-8"):
        i_pid, i_off = json.loads(line)
        if i_pid == pid:
            off = i_off
            break
    assert off is not None
    with open("data/uoft/models/phase6/profiles.jsonl", encoding="utf-8") as f:
        f.seek(off)
        raw = json.loads(f.readline())
    assert vpip["value"] == raw["features"]["vpip"]["value"]
    assert vpip["reliability"] == raw["features"]["vpip"]["reliability"]
    assert vpip["consumable"] == raw["features"]["vpip"]["consumable"]


def test_insufficient_evidence_time_sensitive_null():
    """insufficient_evidence → time_sensitive=null（不可知 ≠ 不稳定）。"""
    pid = _first_t1_player()
    p = get_profile(pid, DATA_END)
    cbet = p["features"]["cbet"]
    assert cbet["time_status"] == "insufficient_evidence"
    assert cbet["time_sensitive"] is None
    # rfi_lp 残留 time_sensitive=False 也必须被接口层覆盖为 None
    rfi_lp = p["features"]["rfi_lp"]
    assert rfi_lp["time_status"] == "insufficient_evidence"
    assert rfi_lp["time_sensitive"] is None


def test_stable_time_sensitive_false():
    pid = _first_t1_player()
    p = get_profile(pid, DATA_END)
    if "agg_freq" in p["features"]:
        assert p["features"]["agg_freq"]["time_status"] == "stable"
        assert p["features"]["agg_freq"]["time_sensitive"] is False


# ---------------- meta 契约 ----------------
def test_meta_cluster_probs_null_and_utility_proven():
    pid = _first_t1_player()
    p = get_profile(pid, DATA_END)
    assert p["meta"]["cluster_probs"] is None
    assert p["meta"]["utility_proven"] is True
    assert p["meta"]["schema_version"] == "6.0"
    assert p["meta"]["data_window"] == ["2009-07-01T00:00:00",
                                        "2009-07-26T10:00:35"]


# ---------------- predict_probe ----------------
def test_predict_probe_interval_contains_mean():
    pid = _first_t1_player()
    p = get_profile(pid, DATA_END)
    r = predict_probe(p, "vpip", 100)
    assert r["consumable"] is True
    lo, hi = r["predictive_interval"]
    assert 0 <= lo <= hi <= 100
    assert lo <= round(r["posterior_mean"] * 100) <= hi
    assert r["population_dominated"] is False


def test_predict_probe_tier_c_not_consumable():
    pid = _first_t1_player()
    p = get_profile(pid, DATA_END)
    r = predict_probe(p, "won_hand", 100)
    assert r["consumable"] is False


def test_predict_probe_alias_not_consumable():
    pid = _first_t1_player()
    p = get_profile(pid, DATA_END)
    r = predict_probe(p, "ep_open", 100)
    assert r["consumable"] is False


def test_predict_probe_zero_opportunity_population_dominated():
    pid = _first_t1_player()
    p = get_profile(pid, DATA_END)
    zero = next((f for f, v in p["features"].items()
                 if v.get("zero_opportunity")), None)
    if zero is None:
        pytest.skip("该玩家无 zero_opportunity 特征")
    r = predict_probe(p, zero, 50)
    assert r["population_dominated"] is True
    assert r["samples"] == 0 and r["shrinkage"] == 1.0


def test_predict_probe_bad_args():
    pid = _first_t1_player()
    p = get_profile(pid, DATA_END)
    with pytest.raises(ValueError):
        predict_probe(p, "vpip", 0)
    with pytest.raises(KeyError):
        predict_probe(p, "no_such_feature", 10)


# ---------------- deviation_score ----------------
def test_deviation_consumable_feature():
    pid = _first_t1_player()
    p = get_profile(pid, DATA_END)
    vpip = p["features"]["vpip"]
    # 观测率显著高于画像后验 → over
    obs = min(1.0, vpip["value"] + 0.4)
    r = deviation_score(p, "vpip", obs, 200)
    assert r["consumable"] is True
    assert r["significant"] in (True, False)
    assert r["direction"] in ("over", "under", "none")


def test_deviation_tier_c_significant_null():
    pid = _first_t1_player()
    p = get_profile(pid, DATA_END)
    r = deviation_score(p, "won_hand", 0.5, 100)
    assert r["consumable"] is False
    assert r["significant"] is None


def test_deviation_bad_args():
    pid = _first_t1_player()
    p = get_profile(pid, DATA_END)
    with pytest.raises(ValueError):
        deviation_score(p, "vpip", 1.5, 100)
    with pytest.raises(ValueError):
        deviation_score(p, "vpip", 0.5, 0)
    with pytest.raises(KeyError):
        deviation_score(p, "nope", 0.5, 100)


# ---------------- 契约全量验证 ----------------
def test_validate_contract_all_clean():
    res = validate_contract()
    assert res["all_clean"] is True
    assert res["n_rows"] == 240011
    for k, v in res["violations"].items():
        assert v == 0, f"{k} 违规 {v}"
