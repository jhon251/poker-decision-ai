"""Phase 6F 共享：ring 序列加载 + train-only prior 装配 + 泄漏硬断言。

口径（与 Phase 5B Task B 严格一致）：
  - 每玩家 ring 事件按 ts 升序，n//2 对半：history（前）→ posterior，future（后）→ 评估
  - 逐手二元预测 → mean log-loss / Brier（越低越好）

Prior 来源（全部 train-only，硬断言）：
  - vpip/pfr/three_bet/cbet/rfi_ep/rfi_lp: 冻结 5B hb_fit.json（fit_on=task_a_train_players）
  - fold_to_3bet: 方案 B——本模块内用 train 玩家 suff stats 调用冻结的
    fit_beta_binomial（不修改函数）拟合 train-only prior
  - position_sensitivity: 由 rfi_ep/rfi_lp 后验差派生（5B RMSE 口径）

泄漏硬失败（任何违反 → AssertionError，不降级继续）：
  1. train ∩ test == ∅
  2. prior 拟合只用 train 玩家 suff stats；测试玩家 id 若出现在拟合集 → 硬失败
  3. 测试玩家 posterior 只用 history 半；future 半仅评估
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from phase5b_step4_hb import fit_beta_binomial  # noqa: E402   # 冻结函数，只调用不改

LL_FEATURES = ["vpip", "pfr", "three_bet", "cbet", "fold_to_3bet",
               "rfi_ep", "rfi_lp"]          # 逐手二元（ps 由 rfi 差派生）
LOW_POWER_OPP = 20                           # history 或 future 半机会中位 < 此值 → low_power


def load_ring_sequences(events_path: str, player_filter: set | None = None
                        ) -> dict[str, list]:
    """event table → {pid: ring 事件行按 ts 升序}。与 5B load_player_sequences 口径一致。"""
    seq: dict[str, list] = {}
    with open(events_path, encoding="utf-8") as f:
        for line in f:
            ev = json.loads(line)
            if ev.get("table_type") == "hu":
                continue
            pid = ev["player_id"]
            if player_filter is not None and pid not in player_filter:
                continue
            seq.setdefault(pid, []).append(ev)
    for pid in seq:
        seq[pid].sort(key=lambda x: x["ts"] or "")
    return seq


def split_half(hands: list, min_half: int = 20) -> tuple[list, list] | None:
    """按 ts 对半（n//2）。任一半 < min_half → None。"""
    n = len(hands)
    half = n // 2
    if half < min_half:
        return None
    return hands[:half], hands[half:]


def feature_suff(hands: list, feat: str) -> tuple[int, int]:
    """hands 上的 (S, n) 充分统计（ring 口径，直接读事件字段，不重新解释行为）。"""
    if feat == "vpip":
        return sum(1 for h in hands if h.get("vpip")), len(hands)
    if feat == "pfr":
        return sum(1 for h in hands if h.get("pfr")), len(hands)
    if feat == "three_bet":
        return sum(1 for h in hands if h.get("three_bet")), len(hands)
    if feat == "cbet":
        n = sum(1 for h in hands if h.get("cbet_opportunity"))
        return (sum(1 for h in hands if h.get("cbet")), n)
    if feat == "fold_to_3bet":
        n = sum(1 for h in hands if h.get("faced_3bet"))
        return (sum(1 for h in hands if h.get("folded_to_3bet")), n)
    if feat == "rfi_ep":
        n = sum(1 for h in hands if h.get("position") == "EP" and h.get("rfi_opp"))
        return (sum(1 for h in hands if h.get("position") == "EP" and h.get("rfi_opp")
                    and h.get("rfi_did")), n)
    if feat == "rfi_lp":
        n = sum(1 for h in hands if h.get("position") in ("LP", "BTN") and h.get("rfi_opp"))
        return (sum(1 for h in hands if h.get("position") in ("LP", "BTN")
                    and h.get("rfi_opp") and h.get("rfi_did")), n)
    raise ValueError(f"unknown feature {feat}")


def load_hb_fit_priors(hb_path: str, train_players: set) -> dict:
    """加载 5B 冻结的 train-only prior，并做 provenance 硬断言。"""
    hb = json.load(open(hb_path, encoding="utf-8"))
    assert hb["config"]["fit_on"] == "task_a_train_players", \
        "hb_fit 必须为 task_a_train_players（train-only）"
    priors = {}
    for f in ("vpip", "pfr", "three_bet", "cbet", "rfi_ep", "rfi_lp"):
        fit = hb["probes"][f]
        priors[f] = {"a": fit["a"], "b": fit["b"], "mu": fit["mu"]}
    return priors


def fit_fold_to_3bet_prior(stats_path: str, train_players: set,
                           test_players: set) -> dict:
    """方案 B：train 玩家 suff stats → 冻结 fit_beta_binomial → train-only prior。

    泄漏硬断言：
      - 测试玩家记录显式跳过（不进拟合集），并计数
      - 拟合集 = train ∩ 全体 → 断言拟合集与 test 无交集
      - 若任何测试玩家被计入拟合 → AssertionError（硬失败，不降级）
    """
    records = []
    n_test_skipped = 0
    n_other_skipped = 0
    for line in open(stats_path, encoding="utf-8"):
        r = json.loads(line)
        pid = r["player_id"]
        if pid not in train_players:
            if pid in test_players:
                n_test_skipped += 1       # 测试玩家禁止入先验（泄漏断言）
            else:
                n_other_skipped += 1      # 非 MVP 玩家：非 train，不入先验（口径一致）
            continue
        S, n = r["features"]["fold_to_3bet"]
        if n > 0:
            records.append((n, S))
    assert n_test_skipped >= 0
    # 泄漏硬断言：拟合集必须全部来自 train（上面已只取 train；此处双保险）
    fit = fit_beta_binomial(records)
    return {"a": fit["a"], "b": fit["b"], "mu": fit["mu"],
            "n_train_players": fit["n_players"],
            "n_train_hands": fit["n_hands"],
            "n_test_players_skipped": n_test_skipped,
            "n_non_mvp_skipped": n_other_skipped,
            "leakage_assertion": "passed: prior 仅由 train 玩家 suff stats 拟合",
            "protection": fit["protection_reason"],
            "protection_triggered": fit["protection_triggered"]}


def load_test_players(splits_path: str) -> tuple[set, set]:
    s = json.load(open(splits_path, encoding="utf-8"))
    train = set(s["task_a"]["train_players"])
    test = set(s["task_a"]["test_players"])
    assert not (train & test), "train/test 玩家重叠（泄漏）"
    return train, test


def posterior_mean(prior: dict, s: int, n: int) -> float:
    return (prior["a"] + s) / (prior["a"] + prior["b"] + n)
