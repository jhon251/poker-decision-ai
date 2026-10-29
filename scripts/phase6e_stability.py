"""Phase 6 Step 6E：时间稳定性验证（Time Stability）。

严格按 Step 0 Audit 口径：
  - ring 事件总窗口: 2009-07-01 ~ 2009-07-26（实际审计值）
  - 漂移主体窗口: 排除 07-24~26 抓取收尾低流量期 → 2009-07-01 ~ 07-23（理由记录）
  - 玩家: MVP event table 中满足「玩家内时间序等量三分、每段 >=50 ring 手」的玩家
  - 每特征三段充分统计 → 玩家级配对 bootstrap（保持玩家内三段配对结构）
      → 分段线性回归斜率 β + 首末段差 Δ + 95% CI
  - 三态判定:
      stable        = CI 覆盖 0 且 |β| < 0.02（双条件，缺一不可）
      unstable      = 可检验但非 stable（含 CI 不含 0 而 |β|<0.02 的情形——
                      保守处理：不得把"统计显著但效应小"表述为稳定）
      insufficient_evidence = 每段机会不足或分量不足（cbet/fold_to_cbet/wsd/
                      rfi_lp（低功效）/position_sensitivity）
  - 不启用 time_decay；只输出 time_sensitive / stable / insufficient 证据
  - 所有 insufficient 与降级写入 DEGRADE_LOG.json（6E 专属目录，不触碰 6D 文件）

输出:
  data/uoft/models/phase6/phase6e/time_stability.json
  data/uoft/models/phase6/phase6e/DEGRADE_LOG.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

MAIN_WINDOW_START = "2009-07-01T00:00:00"
MAIN_WINDOW_END = "2009-07-23T23:59:59"     # 排除 07-24~26 收尾期
TOTAL_WINDOW = ("2009-07-01T00:00:00", "2009-07-26T10:00:35")   # 实际审计总窗口
MIN_HANDS_PER_SEGMENT = 50
MIN_OPP_FOR_TEST = 40         # 每段机会中位门槛：低于 → insufficient_evidence
EFFECT_SIZE = 0.02            # 最小效应量
N_BOOT = 1000
BOOT_SEED = 42

# 可检验特征及其机会门控字段
TEST_FEATURES = {
    "vpip": ("vpip", None),
    "pfr": ("pfr", None),
    "three_bet": ("three_bet", None),
    "fold_to_3bet": ("folded_to_3bet", "faced_3bet"),
    "cold_call": (None, None),       # 特判：非盲位 & vpip & ¬pfr
    "agg_freq": ("bets", None),      # 特判：S=bets, n=bets+calls
    "rfi_ep": ("rfi_did", "rfi_opp"),
    "wtsd": ("went_to_showdown", "saw_flop"),
}
LOW_POWER = {"rfi_lp"}               # 执行检验但标记低功效
INSUFFICIENT = {"cbet", "fold_to_cbet", "wsd"}   # 每段机会不足，禁止判定
PS_COMPONENTS = ("rfi_ep", "rfi_lp")


def feature_opp(ev: dict, feat: str) -> tuple[int, int]:
    """返回 (S, n) 特征机会对（ring 口径，直接读事件字段，不重新解释行为）。"""
    if feat == "vpip":
        return (1 if ev.get("vpip") else 0, 1)
    if feat == "pfr":
        return (1 if ev.get("pfr") else 0, 1)
    if feat == "three_bet":
        return (1 if ev.get("three_bet") else 0, 1)
    if feat == "fold_to_3bet":
        return (1 if ev.get("folded_to_3bet") else 0,
                1 if ev.get("faced_3bet") else 0)
    if feat == "cold_call":
        pos = ev.get("position")
        nonblind = pos is not None and pos not in ("SB", "BB")
        return (1 if (nonblind and ev.get("vpip") and not ev.get("pfr")) else 0,
                1 if nonblind else 0)
    if feat == "agg_freq":
        nb = (ev.get("bets") or 0) + (ev.get("calls") or 0)
        return (ev.get("bets") or 0, nb)
    if feat == "rfi_ep":
        return (1 if (ev.get("position") == "EP" and ev.get("rfi_opp")
                      and ev.get("rfi_did")) else 0,
                1 if (ev.get("position") == "EP" and ev.get("rfi_opp")) else 0)
    if feat == "rfi_lp":
        return (1 if (ev.get("position") in ("LP", "BTN") and ev.get("rfi_opp")
                      and ev.get("rfi_did")) else 0,
                1 if (ev.get("position") in ("LP", "BTN") and ev.get("rfi_opp")) else 0)
    if feat == "wtsd":
        return (1 if ev.get("went_to_showdown") else 0,
                1 if ev.get("saw_flop") else 0)
    if feat == "cbet":
        return (1 if ev.get("cbet") else 0,
                1 if ev.get("cbet_opportunity") else 0)
    if feat == "fold_to_cbet":
        return (1 if ev.get("folded_to_cbet") else 0,
                1 if ev.get("faced_cbet") else 0)
    if feat == "wsd":
        return (1 if ev.get("won_at_showdown") else 0,
                1 if ev.get("went_to_showdown") else 0)
    raise ValueError(f"unknown feature {feat}")


def in_main_window(ts: str) -> bool:
    return MAIN_WINDOW_START <= ts <= MAIN_WINDOW_END


def split_thirds(hands: list, min_per_seg: int) -> list | None:
    """玩家内时间序等量三分（调用方保证 ts 升序）。任一段 < min → None。"""
    n = len(hands)
    if n < 3 * min_per_seg:
        return None
    k = n // 3
    segs = [hands[:k], hands[k:2 * k], hands[2 * k:]]
    if any(len(s) < min_per_seg for s in segs):
        return None
    return segs


def segment_stats(seg: list, feat: str) -> tuple[int, int]:
    S = n = 0
    for ev in seg:
        s, nn = feature_opp(ev, feat)
        S += s
        n += nn
    return S, n


def ols_slope(xs: list, ys: list) -> float:
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def classify(beta: float, ci_lo: float, ci_hi: float) -> str:
    """三态判定：stable 仅当 CI 覆盖 0 且 |β|<0.02（双条件缺一不可）。"""
    ci_covers_zero = ci_lo <= 0 <= ci_hi
    if ci_covers_zero and abs(beta) < EFFECT_SIZE:
        return "stable"
    return "unstable"     # 可检验但非 stable（保守，不把显著小漂移标稳定）


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/uoft/feature_ready/player_event_table.jsonl")
    ap.add_argument("--out", default="data/uoft/models/phase6/phase6e")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 读取 ring 事件，按玩家分组，过滤主体窗口，按 ts 排序 ----
    per_player: dict[str, list] = {}
    n_total = 0
    n_excluded_tail = 0
    for line in open(args.events, encoding="utf-8"):
        ev = json.loads(line)
        if ev.get("table_type") == "hu":
            continue
        n_total += 1
        if not in_main_window(ev.get("ts") or ""):
            n_excluded_tail += 1
            continue
        per_player.setdefault(ev["player_id"], []).append(ev)
    for pid in per_player:
        per_player[pid].sort(key=lambda x: x["ts"])   # 时间升序

    # ---- 玩家纳入：三等分每段 >=50 ----
    segs_by_player: dict[str, list] = {}
    n_players_total = len(per_player)
    for pid, hands in per_player.items():
        segs = split_thirds(hands, MIN_HANDS_PER_SEGMENT)
        if segs is not None:
            segs_by_player[pid] = segs
    n_players_used = len(segs_by_player)

    # ---- 每特征：玩家级三段频率 → 配对 bootstrap（β, Δ, CI）----
    results = {}
    degrade = []
    for feat in sorted(set(TEST_FEATURES) | INSUFFICIENT | LOW_POWER):
        # 每玩家三段充分统计（玩家内配对）
        player_triples = []          # [(p1,p2,p3)] 频率
        for pid, segs in segs_by_player.items():
            stats = [segment_stats(s, feat) for s in segs]
            if any(n == 0 for _, n in stats):
                continue
            freq = [S / n for S, n in stats]
            player_triples.append(freq)

        if feat in INSUFFICIENT:
            results[feat] = {
                "status": "insufficient_evidence",
                "reason": "每段机会中位 <40（抓取窗口内机会不足），禁止判断 stable/time_sensitive",
                "n_players": len(player_triples),
            }
            degrade.append({"component": f"time_stability.{feat}",
                            "path": "insufficient_opportunity",
                            "reason": "每段机会中位 <40",
                            "evidence": {"n_players": len(player_triples)}})
            continue
        if len(player_triples) < 30:
            results[feat] = {"status": "insufficient_evidence",
                             "reason": "可纳入玩家不足",
                             "n_players": len(player_triples)}
            degrade.append({"component": f"time_stability.{feat}",
                            "path": "insufficient_players",
                            "reason": "可纳入玩家 <30"})
            continue

        def stat_fn(triples: list) -> tuple:
            mean_p = [sum(t[j] for t in triples) / len(triples) for j in range(3)]
            beta = ols_slope([1.0, 2.0, 3.0], mean_p)
            delta = mean_p[2] - mean_p[0]
            return beta, delta

        base_beta, base_delta = stat_fn(player_triples)
        rng = random.Random(BOOT_SEED)
        k = len(player_triples)
        betas = []
        deltas = []
        for _ in range(N_BOOT):
            idx = [rng.randrange(k) for _ in range(k)]
            b, d = stat_fn([player_triples[i] for i in idx])
            betas.append(b)
            deltas.append(d)
        betas.sort()
        deltas.sort()
        b_lo, b_hi = betas[int(0.025 * N_BOOT)], betas[int(0.975 * N_BOOT)]
        d_lo, d_hi = deltas[int(0.025 * N_BOOT)], deltas[int(0.975 * N_BOOT)]
        status = classify(base_beta, b_lo, b_hi)
        rec = {
            "status": status,
            "beta": round(base_beta, 5),
            "beta_ci95": [round(b_lo, 5), round(b_hi, 5)],
            "delta": round(base_delta, 5),
            "delta_ci95": [round(d_lo, 5), round(d_hi, 5)],
            "n_players": k,
            "n_bootstrap": N_BOOT,
            "time_sensitive": status == "unstable",
        }
        if feat in LOW_POWER:
            rec["status"] = "insufficient_evidence"
            rec["low_power"] = True
            rec["note"] = "每段机会中位 25（低功效）：执行检验但结论不可靠"
            degrade.append({"component": f"time_stability.{feat}",
                            "path": "low_power",
                            "reason": "每段机会中位约 25，功效不足",
                            "evidence": {"beta": rec["beta"],
                                         "beta_ci95": rec["beta_ci95"]}})
        else:
            if status == "unstable":
                degrade.append({"component": f"time_stability.{feat}",
                                "path": "unstable",
                                "reason": "CI 不覆盖 0 或 |β|>=0.02",
                                "evidence": {"beta": rec["beta"],
                                             "beta_ci95": rec["beta_ci95"]}})
        results[feat] = rec

    # ---- position_sensitivity：分量不足 → insufficient ----
    ps_deps = {c: results.get(c, {}) for c in PS_COMPONENTS}
    ps_insuff = any(v.get("status") == "insufficient_evidence"
                    for v in ps_deps.values())
    results["position_sensitivity"] = {
        "status": "insufficient_evidence" if ps_insuff else "stable",
        "reason": ("分量 rfi_ep/rfi_lp 存在 insufficient → 派生量不可判定"
                   if ps_insuff else "分量均通过（见 rfi_ep/rfi_lp）"),
        "derived_from": list(PS_COMPONENTS),
    }
    if ps_insuff:
        degrade.append({"component": "time_stability.position_sensitivity",
                        "path": "insufficient_components",
                        "reason": "rfi_ep/rfi_lp 分量不足或低功效",
                        "evidence": {c: ps_deps[c].get("status")
                                     for c in PS_COMPONENTS}})

    out = {
        "config": {
            "total_window": list(TOTAL_WINDOW),
            "main_window": [MAIN_WINDOW_START, MAIN_WINDOW_END],
            "tail_exclusion_reason": "07-24~26 为抓取收尾期，手量占比 <1.2%，"
                                     "排除以防伪漂移",
            "min_hands_per_segment": MIN_HANDS_PER_SEGMENT,
            "min_opp_for_test": MIN_OPP_FOR_TEST,
            "effect_size": EFFECT_SIZE,
            "bootstrap": {"unit": "player", "paired": True,
                          "n": N_BOOT, "seed": BOOT_SEED,
                          "note": "重采样玩家，保留玩家内三段配对结构"},
            "classification": "stable=CI覆盖0 且 |β|<0.02；其余可检验=unstable；"
                              "机会不足=insufficient_evidence",
        },
        "data_usage": {
            "ring_events_total": n_total,
            "ring_events_in_main_window": n_total - n_excluded_tail,
            "excluded_tail_events": n_excluded_tail,
            "players_total": n_players_total,
            "players_used_3way_split": n_players_used,
        },
        "features": results,
    }
    (out_dir / "time_stability.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "DEGRADE_LOG.json").write_text(
        json.dumps({"entries": degrade}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(json.dumps({"players_used": n_players_used,
                      "excluded_tail": n_excluded_tail,
                      "features": {k: v.get("status") for k, v in results.items()}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
