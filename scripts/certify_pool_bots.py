"""M3 — 池 bot 资格认证（POOL_BOT_PLAN 防自欺防线 1）。

每个原型 bot 先**自对弈**（同原型两座位，HU 引擎、脱敏视角、确定性种子）
实测统计画像；VPIP/PFR/3bet 全部落在声明区间内才算合格；认证数据落
docs/m3_opponent/certification.json（进 GATE 报告）。

用法：
  <repo>/.venv/Scripts/python.exe scripts/certify_pool_bots.py \
      [--hands 3000] [--stake nl2] [--archetype reg] [--out ...]

诚实边界：自对弈统计是 HU 语境的直接观测（"不是什么"：声明包络为 6-max
统计口径 PRIOR_CONSENSUS；HU 适配声明见 pool_bots.pool_bots_contract）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from external_harness.engine import fresh_nseat_state          # noqa: E402
from external_harness.protocol.pool_bots import (               # noqa: E402
    FACTION_SPECS, POOL_STAKES, declared_envelope, declared_post_envelope,
    hand_strength, in_envelope, postflop_strength, pool_bot_act,
    pool_bots_contract, selfplay_stats,
)
from external_harness.view import guarded_player_view           # noqa: E402

STARTING_STACK_CHIPS = 200
BIG_BLIND_CHIPS = 2


def _selfplay_worker(job: dict) -> dict:
    """同原型自对弈 n_hands（两座位同参数；bot rng 按座位分离）。"""
    stake, archetype = job["stake"], job["archetype"]
    n_hands, seed_base = int(job["n_hands"]), int(job["seed_base"])
    action_log: list[dict] = []
    aborted = 0
    for hand_index in range(n_hands):
        deal_seed = (seed_base + hand_index * 0x9E3779B1) & ((1 << 64) - 1)
        button = hand_index % 2
        state = fresh_nseat_state(seed=deal_seed, n_seats=2,
                                  starting_stack=STARTING_STACK_CHIPS,
                                  button=button)
        hid = f"{stake}-{archetype}-{hand_index:06d}"
        step = 0
        while not state.terminal:
            actor = state.current_player
            view = guarded_player_view(state, seat=actor)
            if not view.get("ok"):
                aborted += 1
                break
            res = pool_bot_act(view["view"], stake=stake, archetype=archetype,
                               seed=seed_base + 7919, hand_index=hand_index,
                               decision_index=step)
            if not res.get("ok"):
                aborted += 1
                break
            row = {"hand_id": hid, "seat": actor,
                   "street": str(state.street),
                   "action": str(res["action"]),
                   "button": button}
            if str(state.street) != "preflop":
                # M4：强度条件化认证——记录决策时点的强度（只看该座位
                # 自己的手牌+公共牌，与 bot 决策同输入）
                row["strength"] = round(postflop_strength(
                    [str(c) for c in state.hole_cards[actor]],
                    [str(c) for c in state.board]), 4)
            action_log.append(row)
            try:
                state = state.apply(res["action"], res["amount_chips"])
            except Exception:                    # noqa: BLE001
                aborted += 1
                break
            step += 1
            if step > 200:
                aborted += 1
                break
    return {"stake": stake, "archetype": archetype, "n_hands": n_hands,
            "aborted": aborted, "action_log": action_log}


def _postflop_checks(stake: str, archetype: str, stats: dict) -> dict:
    """M4 双指标之二：postflop 分街包络 + 强度条件性质。"""
    declared = declared_post_envelope(stake, archetype)
    checks: dict[str, dict] = {}
    ps = stats.get("per_street") or {}
    for metric, streets in (("aggr", declared.get("aggr") or {}),
                            ("call", declared.get("call") or {})):
        for st, env in streets.items():
            m = (ps.get(st) or {}).get(metric)
            checks[f"{metric}_{st}"] = {
                "declared": env, "measured": m,
                "pass": in_envelope(m, env)}
    sc = stats.get("strength_conditioned") or {}
    scd = declared.get("strength_conditioned") or {}
    # 汇总跨街的强度分桶（样本合并）
    def _agg(kind: str) -> tuple[float | None, float | None]:
        num = den = 0.0
        for st, buckets in sc.items():
            for bk, v in buckets.items():
                if v[kind] is None:
                    continue
                den += v["n_facing"]
                num += v[kind] * v["n_facing"]
        return (num / den, den) if den else (None, 0.0)

    weak_fold, weak_n = None, 0
    strong_call, strong_n = None, 0
    fold_num = fold_den = call_num = call_den = 0.0
    for st, buckets in sc.items():
        w = buckets.get("weak") or {}
        s = buckets.get("strong") or {}
        if w.get("fold") is not None and w.get("n_facing"):
            fold_num += w["fold"] * w["n_facing"]
            fold_den += w["n_facing"]
        if s.get("call") is not None and s.get("n_facing"):
            call_num += s["call"] * s["n_facing"]
            call_den += s["n_facing"]
    weak_fold = fold_num / fold_den if fold_den else None
    strong_call = call_num / call_den if call_den else None
    cap = scd.get("fold_weak_max")
    checks["fold_weak_max"] = {
        "declared": ["<=", cap], "measured": (round(weak_fold, 4)
                                              if weak_fold is not None else None),
        "n": int(fold_den),
        "pass": weak_fold is not None and weak_fold <= float(cap)}
    floor = scd.get("call_strong_min")
    checks["call_strong_min"] = {
        "declared": [">=", floor],
        "measured": (round(strong_call, 4)
                     if strong_call is not None else None),
        "n": int(call_den),
        "pass": strong_call is not None and strong_call >= float(floor)}
    # 单调性：call(weak) ≤ call(mid) ≤ call(strong)（±tol；跨街合并）
    if scd.get("monotonic_call"):
        tol = float(scd.get("monotonic_tol") or 0.06)
        calls = {}
        for bk in ("weak", "mid", "strong"):
            num = den = 0.0
            for st, buckets in sc.items():
                v = buckets.get(bk) or {}
                if v.get("call") is not None and v.get("n_facing"):
                    num += v["call"] * v["n_facing"]
                    den += v["n_facing"]
            calls[bk] = (num / den) if den else None
        ok = (calls["weak"] is not None and calls["mid"] is not None
              and calls["strong"] is not None
              and calls["weak"] <= calls["mid"] + tol
              and calls["mid"] <= calls["strong"] + tol)
        checks["monotonic_call"] = {
            "declared": ["weak<=mid<=strong", f"tol={tol}"],
            "measured": {k: (round(v, 4) if v is not None else None)
                         for k, v in calls.items()},
            "pass": ok}
    return checks


def _measure(job: dict) -> dict:
    res = _selfplay_worker(job)
    # 认证口径：只统计 seat 0 的决策（两座位同原型；对手行作 context）
    stats = selfplay_stats(res["action_log"], seat=0)
    declared = declared_envelope(res["stake"], res["archetype"])
    checks = {
        "vpip": {"declared": declared["vpip"], "measured": stats["vpip"],
                 "pass": in_envelope(stats["vpip"], declared["vpip"])},
        "pfr": {"declared": declared["pfr"], "measured": stats["pfr"],
                "pass": in_envelope(stats["pfr"], declared["pfr"])},
        "threebet": {"declared": declared["threebet"],
                     "measured": stats["threebet"],
                     "pass": in_envelope(stats["threebet"],
                                         declared["threebet"])},
    }
    # M4 双指标之二：postflop 包络 + 强度条件性质
    post_checks = _postflop_checks(res["stake"], res["archetype"], stats)
    checks.update(post_checks)
    return {"stake": res["stake"], "archetype": res["archetype"],
            "n_hands": res["n_hands"], "aborted": res["aborted"],
            "stats": stats, "checks": checks,
            "qualified": all(c["pass"] for c in checks.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=3000)
    ap.add_argument("--stake", default=None, choices=list(POOL_STAKES))
    ap.add_argument("--archetype", default=None)
    ap.add_argument("--seed-base", type=int, default=20261002)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="docs/m3_opponent/certification.json")
    args = ap.parse_args()

    jobs = []
    for stake in POOL_STAKES:
        for arch in FACTION_SPECS[stake]["archetypes"]:
            if args.stake and stake != args.stake:
                continue
            if args.archetype and arch != args.archetype:
                continue
            jobs.append({"stake": stake, "archetype": arch,
                         "n_hands": args.hands,
                         "seed_base": args.seed_base})

    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        results = list(ex.map(_measure, jobs))
    wall = time.perf_counter() - t0

    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "contract": pool_bots_contract(),
        "method": ("self-play（同原型双座位，HU 引擎 fresh_nseat_state + "
                   "guarded_player_view 脱敏视角 + 确定性种子）；"
                   "VPIP/PFR/3bet 按 6-max 统计口径的直接观测"),
        "hands_per_archetype": args.hands,
        "seed_base": args.seed_base,
        "wall_sec": round(wall, 1),
        "all_qualified": all(r["qualified"] for r in results),
        "results": results,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    for r in results:
        c = r["checks"]
        print(f"[{r['stake']}/{r['archetype']}] n={r['n_hands']} "
              f"vpip={c['vpip']['measured']}∈{c['vpip']['declared']}"
              f"{'✓' if c['vpip']['pass'] else '✗'} "
              f"pfr={c['pfr']['measured']}∈{c['pfr']['declared']}"
              f"{'✓' if c['pfr']['pass'] else '✗'} "
              f"3bet={c['threebet']['measured']}∈{c['threebet']['declared']}"
              f"{'✓' if c['threebet']['pass'] else '✗'} "
              f"qualified={r['qualified']}")
    print(f"all_qualified={out['all_qualified']} ({round(wall, 1)}s) -> "
          f"{out_path}")
    return 0 if out["all_qualified"] else 1


if __name__ == "__main__":
    sys.exit(main())
