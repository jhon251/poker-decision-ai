"""BENCH-1 驱动：一条命令复跑任一组对阵（同 seed ⇒ 同结果）。

用法：
  python scripts/bench_b1.py --config configs/bench/bench_r1.yaml \
      --matchup hero_vs_pivot [--hands 20000] [--workers 8]
  python scripts/bench_b1.py --config configs/bench/bench_r1.yaml --all

全部种子/手数/权重/覆盖 声明于 bench 配置文件（跑前固定，禁止中途修改）。
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


def _worker_run(job: dict) -> dict:
    """worker 进程：独立 runtime 跑一个 [hand_lo, hand_hi) 区间（确定性）。"""
    from external_harness.adapter.judge_adapter import (
        JudgeHeroController, build_population_freq_table)
    from external_harness.protocol.bench_match import aggregate, run_match

    overrides = dict(job["runtime_overrides"])
    fp_over = dict(overrides.get("frequency_prior") or {})
    fp_over["weight"] = job["weight"]
    overrides["frequency_prior"] = fp_over
    hero = JudgeHeroController(base_config_path=job["base_config"],
                               overrides=overrides)
    freq_table = (build_population_freq_table()["table"]
                  if job["bot_kind"] == "frequency_bot" else None)
    t0 = time.perf_counter()
    res = run_match(hero=hero, n_hands=job["n_hands"],
                    seed_base=job["seed_base"] + 0x9E3779B1 * job["hand_lo"],
                    bot_kind=job["bot_kind"], bot_seed=job["bot_seed"],
                    freq_table=freq_table,
                    bot_params=job.get("bot_params"),
                    progress_every=max(50, job["n_hands"] // 10),
                    log=lambda m: print(f"[{job['matchup']} w={job['weight']} "
                                        f"lo={job['hand_lo']}] {m}",
                                        flush=True))
    wall = time.perf_counter() - t0
    agg = aggregate(res["records"], res["latencies_ms"], res["counters"])
    return {"job": {k: job[k] for k in ("matchup", "bot_kind", "bot_seed",
                                        "weight", "hand_lo", "n_hands",
                                        "seed_base")},
            "aggregate": agg,
            "records": res["records"],
            "latencies": res["latencies_ms"],
            "wall_sec": round(wall, 1)}


def _merge(parts: list[dict]) -> dict:
    from external_harness.protocol.bench_match import aggregate
    records: list[dict] = []
    latencies: list[float] = []
    counters: dict[str, int] = {}
    for part in parts:
        records += part["records"]
        latencies += part.get("latencies") or []
        for k, v in part["aggregate"]["counters"].items():
            counters[k] = counters.get(k, 0) + v
    return aggregate(records, latencies, counters)


def main() -> int:
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bench/bench_r1.yaml")
    ap.add_argument("--matchup", default=None, help="只跑该 id 的对阵")
    ap.add_argument("--hands", type=int, default=None, help="覆盖每组手数")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--all", action="store_true", help="跑全部对阵（含消融）")
    # SOLVER-2：分段跑（本机外杀长跑 python；每段独立进程落盘，事后 merge）
    ap.add_argument("--hand-offset", type=int, default=0,
                    help="从该手索引起跑（seed 推导一致：deal_seed = seed_base"
                         " + hand_index*0x9E3779B1，hand_index 含 offset）")
    ap.add_argument("--out-suffix", default=None,
                    help="工件名后缀（如 s00 → {matchup}.s00.json；不覆盖主工件）")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))
    bench = cfg["bench"]
    n_hands = args.hands or int(bench["n_hands"])
    workers = max(1, args.workers or int(bench.get("workers") or 4))
    seed_base = int(bench["seed_base"])
    hand_offset = int(args.hand_offset or 0)

    matchups = list(bench["matchups"])
    if args.matchup:
        pool = (matchups + list(bench.get("ablation_matchups") or [])
                + list(bench.get("solver_matchups") or [])
                + list(bench.get("solver2_matchups") or [])
                + list(bench.get("m3_knob_matchups") or [])
                + list(bench.get("m4_knob_matchups") or [])
                + list(bench.get("r10_knob_matchups") or [])
                + list(bench.get("r10_diag_matchups") or [])
                + list(bench.get("r10_diag_matchups2") or []))
        matchups = [m for m in pool if m["id"] == args.matchup]
        if not matchups:
            print(f"no matchup {args.matchup}", file=sys.stderr)
            return 2
    elif args.all:
        matchups = (matchups + list(bench.get("ablation_matchups") or [])
                    + list(bench.get("solver_matchups") or [])
                    + list(bench.get("solver2_matchups") or []))

    out_dir = ROOT / "docs" / "bench_r1"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict = {}

    for m in matchups:
        weight = float(m.get("weight", bench["ablation"]["weight_main"]))
        chunk = n_hands // workers
        jobs = []
        for w in range(workers):
            hand_lo = hand_offset + w * chunk
            hand_hi = hand_offset + n_hands if w == workers - 1 \
                else hand_lo + chunk
            if hand_hi <= hand_lo:
                continue
            ro = json.loads(json.dumps(bench.get("runtime_overrides") or {}))
            ro.setdefault("frequency_prior", {})["weight"] = weight
            # SOLVER-1：matchup 级 runtime 覆盖（solver on/off 等）
            for k, v in (m.get("runtime_overrides") or {}).items():
                ro[k] = v
            ref_key = m.get("runtime_overrides_key")
            if ref_key:
                for k, v in (bench.get(ref_key) or {}).items():
                    ro[k] = v
            # 每 worker 隔离 storage：RealtimeRunner 每决策都持久化到
            # decisions.jsonl——共享同一路径会被 8 进程并发写坏（BENCH-1
            # 实测的 UnicodeDecodeError 根因）。隔离后互不干扰。
            st_dir = str(ro.get("storage", {}).get("dir", "data/repository"))
            ro.setdefault("storage", {})["dir"] = f"{st_dir}_w{w}"
            jobs.append({"matchup": m["id"], "bot_kind": m["bot_kind"],
                         "bot_seed": int(m["bot_seed"]), "weight": weight,
                         "hand_lo": hand_lo, "n_hands": hand_hi - hand_lo,
                         "seed_base": seed_base,
                         "base_config": bench["base_config"],
                         "runtime_overrides": ro,
                         "bot_params": dict(m.get("bot_params") or {})})
        print(f"=== {m['id']} (bot={m['bot_kind']}, weight={weight}, "
              f"n={n_hands}, workers={workers}) ===", flush=True)
        t0 = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(_worker_run, jobs))
        wall = time.perf_counter() - t0

        agg = _merge(parts)
        action_dist: dict = {}
        for part in parts:
            for k, v in part["aggregate"]["action_distribution"].items():
                action_dist[k] = action_dist.get(k, 0) + v
        agg["action_distribution"] = dict(sorted(action_dist.items()))
        agg["action_distribution_share"] = {
            k: round(v / max(1, sum(action_dist.values())), 4)
            for k, v in sorted(action_dist.items())}
        # SOLVER-2：分段合并需要原始延迟列表（merge_bench_segments.py 重算
        # P50/P90）；单文件体量 ≈ 手数 × 8B，可接受。
        agg["latencies_ms"] = [round(l, 1) for p in parts
                               for l in (p.get("latencies") or [])]
        agg["ablation_weight"] = weight
        agg["bot_kind"] = m["bot_kind"]
        agg["bot_seed"] = int(m["bot_seed"])
        agg["seed_base"] = seed_base
        agg["hand_offset"] = hand_offset
        agg["out_suffix"] = args.out_suffix
        agg["seed_derivation"] = ("deal_seed = (seed_base + hand_index*0x9E3779B1)"
                                  " mod 2^64; bot rng = f(bot_seed, seat, "
                                  "hand_index, decision_index)")
        agg["n_hands_requested"] = n_hands
        agg["wall_sec"] = round(wall, 1)
        agg["per_chunk"] = {
            str(p["job"]["hand_lo"]): {
                "n_hands": p["job"]["n_hands"],
                "bb_per_100": p["aggregate"]["bb_per_100"],
                "hero_total_bb": p["aggregate"]["hero_total_bb"],
                "n_completed": p["aggregate"]["n_hands_completed"],
                "p90_ms": p["aggregate"]["latency_ms_p90"]}
            for p in parts}
        all_results[m["id"]] = agg
        stem = m["id"] + (f".{args.out_suffix}" if args.out_suffix else "")
        (out_dir / f"{stem}.json").write_text(
            json.dumps(agg, ensure_ascii=False, indent=1), encoding="utf-8")
        (out_dir / f"{stem}.records.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False)
                      for p in parts for r in p["records"]) + "\n",
            encoding="utf-8")
        print(f"=== {m['id']}: bb/100={agg['bb_per_100']} "
              f"CI95={agg['ci95_bb_per_100']} n={agg['n_hands_completed']} "
              f"illegal={agg['counters'].get('hero_illegal_action', 0)} "
              f"({round(wall / 60, 1)} min)", flush=True)

    summary = {"bench": bench["name"],
               "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "config": args.config, "n_hands": n_hands,
               "seed_base": seed_base, "hand_offset": hand_offset,
               "out_suffix": args.out_suffix, "results": all_results}
    summary_name = ("summary." + args.out_suffix + ".json"
                    if args.out_suffix else "summary.json")
    (out_dir / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("summary ->", out_dir / summary_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
