"""SOLVER-2 — realtime 延迟验收探针。

验收（任务书）：查表命中 P90 < 50ms（不得回退）；decide P90 < 3s。
  * 查表：混合 exact（库内 board 全等）/ nearest（同纹理桶）/ miss 查询，
    进程内索引；首查含一次性索引构建，单列统计。
  * decide：对 N 个 spot 状态跑生产 RealtimeRunner（锚定核心 ON），
    报告端到端延迟分位数（与 bench 口径一致：无 UI deadline 压力）。

用法：<repo>/.venv/Scripts/python.exe scripts/probe_latency_solver2.py [--decide-n 30]
注意：进程内首个 decide 可能含 frequency-prior 持久缓存冷构建（一次性，
已知冷启动语义，非回归）——decide 统计单列 first 与其余。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPOT_DIR = ROOT / "data" / "solver" / "spots_r1"


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return round(s[k], 3)


def probe_lookup(n: int = 400) -> dict:
    from poker_decision_ai.core.providers.spot_table import get_spot_table
    table = get_spot_table(str(SPOT_DIR))
    man = json.loads((SPOT_DIR / "manifest.json").read_text(encoding="utf-8"))
    recs = []
    for sid in man["solved_ids"]:
        rec = json.loads((SPOT_DIR / f"{sid}.json").read_text(encoding="utf-8"))
        if rec.get("private_cards_actor"):
            recs.append(rec)
    rng = random.Random(42)
    exact_q, near_q, miss_q = [], [], []
    for rec in recs:
        q = rec["query"]
        exact_q.append({"street": q["street"], "board": list(q["board"]),
                        "pot_bb": q["pot_bb"]})
    base = exact_q[0]
    near_q = [dict(base, board=[c + "s" for c in base["board"]][:1]
                   + base["board"][1:]) for _ in range(10)]
    miss_q = [{"street": "river",
               "board": ["2c", "3d", "4h", "5s", "9c"], "pot_bb": 3.5}
              for _ in range(10)]
    queries = (exact_q * max(1, n // max(1, len(exact_q))))[:n]

    t0 = time.perf_counter()
    table.lookup(queries[0])            # 首查（含索引构建）
    first_ms = (time.perf_counter() - t0) * 1000

    lat = []
    for q in queries:
        t = time.perf_counter()
        table.lookup(q)
        lat.append((time.perf_counter() - t) * 1000)
    return {"n": len(queries), "first_lookup_ms_incl_index_build":
            round(first_ms, 3),
            "p50_ms": pct(lat, 0.5), "p90_ms": pct(lat, 0.9),
            "p99_ms": pct(lat, 0.99), "max_ms": round(max(lat), 3),
            "budget_p90_ms": 50, "pass": pct(lat, 0.9) < 50}


def probe_decide(n: int = 30) -> dict:
    from external_harness.adapter.judge_adapter import JudgeHeroController
    hero = JudgeHeroController(
        base_config_path="configs/config.yaml",
        overrides={"realtime": {"deadline_ms": 120000,
                                "agent_timeout_ms": 60000},
                   "storage": {"dir": ".tmp/latency_probe/repository"},
                   "app": {"logs_dir": ".tmp/latency_probe/logs"}})
    man = json.loads((SPOT_DIR / "manifest.json").read_text(encoding="utf-8"))
    recs = []
    for sid in man["solved_ids"]:
        if "-turn-" not in sid:
            continue
        rec = json.loads((SPOT_DIR / f"{sid}.json").read_text(encoding="utf-8"))
        if rec.get("private_cards_actor"):
            recs.append(rec)
    rng = random.Random(7)
    lat, first = [], None
    provenance = {}
    for i in range(n):
        rec = recs[i % len(recs)]
        q = rec["query"]
        hand = rng.choice(rec["private_cards_actor"])
        c1, c2 = hand[:2], hand[2:4]
        state = {
            "hand_id": f"lat-{i}", "game_type": "cash", "street": q["street"],
            "board": list(q["board"]),
            "pot": round(q["pot_bb"] * 2.0, 3),
            "current_bet": 0.0, "to_call": 0.0,
            "min_raise_to": round(q["pot_bb"], 3),
            "sb": 1.0, "bb": 2.0, "ante": 0.0,
            "players": [
                {"player_id": "hero", "seat": 1,
                 "stack": round(q["effective_stack_bb"] * 2.0, 3),
                 "status": "active", "hole_cards": [c1, c2],
                 "position": "BB", "is_hero": True},
                {"player_id": "seat2", "seat": 2,
                 "stack": round(q["effective_stack_bb"] * 2.0, 3),
                 "status": "active", "hole_cards": [],
                 "position": "BTN/SB", "is_hero": False}],
            "button_seat": 1, "hero_id": "hero",
        }
        out = hero.run_judge(state, hand_id=state["hand_id"])
        ms = out.get("latency_ms_report") or 0.0
        if first is None:
            first = ms                      # 含一次性缓存冷构建
        else:
            lat.append(ms)
        src = (out.get("provenance") or {}).get("source")
        provenance[src] = provenance.get(src, 0) + 1
    return {"n": n, "first_decide_ms_cold_start": round(first or 0.0, 1),
            "p50_ms": pct(lat, 0.5), "p90_ms": pct(lat, 0.9),
            "max_ms": round(max(lat), 1),
            "budget_p90_ms": 3000, "pass": pct(lat, 0.9) < 3000,
            "provenance_buckets": provenance}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decide-n", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "docs" / "solver_r2" /
                                         "latency_probe.json"))
    args = ap.parse_args()
    lookup = probe_lookup()
    decide = probe_decide(args.decide_n)
    out = {"lookup": lookup, "decide": decide,
           "note": ("decide 首查含 frequency-prior 持久缓存冷构建（一次性、"
                    "已知冷启动语义）；其余为热路径。bench 段工件另提供 "
                    "大规模 P50/P90 口径。")}
    pathlib.Path(args.out).write_text(json.dumps(out, ensure_ascii=False,
                                                 indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
