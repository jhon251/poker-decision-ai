"""EXE-1 — GATE-MIN 决策核心零回退冒烟（抽 5 段重跑对比）。

原理：GATE-MIN 战役（docs/m4_gate/gate_min/，10 万手）由 50 段 × 8 worker
驱动，worker 内第 i 手 deal_seed = (20261101 + (hand_lo+i)*0x9E3779B1)
mod 2^64，hero 决策 = 生产 judge 链（solver 锚定 + 频率先验），bot rng =
f(bot_seed, seat, hand_index(相对), decision_index)。

本冒烟对 5 个抽段（s000/s012/s024/s036/s049）各 worker **只重跑前 5 手**
（会话内 villain 先验从空开始——与原战役 worker 启动时相同；更深的手会
依赖原战役进程内累积的会话状态，超出"零回退冒烟"范围），逐手对比冻结
records（同 deal_seed）：hero_net_bb 与 hero_actions 序列（动作/尺寸/
provenance/node）必须**逐字段相等**。

隔离：storage/logs 写 smoke 目录，绝不触碰冻结仓库
（data/m4_gate_gate_min/repository_w* 与 logs/m4_gate_gate_min）。

用法：.venv/Scripts/python.exe scripts/gate_min_smoke_replay.py [--out json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED_BASE = 20261101
SEGMENTS = [0, 12, 24, 36, 49]
SEGMENT_HANDS = 2000
WORKERS = 8
PROBE_HANDS = 5
RECORDS_DIR = ROOT / "docs" / "m4_gate" / "gate_min"
SMOKE_STORAGE = "data/m4_gate_gate_min_smoke/repository_w{w}"
SMOKE_LOGS = "logs/m4_gate_gate_min_smoke"


def _frozen_index(seg: int) -> dict[int, dict]:
    """段冻结 records → {deal_seed: record}。"""
    path = RECORDS_DIR / f"s{seg:03d}.records.jsonl"
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[int(rec["deal_seed"])] = rec
    return out


def _run_probe(job: dict) -> list[dict]:
    """重跑一个 worker 的前 PROBE_HANDS 手（judge 配置与原战役同构）。"""
    import os
    os.chdir(str(ROOT))
    from external_harness.adapter.judge_adapter import JudgeHeroController
    from external_harness.protocol.cash_match_6max import run_match_6max
    from external_harness.protocol.rake_model import RakeModel, load_rake_config

    w = job["w"]
    hero = JudgeHeroController(overrides={
        "realtime": {"deadline_ms": 60000, "agent_timeout_ms": 30000},
        "storage": {"dir": SMOKE_STORAGE.format(w=w)},
        "app": {"logs_dir": SMOKE_LOGS},
        "strategy_core": {"pool_budget_profile": "strong"},
    })
    rake = RakeModel(load_rake_config(), stake="nl2")
    res = run_match_6max(hero=hero, n_hands=PROBE_HANDS,
                         seed_base=(SEED_BASE + 0x9E3779B1 * job["hand_lo"])
                         & ((1 << 64) - 1),
                         stake="nl2", seats=job["seats"], rake=rake,
                         hero_seat=0)
    out = []
    for i, rec in enumerate(res["records"]):
        out.append({
            "seg": job["seg"], "w": w, "abs_hand_index": job["hand_lo"] + i,
            "deal_seed": rec["deal_seed"],
            "hero_net_bb": rec.get("hero_net_bb"),
            "hero_net_bb_raked": rec.get("hero_net_bb_raked"),
            "hero_actions": rec.get("hero_actions"),
            "aborted": rec.get("aborted"),
        })
    return out


def _segment_workers(seg: int) -> int:
    """从 supervisor 日志读该段启动时声明的 worker 数（战役中途从 4→8）。

    bot rng 消费 worker 内相对 hand_index——worker 分区是逐手结果的一部分，
    必须按原段布局重放，否则对比失真（分区伪影，非核心回归）。
    """
    n = None
    log_path = ROOT / ".tmp" / "gate_min_supervisor.log"
    tag = f"s{seg:03d} start"
    if log_path.is_file():
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if tag in line and "workers=" in line:
                    try:
                        n = int(line.rsplit("workers=", 1)[1].strip())
                    except ValueError:
                        pass
    return n or WORKERS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / ".tmp" / "gate_min_smoke.json"))
    args = ap.parse_args()

    from scripts.run_m4_gate_campaign import _seat_configs
    seats = _seat_configs("gate_min")

    # 隔离卫生：清掉上次冒烟的 storage 残留（等价于"全新会话"重放）
    import shutil
    for w in range(WORKERS):
        shutil.rmtree(ROOT / "data" / "m4_gate_gate_min_smoke"
                      / f"repository_w{w}", ignore_errors=True)
    shutil.rmtree(ROOT / "logs" / "m4_gate_gate_min_smoke", ignore_errors=True)

    jobs = []
    seg_workers = {}
    for seg in SEGMENTS:
        n_w = _segment_workers(seg)
        seg_workers[seg] = n_w
        off = seg * SEGMENT_HANDS
        chunk = SEGMENT_HANDS // n_w
        for w in range(n_w):
            jobs.append({"seg": seg, "w": w, "hand_lo": off + w * chunk,
                         "seats": seats})

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        parts = list(ex.map(_run_probe, jobs))
    rerun = [r for part in parts for r in part]

    compared = matched = 0
    mismatches = []
    for seg in SEGMENTS:
        frozen = _frozen_index(seg)
        for r in (x for x in rerun if x["seg"] == seg):
            f = frozen.get(int(r["deal_seed"]))
            if f is None:
                mismatches.append({"seg": seg, "deal_seed": r["deal_seed"],
                                   "reason": "deal_seed_not_in_frozen"})
                continue
            compared += 1
            diffs = []
            if r["hero_net_bb"] != f.get("hero_net_bb"):
                diffs.append(f"net_bb {r['hero_net_bb']}!={f.get('hero_net_bb')}")
            if r["aborted"] != f.get("aborted"):
                diffs.append(f"aborted {r['aborted']}!={f.get('aborted')}")
            a_new = r["hero_actions"] or []
            a_old = f.get("hero_actions") or []
            if len(a_new) != len(a_old):
                diffs.append(f"n_actions {len(a_new)}!={len(a_old)}")
            else:
                for i, (x, y) in enumerate(zip(a_new, a_old)):
                    for k in ("a", "size_bb", "prov", "node", "mw", "dev"):
                        if x.get(k) != y.get(k):
                            diffs.append(f"action{i}.{k} "
                                         f"{x.get(k)}!={y.get(k)}")
            if diffs:
                mismatches.append({"seg": seg, "deal_seed": r["deal_seed"],
                                   "diffs": diffs[:6]})
            else:
                matched += 1

    report = {
        "contract": "exe1-gate-min-smoke-v1",
        "segments": SEGMENTS, "probe_hands_per_worker": PROBE_HANDS,
        "segment_workers": {str(k): v for k, v in seg_workers.items()},
        "n_compared": compared, "n_matched": matched,
        "zero_regression": compared > 0 and matched == compared,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
        "wall_sec": round(time.time() - t0, 1),
        "isolation": {"storage": SMOKE_STORAGE, "logs": SMOKE_LOGS,
                      "frozen_touched": False},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("n_compared", "n_matched", "zero_regression",
                       "n_mismatches", "wall_sec")}, ensure_ascii=False))
    return 0 if report["zero_regression"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
