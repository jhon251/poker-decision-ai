"""M4 Phase B3-B5 — GATE 战役驱动（GATE-MIN $1/$2 / GATE-1 $2/$5）。

协议（任务书冻结）：
  * 6-max；hero=seat0（button 轮转）；对手 = B 组解封参数（同包络、
    OS 熵扰动种子/尺寸——解封须先经 `scripts/unseal_group_b_m4.py`
    哈希核验 5b040e47…，解封动作记录进进度档案）；
  * 桌面构成（hero 占据构成中一席，其余 5 席为 bot）：
      - GATE-MIN（nl2，娱乐重仓）：2 fish + 2 recreational + 1 semi_reg
        （hero 顶替构成中的 reg 席）；
      - GATE-1（nl5，常规客重仓）：1 fish + 1 recreational + 2 semi_reg
        + 1 reg（hero 顶替构成中 2 reg 席之一）；
  * ≥100,000 手，分段落盘（断点续跑；分段期间禁止改任何配置/参数）；
  * 三口径逐手记录（rake-free / net@nfd / net@full）+ provenance 分布。

用法（后台）：
  <repo>/.venv/Scripts/python.exe scripts/run_m4_gate_campaign.py \
      --campaign gate_min > .tmp/gate_min_driver.log 2>&1
  <repo>/.venv/Scripts/python.exe scripts/run_m4_gate_campaign.py \
      --campaign gate_min --summarize
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UNSEALED_PATH = ROOT / "configs" / "bench" / "pool_factions_group_b.unsealed.json"

#: 冻结的战役协议（跑前固定；分段期间禁止修改）
CAMPAIGNS: dict[str, dict] = {
    "gate_min": {
        "stake": "nl2",
        "n_hands": 100_000,
        "segment_hands": 2_000,
        "seed_base": 20261101,
        "workers": 4,  # R6 内存纪律：8 worker 曾致 OOM 重启；一律 4
        "seats": {  # hero=seat0；构成：2 fish + 2 rec + 1 semi_reg
            1: {"stake": "nl2", "archetype": "loose_passive_fish"},
            2: {"stake": "nl2", "archetype": "loose_passive_fish"},
            3: {"stake": "nl2", "archetype": "recreational"},
            4: {"stake": "nl2", "archetype": "recreational"},
            5: {"stake": "nl2", "archetype": "semi_reg"},
        },
        "composition_note": ("POOL_BOT_PLAN 构成 6 席含 hero（6-max）："
                             "hero 顶替 reg 席 → 对手 5 席为娱乐重仓"),
    },
    "gate_1": {
        "stake": "nl5",
        "n_hands": 100_000,
        "segment_hands": 2_000,
        "seed_base": 20261201,
        "workers": 4,  # R6 内存纪律：8 worker 曾致 OOM 重启；一律 4
        "seats": {  # hero=seat0；构成：1 fish + 1 rec + 2 semi_reg + 1 reg
            1: {"stake": "nl5", "archetype": "fish"},
            2: {"stake": "nl5", "archetype": "recreational"},
            3: {"stake": "nl5", "archetype": "semi_reg"},
            4: {"stake": "nl5", "archetype": "semi_reg"},
            5: {"stake": "nl5", "archetype": "reg"},
        },
        "composition_note": ("POOL_BOT_PLAN 构成 6 席含 hero（6-max）："
                             "hero 顶替 2 reg 席之一 → 常规客重仓"),
    },
    # R10（KNOB+GATE1）：GATE-1 判决战役的执行实例。协议与 R6 冻结的 gate_1
    # 定义一致（$2/$5、CoinPoker 费率、1 fish + 2 recreational + 2 semi_reg
    # + 2 reg 中 hero 顶替一席 reg）；seed_base 换新值避免与 gate_min/
    # gate_1 占位定义同种子；gate_1 从未落盘任何段，本定义是唯一执行体。
    # 布局纪律：全战役统一 8 worker（分段执行器 run_m4_gate_segment_once
    # 的常量；段 ~25-33 min < ~45 min 外杀窗口；跨段禁止改布局——bot rng
    # 消费 worker 内相对 hand_index）。
    "gate_gate1": {
        "stake": "nl5",
        "n_hands": 100_000,
        "segment_hands": 2_000,
        "seed_base": 20261301,
        "workers": 8,
        "budget_profile": "strong",  # 冻结快照无此 campaign 条目 → 显式声明
        "pool_params_provenance": (
            "PRIOR_CONSENSUS：$2/$5 阵容 VPIP/PFR/3bet 目标值取公开统计共识"
            "（POOL_BOT_PLAN 冻结 v1），非本项目实测；B 组种子/扰动尺寸按 M3"
            " 封存协议解封（哈希 5b040e47…）"),
        "seats": {  # hero=seat0；构成：1 fish + 1 rec + 2 semi_reg + 1 reg
            1: {"stake": "nl5", "archetype": "fish"},
            2: {"stake": "nl5", "archetype": "recreational"},
            3: {"stake": "nl5", "archetype": "semi_reg"},
            4: {"stake": "nl5", "archetype": "semi_reg"},
            5: {"stake": "nl5", "archetype": "reg"},
        },
        "composition_note": ("POOL_BOT_PLAN 构成 6 席含 hero（6-max）：1 fish"
                             " + 2 recreational + 2 semi_reg + 2 reg，hero"
                             " 顶替 2 reg 席之一 → 常规客重仓"),
    },
}


def _load_unsealed() -> dict:
    if not UNSEALED_PATH.is_file():
        raise SystemExit(
            "B 组尚未解封：先跑 scripts/unseal_group_b_m4.py（哈希核验后落"
            " pool_factions_group_b.unsealed.json）")
    return json.loads(UNSEALED_PATH.read_text(encoding="utf-8"))


def _seat_configs(campaign: str) -> dict[int, dict]:
    """桌面座位配置：B 组种子 + B 组 bet_cbb + v2 模型基础参数。"""
    spec = CAMPAIGNS[campaign]
    unsealed = _load_unsealed()
    bots = unsealed["bots"]
    out: dict[int, dict] = {}
    for seat, comp in spec["seats"].items():
        key = f"{comp['stake']}/{comp['archetype']}"
        b = bots.get(key)
        if b is None:
            raise SystemExit(f"unsealed file missing {key}")
        from external_harness.protocol.pool_bots import archetype_params
        params = archetype_params(comp["stake"], comp["archetype"])
        params["bet_cbb"] = float(b["bet_cbb"])     # B 组扰动尺寸
        out[int(seat)] = {"bot_kind": "pool_bot",
                          "bot_seed": int(b["seed"]),   # B 组扰动种子
                          "bot_params": {"stake": comp["stake"],
                                         "archetype": comp["archetype"],
                                         "params": params}}
    return out


def _run_segment(job: dict) -> dict:
    """worker 进程：独立 runtime 跑 [hand_lo, hand_hi)（确定性）。"""
    from external_harness.adapter.judge_adapter import JudgeHeroController
    from external_harness.protocol.cash_match_6max import (
        aggregate_6max, run_match_6max)
    from external_harness.protocol.rake_model import RakeModel, load_rake_config

    campaign = job["campaign"]
    spec = CAMPAIGNS[campaign]
    # 预算档：冻结快照（若存在）> 本文件常量（两者都默认 strong）
    budget_profile = spec.get("budget_profile") or "strong"
    frozen = ROOT / "configs" / "bench" / "m4_gate_frozen.json"
    if frozen.is_file():
        snap = json.loads(frozen.read_text(encoding="utf-8"))
        budget_profile = (snap.get("campaigns", {}).get(campaign, {})
                          .get("pool_budget_profile") or budget_profile)
    hero = JudgeHeroController(overrides={
        "realtime": {"deadline_ms": 60000, "agent_timeout_ms": 30000},
        "storage": {"dir": f"data/m4_gate_{campaign}/repository_w{job['w']}"},
        "app": {"logs_dir": f"logs/m4_gate_{campaign}"},
        "strategy_core": {"pool_budget_profile": budget_profile},
    })
    rake = RakeModel(load_rake_config(), stake=spec["stake"])
    seats = {}
    for seat, cfg in job["seats"].items():
        seats[int(seat)] = {"bot_kind": "pool_bot",
                            "bot_seed": int(cfg["bot_seed"]),
                            "bot_params": cfg["bot_params"]}
    # 种子偏移（R6 启动前修正）：deal_seed = (seed_base + hand_index*0x9E3779B1)
    # mod 2^64，hand_index 为**绝对**手索引（= hand_lo + i）。原实现漏传
    # hand_lo → 所有 worker/所有段重放同一批手（smoke 实证：4 worker 逐手
    # 同值）。推导与 bench_b1 完全一致；修正发生在任何战役手数之前（冻结
    # 快照随之重落盘，见进度档案）。
    worker_seed_base = (spec["seed_base"]
                        + 0x9E3779B1 * job["hand_lo"]) & ((1 << 64) - 1)
    res = run_match_6max(hero=hero, n_hands=job["n_hands"],
                         seed_base=worker_seed_base,
                         stake=spec["stake"], seats=seats, rake=rake,
                         hero_seat=0,
                         progress_every=max(50, job["n_hands"] // 5),
                         log=lambda m: print(f"[{campaign} {job['seg']} "
                                             f"w={job['w']}] {m}",
                                             flush=True))
    agg = aggregate_6max(res["records"], res["latencies_ms"],
                         res["counters"], spec["stake"])
    return {"job": {k: job[k] for k in
                    ("campaign", "seg", "hand_lo", "n_hands", "seed_base")},
            "aggregate": agg,
            "records": res["records"],
            "latencies": res["latencies_ms"],
            "wall_sec": round(res.get("wall_sec") or 0, 1)}


def _merge(parts: list[dict], campaign: str) -> dict:
    from external_harness.protocol.cash_match_6max import aggregate_6max
    spec = CAMPAIGNS[campaign]
    records: list[dict] = []
    latencies: list[float] = []
    counters: dict[str, int] = {}
    for p in parts:
        records += p["records"]
        latencies += p.get("latencies") or []
        for k, v in p["aggregate"]["counters"].items():
            counters[k] = counters.get(k, 0) + v
    agg = aggregate_6max(records, latencies, counters, spec["stake"])
    agg["per_segment"] = {
        p["job"]["seg"]: {"hand_lo": p["job"]["hand_lo"],
                          "n_hands": p["job"]["n_hands"],
                          "bb_per_100_rakefree":
                              p["aggregate"]["bb_per_100_rakefree"],
                          "bb_per_100_net": p["aggregate"]["bb_per_100_net"],
                          "bb_per_100_net_fullrake":
                              p["aggregate"]["bb_per_100_net_fullrake"],
                          "n_aborted": p["aggregate"]["n_aborted"],
                          "p90_ms": p["aggregate"]["latency_ms_p90"]}
        for p in parts}
    return agg


def run_campaign(campaign: str) -> int:
    spec = CAMPAIGNS[campaign]
    out_dir = ROOT / "docs" / "m4_gate" / campaign
    out_dir.mkdir(parents=True, exist_ok=True)
    seats = _seat_configs(campaign)
    n_hands = spec["n_hands"]
    seg_h = spec["segment_hands"]
    workers = spec["workers"]
    n_segs = n_hands // seg_h
    plan = []
    for i in range(n_segs):
        seg = f"s{i:03d}"
        if (out_dir / f"{seg}.json").is_file() \
                and (out_dir / f"{seg}.records.jsonl").is_file():
            continue
        plan.append((seg, i * seg_h, seg_h))
    total = len(plan)
    print(f"[{campaign}] segments to run: {total}/{n_segs} "
          f"(hands={n_hands}, seg={seg_h})", flush=True)
    t0 = time.perf_counter()
    for i, (seg, off, n) in enumerate(plan, 1):
        chunk = n // workers
        jobs = []
        for w in range(workers):
            lo = off + w * chunk
            hi = off + n if w == workers - 1 else lo + chunk
            jobs.append({"campaign": campaign, "seg": seg, "w": w,
                         "hand_lo": lo, "n_hands": hi - lo,
                         "seed_base": spec["seed_base"], "seats": seats})
        print(f"[{campaign}] ({i}/{total}) {seg} off={off} n={n} "
              f"elapsed={round((time.perf_counter() - t0) / 60, 1)}min",
              flush=True)
        t_seg = time.perf_counter()
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                parts = list(ex.map(_run_segment, jobs))
        except Exception as exc:                 # noqa: BLE001
            print(f"[{campaign}]   !! segment failed: {exc} "
                  f"(rerun driver to retry)", flush=True)
            continue
        agg = _merge(parts, campaign)
        agg["seat_configs_public"] = {
            str(s): {"stake": c["bot_params"]["stake"],
                     "archetype": c["bot_params"]["archetype"]}
            for s, c in sorted(seats.items())}
        agg["composition_note"] = spec["composition_note"]
        agg["seed_derivation"] = ("deal_seed = (seed_base + "
                                  "hand_index*0x9E3779B1) mod 2^64; "
                                  "bot rng = f(bot_seed, seat, hand_index, "
                                  "decision_index)")
        (out_dir / f"{seg}.json").write_text(
            json.dumps(agg, ensure_ascii=False, indent=1), encoding="utf-8")
        (out_dir / f"{seg}.records.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False)
                      for p in parts for r in p["records"]) + "\n",
            encoding="utf-8")
        print(f"[{campaign}] {seg}: rakefree={agg['bb_per_100_rakefree']} "
              f"net={agg['bb_per_100_net']} "
              f"({round((time.perf_counter() - t_seg) / 60, 1)}min)",
              flush=True)
    done = sum(1 for i in range(n_segs)
               if (out_dir / f"s{i:03d}.json").is_file())
    print(f"[{campaign}] FINISHED {done}/{n_segs} segments "
          f"in {round((time.perf_counter() - t0) / 60, 1)}min", flush=True)
    return 0 if done == n_segs else 1


def summarize(campaign: str) -> int:
    spec = CAMPAIGNS[campaign]
    out_dir = ROOT / "docs" / "m4_gate" / campaign
    n_segs = spec["n_hands"] // spec["segment_hands"]
    parts = []
    for i in range(n_segs):
        seg = f"s{i:03d}"
        jf = out_dir / f"{seg}.json"
        rf = out_dir / f"{seg}.records.jsonl"
        if not jf.is_file() or not rf.is_file():
            print(f"missing {seg}", file=sys.stderr)
            return 2
        agg = json.loads(jf.read_text(encoding="utf-8"))
        records = [json.loads(ln) for ln in
                   rf.read_text(encoding="utf-8").splitlines() if ln.strip()]
        parts.append({"job": {"seg": seg,
                              "hand_lo": i * spec["segment_hands"],
                              "n_hands": len(records)},
                      "aggregate": agg,
                      "records": records,
                      "latencies": [],
                      "wall_sec": agg.get("wall_sec")})
    agg = _merge(parts, campaign)
    agg["campaign"] = campaign
    agg["seat_configs_public"] = {
        str(s): {"stake": c["stake"], "archetype": c["archetype"]}
        for s, c in sorted(CAMPAIGNS[campaign]["seats"].items())}
    agg["composition_note"] = spec["composition_note"]
    agg["protocol_frozen"] = {k: v for k, v in spec.items()
                              if k not in ("seats",)}
    agg["unsealed_file"] = str(UNSEALED_PATH.relative_to(ROOT))
    # 判决（TARGET_SPEC 冻结判据：净 bb/100 ≥ 5 且 95% CI 下界 > 0）
    net = agg.get("bb_per_100_net")
    ci = agg.get("ci95_net") or [None, None]
    agg["verdict"] = {
        "criterion": "净 bb/100 ≥ 5 且 95% CI 下界 > 0（TARGET_SPEC v3 冻结）",
        "net_bb_per_100": net,
        "ci95_lower": ci[0],
        "pass": bool(net is not None and ci[0] is not None
                     and net >= 5.0 and ci[0] > 0.0),
    }
    out_path = ROOT / "docs" / "m4_gate" / f"{campaign}_campaign.json"
    out_path.write_text(json.dumps(agg, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(json.dumps({"campaign": campaign,
                      "n_hands": agg["n_hands_completed"],
                      "rakefree": agg["bb_per_100_rakefree"],
                      "net": agg["bb_per_100_net"],
                      "net_fullrake": agg["bb_per_100_net_fullrake"],
                      "ci95_net": agg["ci95_net"],
                      "rake_total_bb": agg["rake_total_bb"],
                      "verdict": agg["verdict"]},
                     ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", choices=sorted(CAMPAIGNS))
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if not args.campaign:
        ap.error("--campaign required")
    if args.summarize:
        return summarize(args.campaign)
    return run_campaign(args.campaign)


if __name__ == "__main__":
    sys.exit(main())
