"""SOLVER-1 — spot 套件生成 + 离线批量求解（真实 postflop solver）。

用法：
  python scripts/solve_spots.py --emit-config      # 生成 configs/solver/spots_r1.yaml
  python scripts/solve_spots.py --config configs/solver/spots_r1.yaml [--workers 4]
  python scripts/solve_spots.py --config ... --only <spot_id>

协议：每个 spot 调用 tools/bin/postflop-bench-cli.exe（真实 CFR 求解），
策略+EV 落盘 data/solver/spots_r1/{spot_id}.json + 覆盖率 manifest.json。
已解 spot 跳过（断点续跑）。同参数 ⇒ 同结果（CLI 确定性）。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CLI = str(ROOT / "tools" / "bin" / "postflop-bench-cli.exe")
DEFAULT_OUT = ROOT / "data" / "solver" / "spots_r1"

# ── 规范 spot 套件（HU 100bb cash；声明式参数）────────────────────────
FLOP_TEXTURES = [
    ("wet_twotone", ["Td", "9d", "6h"]),
    ("dry_rainbow", ["Ks", "7h", "2c"]),
    ("monotone", ["Ah", "Kh", "Qh"]),
    ("connected", ["9c", "8c", "7d"]),
    ("wet_twotone2", ["Qs", "9s", "4h"]),
    ("middle", ["Jc", "8d", "4h"]),
    ("paired_low", ["5c", "5h", "2d"]),
    ("broadway", ["Kd", "Qd", "Jc"]),
    ("dry_low", ["8h", "4c", "3d"]),
    ("ace_high", ["As", "7s", "2h"]),
]
TURNS = ["2c", "Qd", "9h", "As"]      # 每张 flop 配 turn（SOLVER-2 扩到 4 张）
RIVERS = ["Kc", "7s", "2d", "Qh", "Ad", "6c"]   # 每 (flop,turn) 配 river（SOLVER-2 扩到 6 张）
POT_TYPES = {
    "SRP": {"pot_bb": 3.5, "effective_stack_bb": 96.5},
    "3BP": {"pot_bb": 22.0, "effective_stack_bb": 89.0},
}
RANGE_SETS = {
    "SRP": {"oop": "QQ-22,AQs-AJs,KQs,QJs,JTs,T9s,98s,87s,76s,AQo",
            "ip": "22+,A2s+,K2s+,Q4s+,J6s+,T7s+,97s+,86s+,75s+,64s+,53s+,A2o+,K8o+,Q9o+,J9o+,T9o"},
    "3BP": {"oop": "JJ+,AJs+,A5s-A2s,KQs,JTs,T9s,98s,87s,AQo+",
            "ip": "QQ-22,AQs-A2s,ATo+,K5s+,KJo+,Q8s+,J8s+,T7s+,96s+,86s+,75s+,64s+,53s+"},
}
STREET_PARAMS = {
    "flop": {"max_iterations": 400, "target_exploitability_pot_pct": 1.0},
    "turn": {"max_iterations": 600, "target_exploitability_pot_pct": 0.75},
    "river": {"max_iterations": 600, "target_exploitability_pot_pct": 0.5},
}

# ── SOLVER-2 扩库声明 ─────────────────────────────────────────────────
# 覆盖扩展（重点 turn/river 大池节点——理论锚点 top 分歧类）：
#   * turn：SRP/3BP 均扩到 4 张 turn 卡；
#   * river：SRP/3BP 均扩到 4 turn × 6 river 组合。
# 2 个 SOLVER-1 超时 flop spot（SRP 宽范围全 3 街树）按任务书"减尺寸
# 抽象"补解：bet_sizes {60%, geometric, allin} → {66%, allin}（查询随
# manifest 如实记录；其余 spot 保持 r1 原抽象）。
HARD_FLOP_ABSTRACTION = {"bet_sizes": "66%,a", "timeout_s": 2400}
HARD_FLOPS = {("SRP", "dry_rainbow"), ("SRP", "middle")}
TURN_COVER = {"SRP": TURNS, "3BP": TURNS}
RIVER_TURNS = {"SRP": TURNS, "3BP": TURNS}
RIVER_CARDS = {"SRP": RIVERS, "3BP": RIVERS}


def board_cards(flop: list[str], turn: str | None, river: str | None) -> list[str]:
    used = set(flop) | ({turn} if turn else set()) | ({river} if river else set())
    assert len(used) == len(flop) + (1 if turn else 0) + (1 if river else 0), \
        f"card collision: {flop} {turn} {river}"
    out = list(flop)
    if turn:
        out.append(turn)
    if river:
        out.append(river)
    return out


def build_spots() -> list[dict]:
    """生成规范 spot 套件（SOLVER-2 r2：turn/river 扩库；无牌面冲突）。"""
    spots = []
    for pot_name, pot_cfg in POT_TYPES.items():
        ranges = RANGE_SETS[pot_name]
        for tex_name, flop in FLOP_TEXTURES:
            extra = (dict(HARD_FLOP_ABSTRACTION)
                     if (pot_name, tex_name) in HARD_FLOPS else {})
            spots.append({
                "spot_id": f"{pot_name}-flop-{tex_name}",
                "street": "flop",
                "board": flop,
                "pot_bb": pot_cfg["pot_bb"],
                "effective_stack_bb": pot_cfg["effective_stack_bb"],
                "ranges": ranges,
                **STREET_PARAMS["flop"],
                **extra,
            })
        for turn in TURN_COVER[pot_name]:
            for tex_name, flop in FLOP_TEXTURES:
                if turn in flop:
                    continue                     # 牌面冲突：turn 已在 flop 上
                spots.append({
                    "spot_id": f"{pot_name}-turn-{tex_name}-{turn}",
                    "street": "turn",
                    "board": board_cards(flop, turn, None),
                    "pot_bb": pot_cfg["pot_bb"],
                    "effective_stack_bb": pot_cfg["effective_stack_bb"],
                    "ranges": ranges,
                    **STREET_PARAMS["turn"],
                })
        for turn in RIVER_TURNS[pot_name]:
            for river in RIVER_CARDS[pot_name]:
                for tex_name, flop in FLOP_TEXTURES:
                    if turn in flop or river in flop or river == turn:
                        continue
                    spots.append({
                        "spot_id": (f"{pot_name}-river-{tex_name}-"
                                    f"{turn}{river}"),
                        "street": "river",
                        "board": board_cards(flop, turn, river),
                        "pot_bb": pot_cfg["pot_bb"],
                        "effective_stack_bb": pot_cfg["effective_stack_bb"],
                        "ranges": ranges,
                        **STREET_PARAMS["river"],
                    })
    ids = [s["spot_id"] for s in spots]
    assert len(set(ids)) == len(ids), "duplicate spot ids"
    return spots


def emit_config(path: pathlib.Path) -> None:
    spots = build_spots()
    doc = {
        "spot_suite": {
            "name": "SOLVER2-r2",
            "game": "HU 100bb cash",
            "cli": "tools/bin/postflop-bench-cli.exe",
            "solver": "postflop-solver (b-inary) — 真实 CFR；HU；尺寸抽象；从当前街起树",
            "n_spots": len(spots),
            "seed_note": "求解确定性：同参数 ⇒ 同结果（solver 无随机）",
            "r2_note": ("SOLVER-2 扩库：turn 4 卡 × SRP/3BP、river 4×6 组合 "
                        "× SRP/3BP（重点大池节点）；2 个 r1 超时 flop spot "
                        "按减尺寸抽象（bet_sizes=66%,a）补解"),
        },
        "spots": spots,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"emitted {len(spots)} spots -> {path}")


def solve_one(cli: str, spot: dict, out_dir: pathlib.Path,
              solver_threads: int = 4) -> dict:
    out_path = out_dir / f"{spot['spot_id']}.json"
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if prev.get("solved"):
                return {"spot_id": spot["spot_id"], "status": "cached"}
        except Exception:
            pass
    query = {k: spot[k] for k in ("street", "board", "pot_bb",
                                  "effective_stack_bb", "max_iterations",
                                  "target_exploitability_pot_pct")}
    query["oop_range"] = spot["ranges"]["oop"]
    query["ip_range"] = spot["ranges"]["ip"]
    # SOLVER-2：per-spot 尺寸抽象覆盖（如 2 个 r1 超时 flop spot 的
    # bet_sizes="66%,a"）；未声明 → CLI 缺省（=r1 抽象）。
    for k in ("bet_sizes", "bet_sizes_ip", "raise_sizes"):
        if spot.get(k):
            query[k] = spot[k]
    timeout_s = int(spot.get("timeout_s") or 2400)
    t0 = time.perf_counter()
    import os
    env = dict(os.environ)
    env["RAYON_NUM_THREADS"] = str(solver_threads)
    rec = None
    try:
        proc = subprocess.run([cli], input=json.dumps(query).encode("utf-8"),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout_s, env=env)
        wall = round(time.perf_counter() - t0, 1)
        if proc.returncode != 0:
            rec = {"spot_id": spot["spot_id"], "solved": False,
                   "error": proc.stderr.decode(errors="replace")[:300],
                   "wall_sec": wall}
        else:
            rec = json.loads(proc.stdout.decode("utf-8"))
            rec["spot_id"] = spot["spot_id"]
            rec["query"] = query
            rec["wall_sec"] = wall
    except subprocess.TimeoutExpired:
        wall = round(time.perf_counter() - t0, 1)
        rec = {"spot_id": spot["spot_id"], "solved": False,
               "error": f"timeout after {wall}s (slow spot; "
                        f"reduce abstraction or raise timeout)",
               "wall_sec": wall}
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return {"spot_id": spot["spot_id"],
            "status": ("solved" if rec.get("solved") else "failed"),
            "wall_sec": wall,
            "exploitability_bb": rec.get("exploitability_bb")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-config", action="store_true")
    ap.add_argument("--config", default=str(ROOT / "configs" / "solver" / "spots_r1.yaml"))
    ap.add_argument("--cli", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--solver-threads", type=int, default=4)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    if args.emit_config:
        emit_config(ROOT / "configs" / "solver" / "spots_r1.yaml")
        return 0

    cfg = json.loads(pathlib.Path(args.config).read_text(encoding="utf-8"))
    suite = cfg["spot_suite"]
    cli = args.cli or suite.get("cli") or DEFAULT_CLI
    out_dir = pathlib.Path(args.out_dir or DEFAULT_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)

    spots = cfg["spots"]
    if args.only:
        spots = [s for s in spots if s["spot_id"] == args.only]
    print(f"spots: {len(spots)} | cli: {cli} | out: {out_dir}", flush=True)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda s: solve_one(cli, s, out_dir,
                                                  args.solver_threads),
                              spots))
    solved = [r for r in results if r["status"] in ("solved", "cached")]
    failed = [r for r in results if r["status"] == "failed"]
    manifest = {
        "suite": suite["name"],
        "solver": suite["solver"],
        "cli": cli,
        "n_spots": len(spots),
        "n_solved": len(solved),
        "n_failed": len(failed),
        "solved_ids": [r["spot_id"] for r in solved],
        "failed": failed,
        "params": {s["spot_id"]: {
            **{k: s[k] for k in ("street", "board", "pot_bb",
                                 "effective_stack_bb", "ranges",
                                 "max_iterations",
                                 "target_exploitability_pot_pct")},
            **{k: s[k] for k in ("bet_sizes", "bet_sizes_ip", "raise_sizes",
                                 "timeout_s") if s.get(k)},
        } for s in spots},
        "elapsed_sec": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in
                      ("n_spots", "n_solved", "n_failed", "elapsed_sec")},
                     indent=1))
    print(f"manifest -> {out_dir / 'manifest.json'}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
