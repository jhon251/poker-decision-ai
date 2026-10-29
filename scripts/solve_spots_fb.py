"""M4 Phase A1 — facing-bet spot 套件（hero 面对下注的 call/fold/raise 锚点）。

背景（M3 结构性发现 a）：spot 库只有 to_call=0 首入节点 → 有界偏离的
call/fold 通道结构性缺失。本脚本用**同一个真实 CFR CLI**（已扩展
`export_nodes`：树内节点级均衡策略导出）为 spot 库补充 facing-bet 节点。

原理：单街树从「该街根节点（to_call=0，oop 先行动）」起解——树内天然包含
facing-bet 节点（对手下注后的 Fold/Call/Raise 决策点）。解一次树，按声明的
行动索引路径（export_nodes）导出这些节点的均衡策略/EV（与根节点同一聚合
口径），落盘为独立 spot 记录（node=facing_bet）。

每个求解导出 3 个节点（无牌面/尺寸歧义，声明式）：
  * canonical_canon  —— 典型 c-bet 面对点：SRP=[check,bet1]（BB 面对开注者
    c-bet）/ 3BP=[bet1]（跟注方面对 3bet 者 c-bet）；
  * canonical_large  —— 同线的大尺寸（bet2，几何尺寸）版本；
  * inverse_donk     —— 反角色 donk 面对点：SRP=[bet1]（开注者面对 donk）/
    3BP=[check,bet1]（3bet 者面对 donk）。

bet 尺寸桶（查询端匹配键）：bet_ratio = to_call / (pot − to_call)，
[0.45, 0.80] → mid；> 0.80 → large。本套件只解这两桶（池 bot 常用
0.5–0.66 pot 下注）；桶外查询诚实 miss，绝不跨桶套用。

用法：
  python scripts/solve_spots_fb.py --emit-config   # 生成 configs/solver/spots_fb.yaml
  python scripts/solve_spots_fb.py --config ... [--workers 3]  # 断点续跑
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
DEFAULT_OUT = ROOT / "data" / "solver" / "spots_fb"

# 与 r1/r2 套件同源的声明（范围/参数逐字复用——同一均衡上下文）
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
# turn/river 只取 4 个纹理 × 1 张后继卡（成本边界；flop 全 10 纹理）
FB_FLOP_TEX = [t for t, _ in FLOP_TEXTURES]
FB_TEX_LATE = ["wet_twotone", "dry_rainbow", "monotone", "connected"]
FB_TURN_CARD = {"wet_twotone": "Qd", "dry_rainbow": "2h",
                "monotone": "Qd", "connected": "2c"}
FB_RIVER_CARD = {"wet_twotone": "Kc", "dry_rainbow": "Ad",
                 "monotone": "Kc", "connected": "Ad"}
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
HARD_FLOP_ABSTRACTION = {"bet_sizes": "66%,a", "timeout_s": 2400}
HARD_FLOPS = {("SRP", "dry_rainbow"), ("SRP", "middle")}
#: 行动索引路径约定（树内确定性顺序：Check=0，Bet 按声明尺寸序，AllIn 最后）
CANON_PATH = {"SRP": [0, 1], "3BP": [1], "LIMP": [0, 1]}          # 典型下注面对点
CANON_PATH_LARGE = {"SRP": [0, 2], "3BP": [2], "LIMP": [0, 2]}    # 大尺寸/全下
INVERSE_PATH = {"SRP": [1], "3BP": [0, 1], "LIMP": [1]}           # 反角色 donk 面

# ── LIMP 族（R6 补：被动池的典型局面）────────────────────────────────
# HU 平跟池（SB complete、BB check）是 vs fish 的主导情形——没有翻前加注，
# SRP/3BP 的角色约定（跟注方/3bet 方）不适用。套件声明：根底池 1.5bb；
# oop = BB（翻后先行动方）、ip = SB；双方范围 = 声明式宽范围（PRIOR_
# CONSENSUS 式简化，见套件 note）。
LIMP_POT = {"pot_bb": 1.5, "effective_stack_bb": 98.5}
LIMP_RANGES = {
    "oop": "22+,A2s+,K2s+,Q4s+,J6s+,T6s+,96s+,86s+,76s,65s,54s,"
           "A2o+,K8o+,Q9o+,J9o+,T9o",
    "ip": "22+,A2s+,K2s+,Q4s+,J6s+,T6s+,95s+,85s+,75s,64s,53s+,"
          "A2o+,K9o+,Q9o+,J9o+,T9o",
}
LIMP_FLOP_TEX = ["wet_twotone", "dry_rainbow", "connected", "middle",
                 "paired_low", "ace_high"]


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
    """facing-bet 套件：{SRP, 3BP} 全纹理 + LIMP 族（被动池典型局面）。"""
    spots = []
    for pot_name, pot_cfg in POT_TYPES.items():
        ranges = RANGE_SETS[pot_name]
        exports = [
            {"export_id": "canonical_canon", "path": CANON_PATH[pot_name]},
            {"export_id": "canonical_large", "path": CANON_PATH_LARGE[pot_name]},
            {"export_id": "inverse_donk", "path": INVERSE_PATH[pot_name]},
        ]
        flop_rows = []
        for tex_name, flop in FLOP_TEXTURES:
            extra = (dict(HARD_FLOP_ABSTRACTION)
                     if (pot_name, tex_name) in HARD_FLOPS else {})
            flop_rows.append({
                "spot_id": f"FB-{pot_name}-flop-{tex_name}",
                "street": "flop",
                "board": flop,
                "pot_bb": pot_cfg["pot_bb"],
                "effective_stack_bb": pot_cfg["effective_stack_bb"],
                "ranges": ranges,
                **STREET_PARAMS["flop"],
                "export_nodes": [e["path"] for e in exports],
                "exports": exports,
                "family": pot_name,
                **extra,
            })
        for tex_name in FB_TEX_LATE:
            flop = dict(FLOP_TEXTURES)[tex_name]
            turn = FB_TURN_CARD[tex_name]
            river = FB_RIVER_CARD[tex_name]
            flop_rows.append({
                "spot_id": f"FB-{pot_name}-turn-{tex_name}-{turn}",
                "street": "turn",
                "board": board_cards(flop, turn, None),
                "pot_bb": pot_cfg["pot_bb"],
                "effective_stack_bb": pot_cfg["effective_stack_bb"],
                "ranges": ranges,
                **STREET_PARAMS["turn"],
                "export_nodes": [e["path"] for e in exports],
                "exports": exports,
                "family": pot_name,
            })
            flop_rows.append({
                "spot_id": f"FB-{pot_name}-river-{tex_name}-{turn}{river}",
                "street": "river",
                "board": board_cards(flop, turn, river),
                "pot_bb": pot_cfg["pot_bb"],
                "effective_stack_bb": pot_cfg["effective_stack_bb"],
                "ranges": ranges,
                **STREET_PARAMS["river"],
                "export_nodes": [e["path"] for e in exports],
                "exports": exports,
                "family": pot_name,
            })
        spots.extend(flop_rows)
    # ── LIMP 族（被动池典型局面；范围/参数声明见上方 LIMP_RANGES）──
    tex_map = dict(FLOP_TEXTURES)
    for tex_name in LIMP_FLOP_TEX:
        flop = tex_map[tex_name]
        exports = [
            {"export_id": "canonical_canon", "path": CANON_PATH["LIMP"]},
            {"export_id": "canonical_large", "path": CANON_PATH_LARGE["LIMP"]},
            {"export_id": "inverse_donk", "path": INVERSE_PATH["LIMP"]},
        ]
        spots.append({
            "spot_id": f"FB-LIMP-flop-{tex_name}",
            "street": "flop", "board": flop,
            "pot_bb": LIMP_POT["pot_bb"],
            "effective_stack_bb": LIMP_POT["effective_stack_bb"],
            "ranges": LIMP_RANGES,
            "max_iterations": 300, "target_exploitability_pot_pct": 1.5,
            "export_nodes": [e["path"] for e in exports],
            "exports": exports,
            "family": "LIMP",
            "timeout_s": 2400,
        })
    for tex_name in ("wet_twotone", "dry_rainbow"):
        flop = tex_map[tex_name]
        turn = FB_TURN_CARD[tex_name]
        river = FB_RIVER_CARD[tex_name]
        exports = [
            {"export_id": "canonical_canon", "path": CANON_PATH["LIMP"]},
            {"export_id": "canonical_large", "path": CANON_PATH_LARGE["LIMP"]},
            {"export_id": "inverse_donk", "path": INVERSE_PATH["LIMP"]},
        ]
        spots.append({
            "spot_id": f"FB-LIMP-turn-{tex_name}-{turn}",
            "street": "turn", "board": board_cards(flop, turn, None),
            "pot_bb": LIMP_POT["pot_bb"],
            "effective_stack_bb": LIMP_POT["effective_stack_bb"],
            "ranges": LIMP_RANGES,
            "max_iterations": 400, "target_exploitability_pot_pct": 1.0,
            "export_nodes": [e["path"] for e in exports],
            "exports": exports,
            "family": "LIMP",
        })
        spots.append({
            "spot_id": f"FB-LIMP-river-{tex_name}-{turn}{river}",
            "street": "river", "board": board_cards(flop, turn, river),
            "pot_bb": LIMP_POT["pot_bb"],
            "effective_stack_bb": LIMP_POT["effective_stack_bb"],
            "ranges": LIMP_RANGES,
            "max_iterations": 400, "target_exploitability_pot_pct": 0.75,
            "export_nodes": [e["path"] for e in exports],
            "exports": exports,
            "family": "LIMP",
        })
    ids = [s["spot_id"] for s in spots]
    assert len(set(ids)) == len(ids), "duplicate spot ids"
    return spots


def emit_config(path: pathlib.Path) -> None:
    spots = build_spots()
    doc = {
        "spot_suite": {
            "name": "M4-facing-bet-r2",
            "game": "HU 100bb cash — facing-bet nodes (node=facing_bet)",
            "cli": "tools/bin/postflop-bench-cli.exe",
            "solver": ("postflop-solver (b-inary) — 真实 CFR；HU；单街树从该街"
                       "根起解；export_nodes 导出树内 facing-bet 节点的均衡策略"),
            "n_spots": len(spots),
            "n_facing_nodes": len(spots) * 3,
            "families": ("SRP（开注+跟注，pot 3.5bb）/ 3BP（3bet+跟注，pot 22bb）"
                         " / LIMP（平跟池 pot 1.5bb，oop=BB、ip=SB，双方声明式"
                         "宽范围——vs 被动池的典型局面，R6 补族）"),
            "bet_bucket_note": ("bet_ratio = to_call/(pot−to_call)；mid∈[0.45,0.80]"
                                "（bet1=60% pot）、large>0.80（bet2=geometric/"
                                "shove）；桶外诚实 miss"),
            "seed_note": "求解确定性：同参数 ⇒ 同结果（solver 无随机）",
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
    for k in ("bet_sizes", "bet_sizes_ip", "raise_sizes", "export_nodes"):
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
            raw = json.loads(proc.stdout.decode("utf-8"))
            rec = {
                "spot_id": spot["spot_id"],
                "solved": bool(raw.get("solved")),
                "solver": raw.get("solver"),
                "street": spot["street"],
                "node": "facing_bet",
                "family": spot.get("family") or "",     # LIMP / SRP / 3BP
                "root_actions": raw.get("actions"),
                "exploitability_bb": raw.get("exploitability_bb"),
                "iterations": raw.get("iterations"),
                "exports": raw.get("node_exports"),
                "export_decl": spot.get("exports"),
                "simplifications": raw.get("simplifications"),
                "query": query,
                "wall_sec": wall,
            }
            if rec["solved"]:
                errs = [e for e in (rec.get("exports") or []) if e.get("error")]
                if errs:
                    rec["export_errors"] = errs
    except subprocess.TimeoutExpired:
        wall = round(time.perf_counter() - t0, 1)
        rec = {"spot_id": spot["spot_id"], "solved": False,
               "error": f"timeout after {wall}s", "wall_sec": wall}
    out_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    return {"spot_id": spot["spot_id"],
            "status": ("solved" if rec.get("solved") else "failed"),
            "wall_sec": wall}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-config", action="store_true")
    ap.add_argument("--config",
                    default=str(ROOT / "configs" / "solver" / "spots_fb.yaml"))
    ap.add_argument("--cli", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--solver-threads", type=int, default=4)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    if args.emit_config:
        emit_config(ROOT / "configs" / "solver" / "spots_fb.yaml")
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
        "cli_sha256": __import__("hashlib").sha256(
            pathlib.Path(cli).read_bytes()).hexdigest(),
        "node": "facing_bet",
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
            **{k: s[k] for k in ("bet_sizes", "export_nodes", "exports",
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
