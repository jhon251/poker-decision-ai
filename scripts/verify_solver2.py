"""SOLVER-2 第 5 步（理论锚点复测）：锚定 judge vs solver 策略。

与 SOLVER-1 的 verify_solver1.py **同一协议**（任务书要求 894 配对同协议）：
  * 对每个已解 spot：solver private_cards_actor 均匀采样 ≤6 手作 hero 底牌；
  * 构造 HU 状态（hero=OOP 行动者；to_call=0；pot/stack 来自 spot）；
  * 跑生产 judge 全链（RealtimeRunner；SOLVER-2 锚定核心 ON）；
  * judge 动作+尺寸映射回 solver 动作行（同名 + 最近尺寸）；
  * 一致 = judge 动作 ∈ solver 显著动作集（freq ≥ 10%）；
    EV 损失 = best_ev − ev(judge 动作)（求解器逐动作 EV，bb）。

SOLVER-2 增量口径：
  * 每决策记录 provenance（solver_hit / exploit_deviation / preflop_chart /
    solver_miss_fallback），并输出分桶统计；
  * 报告同时切 **r1 原 154 spot**（与 894 配对基线可比）与 **r2 扩库新增**
    两个切片；
  * 分片断点续跑：--shard i --num-shards n 各自落盘，--merge 合并
    （本机环境会外杀长跑 python，长验收必须分片）。

输出：docs/solver_r2/theory_verification.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SPOT_DIR = ROOT / "data" / "solver" / "spots_r1"
SIGNIFICANT_FREQ = 0.10
MAX_HANDS_PER_SPOT = 6
#: r1 原 154 spot 的 id 集合（与 894 配对基线可比的切片）
_R1_IDS: set[str] | None = None


def r1_ids(spot_dir: pathlib.Path) -> set[str]:
    """r1 原 154 spot = r1 时代已解的 id（按 id 结构无法判定 → 用
    docs/solver_r1/theory_verification.json 记录的 spot_id 集合）。
    兜底：manifest r2 参数中带 bet_sizes 覆盖的 2 个硬 flop 视为 r1。
    """
    global _R1_IDS
    if _R1_IDS is not None:
        return _R1_IDS
    ref = ROOT / "docs" / "solver_r1" / "theory_verification.json"
    ids: set[str] = set()
    if ref.is_file():
        data = json.loads(ref.read_text(encoding="utf-8"))
        ids = {r["spot_id"] for r in data.get("top10_divergences", [])}
    if not ids:
        # 兜底 1：全部 r1 spot id 形如 {SRP,3BP}-{flop,turn,river}-...
        # 且 r2 扩展只新增 turn 第 3/4 卡与 river 组合——无法从 id 判定，
        # 退化为"逐 spot 与 r1 manifest 对比"不可得时全算 r2。
        pass
    _R1_IDS = ids
    return ids


def _r1_spot_ids_from_solved_list(spot_dir: pathlib.Path) -> set[str]:
    """r1 已解 154 = r1 manifest 备份（data/solver/spots_r1_r1backup/
    manifest.json）。SOLVER-2 求解前把 r1 manifest 备份于此。"""
    p = spot_dir.parent / "spots_r1_r1backup" / "manifest.json"
    if p.is_file():
        m = json.loads(p.read_text(encoding="utf-8"))
        return set(m.get("solved_ids") or [])
    return set()


def run(spot_dir: pathlib.Path, out_path: pathlib.Path,
        max_spots: int | None, hero, shard: int, num_shards: int) -> dict:
    manifest = json.loads((spot_dir / "manifest.json").read_text(encoding="utf-8"))
    solved_ids = manifest["solved_ids"]
    if max_spots:
        solved_ids = solved_ids[:max_spots]
    if num_shards > 1:
        solved_ids = solved_ids[shard::num_shards]
    rows = []
    t0 = time.perf_counter()
    for idx, sid in enumerate(solved_ids):
        rec = json.loads((spot_dir / f"{sid}.json").read_text(encoding="utf-8"))
        q = rec.get("query") or {}
        cards_actor = rec.get("private_cards_actor") or []
        if not cards_actor:
            continue
        n_hands = len(cards_actor)
        step = max(1, n_hands // MAX_HANDS_PER_SPOT)
        sampled = cards_actor[::step][:MAX_HANDS_PER_SPOT]
        s_actions = [dict(a, name=str(a.get("name") or "").lower())
                     for a in (rec.get("actions") or [])]
        best_ev = max((a.get("ev_bb") or -1e9) for a in s_actions)
        significant = {a["name"] for a in s_actions
                       if a.get("frequency", 0.0) >= SIGNIFICANT_FREQ}
        argmax = max(s_actions, key=lambda a: a["frequency"])["name"]
        for entry in sampled:
            c1, c2 = entry[:2], entry[2:4]      # CLI 输出 '5c4c' 形式
            state = {
                "hand_id": f"verify2-{sid}-{c1}{c2}",
                "game_type": "cash", "street": q["street"],
                "board": list(q["board"]),
                "pot": round(q["pot_bb"] * 2.0, 3),
                "current_bet": 0.0, "to_call": 0.0,
                "min_raise_to": round(q["pot_bb"], 3),
                "sb": 1.0, "bb": 2.0, "ante": 0.0,
                "players": [
                    {"player_id": "hero", "seat": 1,
                     "stack": round(q["effective_stack_bb"] * 2.0, 3),
                     "status": "active",
                     "hole_cards": [c1, c2], "position": "BB",
                     "is_hero": True},
                    {"player_id": "seat2", "seat": 2,
                     "stack": round(q["effective_stack_bb"] * 2.0, 3),
                     "status": "active", "hole_cards": [],
                     "position": "BTN/SB", "is_hero": False}],
                "button_seat": 1, "hero_id": "hero",
            }
            try:
                out = hero.run_judge(state, hand_id=state["hand_id"])
            except Exception as exc:                      # noqa: BLE001
                rows.append({"spot_id": sid, "hand": [c1, c2],
                             "error": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            action = out.get("action")
            size = out.get("recommended_size_bb")
            prov = out.get("provenance") or {}
            cand = [a for a in s_actions if a["name"] == action]
            if action in ("bet", "raise", "allin") and cand and size:
                cand = sorted(cand, key=lambda a: abs(
                    (a.get("amount_bb") or 0.0) - float(size)))
            ev_judge = cand[0].get("ev_bb") if cand else None
            freq = cand[0].get("frequency") if cand else None
            ev_loss = (round(best_ev - ev_judge, 4)
                       if ev_judge is not None else None)
            agree = (action in significant) if ev_judge is not None else None
            rows.append({
                "spot_id": sid, "street": q["street"], "hand": [c1, c2],
                "judge_action": action, "judge_size_bb": size,
                "solver_argmax": argmax,
                "solver_action_ev_bb": ev_judge,
                "solver_action_freq": freq,
                "best_ev_bb": round(best_ev, 4),
                "ev_loss_bb": ev_loss,
                "agreement_significant": agree,
                "provenance_source": prov.get("source"),
                "anchored": prov.get("anchored"),
                "deviated": prov.get("deviated"),
                "lookup_mode": prov.get("lookup_mode"),
                "deviation_signal": ((prov.get("deviation") or {}).get("signal")
                                     if prov.get("deviation") else None),
                "fallback_conservative": out.get("fallback_conservative"),
            })
        if (idx + 1) % 20 == 0:
            print(f"  [{shard}/{num_shards}] {idx + 1}/{len(solved_ids)} spots",
                  flush=True)
    return {"rows": rows, "elapsed_sec": round(time.perf_counter() - t0, 1),
            "shard": shard, "num_shards": num_shards,
            "n_spots_in_shard": len(solved_ids)}


def summarize(rows: list[dict], label: str) -> dict:
    ok_rows = [r for r in rows if "error" not in r]
    per_street: dict[str, dict] = {}
    for r in ok_rows:
        st = per_street.setdefault(r["street"], {
            "n": 0, "agree": 0, "ev_loss_sum": 0.0, "ev_loss_n": 0})
        st["n"] += 1
        if r["agreement_significant"]:
            st["agree"] += 1
        if r["ev_loss_bb"] is not None:
            st["ev_loss_sum"] += r["ev_loss_bb"]
            st["ev_loss_n"] += 1
    for st in per_street.values():
        st["agreement_rate"] = (round(st["agree"] / st["n"], 4)
                                if st["n"] else None)
        st["ev_loss_avg_bb"] = (round(st["ev_loss_sum"] / st["ev_loss_n"], 4)
                                if st["ev_loss_n"] else None)
    overall_agree = sum(s["agree"] for s in per_street.values())
    overall_n = sum(s["n"] for s in per_street.values())
    ev_losses = [r["ev_loss_bb"] for r in ok_rows if r["ev_loss_bb"] is not None]
    prov_bucket: dict[str, int] = {}
    for r in ok_rows:
        k = str(r.get("provenance_source"))
        prov_bucket[k] = prov_bucket.get(k, 0) + 1
    divergent = sorted((r for r in ok_rows if r["ev_loss_bb"] is not None),
                       key=lambda r: -r["ev_loss_bb"])[:10]
    return {
        "label": label,
        "n_comparisons": len(ok_rows),
        "n_errors": len(rows) - len(ok_rows),
        "agreement_rate_significant_overall": (
            round(overall_agree / overall_n, 4) if overall_n else None),
        "ev_loss_avg_bb_overall": (round(sum(ev_losses) / len(ev_losses), 4)
                                   if ev_losses else None),
        "per_street": per_street,
        "provenance_buckets": prov_bucket,
        "n_deviated": sum(1 for r in ok_rows if r.get("deviated")),
        "top10_divergences": divergent,
    }


def merge(spot_dir: pathlib.Path, out_path: pathlib.Path,
          shard_files: list[pathlib.Path]) -> None:
    rows: list[dict] = []
    elapsed = 0.0
    for f in shard_files:
        d = json.loads(f.read_text(encoding="utf-8"))
        rows += d["rows"]
        elapsed += d.get("elapsed_sec", 0.0)
    r1 = _r1_spot_ids_from_solved_list(spot_dir)
    rows_r1 = [r for r in rows if r["spot_id"] in r1] if r1 else []
    rows_r2 = [r for r in rows if r["spot_id"] not in r1] if r1 else []
    result = {
        "protocol": ("同 verify_solver1 协议：spot 全覆盖 × 采样 ≤6 手；"
                     "一致 = judge 动作 ∈ solver 显著动作（freq≥10%）；"
                     "EV 损失 = best_ev − ev(judge动作)"),
        "strategy_core": "SOLVER-2 锚定核心 ON（base=solver 策略 + 有界偏离）",
        "n_spots_total": len({r["spot_id"] for r in rows if "error" not in r}),
        "elapsed_sec_shards": round(elapsed, 1),
        "overall": summarize(rows, "all_spots(r1+r2)"),
        "r1_slice_154": (summarize(rows_r1, "r1_original_154")
                         if rows_r1 else None),
        "r1_slice_note": (f"r1 基线切片 n_spots={len(r1 or set())}（备份 "
                          f"manifest），与 SOLVER-1 894 配对口径可比"
                          if r1 else "r1 备份 manifest 缺失，切片不可用"),
        "r2_new_slice": (summarize(rows_r2, "r2_expanded_new")
                         if rows_r2 else None),
        "shards": [str(f) for f in shard_files],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    o = result["overall"]
    r1s = result["r1_slice_154"]
    print(json.dumps({
        "n_spots_total": result["n_spots_total"],
        "n_comparisons": o["n_comparisons"],
        "agreement": o["agreement_rate_significant_overall"],
        "ev_loss_bb": o["ev_loss_avg_bb_overall"],
        "provenance": o["provenance_buckets"],
        "r1_slice": ({k: r1s[k] for k in ("n_comparisons",
                                          "agreement_rate_significant_overall",
                                          "ev_loss_avg_bb_overall")}
                     if r1s else None),
    }, ensure_ascii=False, indent=1))
    print("written:", out_path)


def main() -> int:
    global _R1_IDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot-dir", default=str(DEFAULT_SPOT_DIR))
    ap.add_argument("--out", default=str(ROOT / "docs" / "solver_r2" /
                                         "theory_verification.json"))
    ap.add_argument("--max-spots", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    spot_dir = pathlib.Path(args.spot_dir)
    out_path = pathlib.Path(args.out)
    if args.merge:
        shard_files = sorted(out_path.parent.glob(
            out_path.name.replace(".json", "") + "_shard*.json"))
        merge(spot_dir, out_path, shard_files)
        return 0

    from external_harness.adapter.judge_adapter import JudgeHeroController
    hero = JudgeHeroController(
        base_config_path="configs/config.yaml",
        overrides={"realtime": {"deadline_ms": 120000,
                                "agent_timeout_ms": 60000},
                   "storage": {"dir": "data/solver_r2/repository"},
                   "app": {"logs_dir": "logs/solver_r2"}})
    res = run(spot_dir, out_path, args.max_spots, hero,
              args.shard, args.num_shards)
    if args.num_shards > 1:
        shard_out = out_path.parent / (
            out_path.name.replace(".json", "") + f"_shard{args.shard}.json")
    else:
        shard_out = out_path
    shard_out.parent.mkdir(parents=True, exist_ok=True)
    shard_out.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    ok_rows = [r for r in res["rows"] if "error" not in r]
    agrees = sum(1 for r in ok_rows if r["agreement_significant"])
    losses = [r["ev_loss_bb"] for r in ok_rows if r["ev_loss_bb"] is not None]
    print(f"shard {args.shard}/{args.num_shards}: n={len(ok_rows)} "
          f"agree={agrees / max(1, len(ok_rows)):.4f} "
          f"ev_loss={sum(losses) / max(1, len(losses)):.4f}bb "
          f"({res['elapsed_sec']}s) -> {shard_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
