"""SOLVER-1 第 6 步（理论锚点）：judge base_decision vs solver 策略。

对每个已解 spot：
  * 取 solver private_cards_actor 中**均匀采样的 ≤6 手**作为 hero 底牌；
  * 构造 HU 状态（hero = OOP 行动者；to_call=0；pot/stack 来自 spot）；
  * 跑生产 judge（base_decision 口径 = judge 链直接输出）；
  * 把 judge 动作+尺寸映射回 solver 动作行（同名 + 最近尺寸）；
  * 一致 = judge 动作 ∈ solver 显著动作集（freq ≥ 10%）；
    EV 损失 = best_ev − ev(judge 动作)（bb，来自求解器逐手 EV）。

输出：docs/solver_r1/theory_verification.json（分街一致率/EV 损失 +
top-10 分歧逐一列明）。
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


def run(spot_dir: pathlib.Path, out_dir: pathlib.Path,
        max_spots: int | None, hero) -> dict:
    manifest = json.loads((spot_dir / "manifest.json").read_text(encoding="utf-8"))
    solved_ids = manifest["solved_ids"]
    if max_spots:
        solved_ids = solved_ids[:max_spots]
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
        for entry in sampled:
            c1, c2 = entry[:2], entry[2:4]      # CLI 输出 '5c4c' 形式
            state = {
                "hand_id": f"verify-{sid}-{c1}{c2}",
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
            # 映射回 solver 动作行：同名 + 尺寸最近
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
                "solver_argmax": max(s_actions, key=lambda a: a["frequency"])[
                    "name"],
                "solver_action_ev_bb": ev_judge,
                "solver_action_freq": freq,
                "best_ev_bb": round(best_ev, 4),
                "ev_loss_bb": ev_loss,
                "agreement_significant": agree,
                "fallback_conservative": out.get("fallback_conservative"),
            })
        if (idx + 1) % 10 == 0:
            print(f"  {idx + 1}/{len(solved_ids)} spots", flush=True)

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
    divergent = sorted((r for r in ok_rows if r["ev_loss_bb"] is not None),
                       key=lambda r: -r["ev_loss_bb"])[:10]
    return {
        "n_spots": len(solved_ids),
        "n_comparisons": len(ok_rows),
        "n_errors": len(rows) - len(ok_rows),
        "agreement_rate_significant_overall": (
            round(overall_agree / overall_n, 4) if overall_n else None),
        "ev_loss_avg_bb_overall": (round(sum(r["ev_loss_bb"] for r in ok_rows
                                             if r["ev_loss_bb"] is not None)
                                         / max(1, sum(1 for r in ok_rows
                                                      if r["ev_loss_bb"]
                                                      is not None)), 4)),
        "per_street": per_street,
        "top10_divergences": divergent,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "judgement": "base_decision 口径 = judge 链直接输出（未含 deviation）",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spot-dir", default=str(DEFAULT_SPOT_DIR))
    ap.add_argument("--out", default=str(ROOT / "docs" / "solver_r1" /
                                         "theory_verification.json"))
    ap.add_argument("--max-spots", type=int, default=None)
    args = ap.parse_args()

    from external_harness.adapter.judge_adapter import JudgeHeroController
    hero = JudgeHeroController(
        base_config_path="configs/config.yaml",
        overrides={"realtime": {"deadline_ms": 120000,
                                "agent_timeout_ms": 60000},
                   "storage": {"dir": "data/solver_r1/repository"},
                   "app": {"logs_dir": "logs/solver_r1"}})
    spot_dir = pathlib.Path(args.spot_dir)
    out_dir = pathlib.Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run(spot_dir, out_dir, args.max_spots, hero)
    out = pathlib.Path(args.out)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "n_spots", "n_comparisons", "agreement_rate_significant_overall",
        "ev_loss_avg_bb_overall", "per_street")}, ensure_ascii=False, indent=1))
    print("written:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
