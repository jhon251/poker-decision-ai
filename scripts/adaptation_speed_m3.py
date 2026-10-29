"""M3 — 在线适应速度验收（POOL_BOT_PLAN 第 5 条 / R5 M3 任务 2）。

对**隐藏参数**的池 bot（A 组原型，真值 hidden_context_truth 声明式暴露给
本验收脚本、对决策路径不可见）：会话个体先验（VillainSessionStore）逐手
累积该 villain 观测，读数须在 ~200 手内收敛。

阈值声明（写明理由）：
  * 指标：主要频率 |估计 − 真值| < 0.10；
  * 理由：二项采样在 n=100、p=0.5 时的 95% 置信半宽 = 1.96·√(0.25/100)
    ≈ 0.098 ≈ 0.10——即 0.10 ≈ 200 手时每个目标 context（期望 n≈100）
    的 2σ 采样噪声；再要求真值 ∈ Wilson95（区间覆盖口径）双门；
  * context 样本量 n<50 的（如 river 稀疏决策）不计入验收、如实单列。

用法：
  <repo>/.venv/Scripts/python.exe scripts/adaptation_speed_m3.py \
      [--hands 400] [--horizon 200] [--out docs/m3_opponent/adaptation_speed.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from external_harness.engine import fresh_nseat_state          # noqa: E402
from external_harness.protocol.bench_bots import bot_act       # noqa: E402
from external_harness.protocol.pool_bots import (              # noqa: E402
    FACTION_SPECS, hand_strength, hidden_context_truth,
)
from external_harness.view import guarded_player_view          # noqa: E402
from poker_decision_ai.agents.frequency_prior.session_store import (  # noqa: E402
    VillainSessionStore,
)
from poker_decision_ai.domain.poker import frequency_prior as fp  # noqa: E402

STARTING_STACK_CHIPS = 200
POSTFLOP = ("flop", "turn", "river")
#: 验收目标 context（postflop 两类主要频率）
TARGET_CONTEXTS = (("no_bet", "bet"), ("facing", "fold"))


def _estimate(store: VillainSessionStore, pid: str, *,
              street: str, facing_bet: bool, action: str) -> tuple:
    """会话个体读数：(估计率, wilson_low, wilson_high, n)。无样本 → None。"""
    obs = [o for o in store.observations(pid)
           if o.street == street and o.facing_bet is facing_bet]
    if not obs:
        return None, None, None, 0
    prior = fp.action_frequencies(
        obs, player_id=pid, facing_bet=facing_bet, street=street,
        config=fp.FrequencyPriorConfig(min_sample_size=1,
                                       half_life_days=1e9, wilson_z=1.96))
    f = (prior.frequencies or {}).get(action)
    if not f:
        return None, None, None, len(obs)
    return f["rate"], f["wilson_low"], f["wilson_high"], len(obs)


#: 紧原型（低每手决策率）的 sparring 用池鱼（同档软端原型）
SPAR_ARCH = {"nl2": "loose_passive_fish", "nl5": "fish"}
TIGHT_ARCH = ("semi_reg", "reg")


def run_session(stake: str, archetype: str, *, n_hands: int,
                seed_base: int, horizon: int, threshold: float = 0.10) -> dict:
    """villain(seat1)=池原型 bot（被观测对象，隐藏参数）；sparring(seat0)=
    bench loose_aggressive（软 villain）或同档池鱼 bot（紧 villain，声明式
    选择——保证紧原型每手 postflop 决策采样量）。

    逐手把整手动作喂进会话 store（只记 villain 观测），逐手记录读数曲线。
    """
    villain_id = f"VILLAIN_{stake}_{archetype}"
    spar_kind = ("pool_fish" if archetype in TIGHT_ARCH else "bench_lag")
    spar_id = f"SPAR_{stake}_{archetype}"
    store = VillainSessionStore(session_id=f"adapt-{stake}-{archetype}",
                                persist_dir=None, min_sample_size=20,
                                half_life_days=1e9, wilson_z=1.96)
    truth = hidden_context_truth(stake, archetype)
    curve: list[dict] = []
    for hand_index in range(n_hands):
        deal_seed = (seed_base + hand_index * 0x9E3779B1) & ((1 << 64) - 1)
        state = fresh_nseat_state(seed=deal_seed, n_seats=2,
                                  starting_stack=STARTING_STACK_CHIPS,
                                  button=hand_index % 2)
        rows: list[dict] = []
        step = 0
        while not state.terminal:
            actor = state.current_player
            view = guarded_player_view(state, seat=actor)
            if not view.get("ok"):
                break
            if actor == 1:      # villain（被观测对象）
                res = __import__(
                    "external_harness.protocol.pool_bots",
                    fromlist=["pool_bot_act"]).pool_bot_act(
                        view["view"], stake=stake, archetype=archetype,
                        seed=seed_base + 4242, hand_index=hand_index,
                        decision_index=step)
            else:
                # sparring（声明式选择）：软原型 villain ← bench 松凶
                # （制造面对下注压力）；紧原型 villain ← 池鱼 bot（limp/call
                # 多，保证 postflop 决策采样量——紧原型每手决策率低）。
                if spar_kind == "pool_fish":
                    from external_harness.protocol.pool_bots import (
                        pool_bot_act as _pba)
                    res = _pba(view["view"], stake=stake,
                               archetype=SPAR_ARCH[stake], seed=seed_base + 555,
                               hand_index=hand_index, decision_index=step)
                else:
                    res = bot_act(view["view"], kind="loose_aggressive",
                                  seed=seed_base + 999, hand_index=hand_index,
                                  decision_index=step)
            if not res.get("ok"):
                break
            rows.append({"street": str(state.street), "seat": actor,
                         "action": str(res["action"]),
                         "chips": res.get("amount_chips")})
            try:
                state = state.apply(res["action"], res["amount_chips"])
            except Exception:                    # noqa: BLE001
                break
            step += 1
            if step > 200:
                break
        # 逐手喂入：喂**整手**逐街动作行（facing_bet 由重放按全部玩家推导；
        # 只记录 villain 的观测——hero_id=sparring 座位被 record_hand 跳过）
        street_actions: dict[str, list[dict]] = {}
        for r in rows:
            row = {"player_id": villain_id if r["seat"] == 1 else spar_id,
                   "action": r["action"]}
            if r.get("chips") is not None:
                # 引擎语义：amount_chips = 结束后本街水平（raise-to）
                row["to" if r["action"] == "raise" else "amount"] = \
                    r["chips"]
            street_actions.setdefault(r["street"], []).append(row)
        store.record_hand(street_actions, hero_id=spar_id, ts=None)
        # 逐手读数（收敛判据按逐手曲线计算）
        snap = {"hand": hand_index + 1}
        for st in POSTFLOP:
            for ctx, act in TARGET_CONTEXTS:
                est, lo, hi, n = _estimate(
                    store, villain_id, street=st,
                    facing_bet=(ctx == "facing"), action=act)
                truth_key = ("no_bet_bet" if ctx == "no_bet"
                             else f"facing_bet_{act}")
                snap[f"{st}/{ctx}_{act}"] = {
                    "est": est, "wilson": [lo, hi], "n": n,
                    "truth": truth[st][truth_key],
                }
        curve.append(snap)

    # ---- 收敛判据（点态门 + first-touch 报告）----
    # 验收门（点态，非逐手单调——后者在 2σ 口径下每手各有 ~5% 假阴性，
    # 统计上不成立）：
    #   主要频率（n@200 ≥ 50 的 context）：hand 200 读数 |est−truth| < 0.10
    #   且真值 ∈ Wilson95(@200)；
    #   其余 n_final ≥ 50 的 context：终读数同门；
    #   measured 总数 ≥ 2（防"全稀疏 → 空真"）。
    # convergence_hand（信息性报告）= 首个 |err|<threshold 且 truth∈Wilson95 的手。
    keys = [k for k in curve[-1] if k != "hand"]
    per_context = []
    for k in keys:
        seq = [c[k] for c in curve]
        n_end = seq[-1]["n"] or 0
        truth = seq[-1]["truth"]
        at200 = next((c[k] for c in curve if c["hand"] == 200), None)
        conv_hand = next((i + 1 for i, d in enumerate(seq)
                          if d["est"] is not None and d["n"] >= 20
                          and abs(d["est"] - truth) < threshold
                          and d["wilson"][0] <= truth <= d["wilson"][1]),
                         None)
        entry = {
            "context": k, "truth": truth, "est_final": seq[-1]["est"],
            "wilson_final": seq[-1]["wilson"], "n_final": n_end,
            "n_at_200": (at200 or {}).get("n", 0) if at200 else 0,
            "convergence_hand_first_touch": conv_hand,
            "abs_err_final": (round(abs(seq[-1]["est"] - truth), 4)
                              if seq[-1]["est"] is not None else None),
        }
        if entry["n_at_200"] >= 50 and at200 and at200["est"] is not None:
            err200 = abs(at200["est"] - truth)
            cov200 = bool(at200["wilson"][0] <= truth <= at200["wilson"][1])
            entry.update({"gate": "hand200", "abs_err_gate": round(err200, 4),
                          "wilson_covers_gate": cov200,
                          "gate_pass": bool(err200 < threshold and cov200)})
        elif n_end >= 50 and seq[-1]["est"] is not None:
            cov_final = bool(seq[-1]["wilson"][0] <= truth
                             <= seq[-1]["wilson"][1])
            # 样本量匹配阈值：紧原型 context 在终点的 2σ 噪声（同 0.10 的
            # 推导公式 1.96·sqrt(0.25/n)——n<96 时宽于 0.10，下限仍 0.10）
            thr_n = max(threshold, 1.96 * (0.25 / n_end) ** 0.5)
            entry.update({"gate": "final", "threshold_gate": round(thr_n, 4),
                          "abs_err_gate": entry["abs_err_final"],
                          "wilson_covers_gate": cov_final,
                          "gate_pass": bool(entry["abs_err_final"] < thr_n
                                            and cov_final)})
        else:
            entry.update({"gate": "sparse(n<50)", "abs_err_gate": None,
                          "wilson_covers_gate": None, "gate_pass": None})
        per_context.append(entry)
    gated = [c for c in per_context if c["gate_pass"] is not None]
    g200 = [c for c in gated if c["gate"] == "hand200"]
    # 多次比较的诚实处理：24 个 gated context 在 95% 单门口径下期望 ~1 次
    # 尾部失守——验收 = gated 总通过率 ≥ 15/16 且 hand200 门（主要频率）
    # 全过、gated ≥ 2。
    n_pass = sum(1 for c in gated if c["gate_pass"])
    accepted = (len(gated) >= 2 and n_pass >= max(len(gated) - 1, 2)
                and (not g200 or all(c["gate_pass"] for c in g200)))
    return {"stake": stake, "archetype": archetype, "villain_id": villain_id,
            "n_hands": n_hands, "horizon": horizon, "threshold": threshold,
            "threshold_rationale": ("|est−truth|<0.10 ≈ 1.96·sqrt(0.25/100)"
                                    "——200 手时每 context 期望 n≈100 的 2σ"
                                    "采样噪声；另加真值∈Wilson95 覆盖门"),
            "acceptance": ("主要频率(n@200≥50)在 hand 200 点态达标"
                           "(threshold=0.10)；其余 n_final≥50 的 context 在"
                           "终点按样本量匹配阈值 max(0.10, 1.96·sqrt(0.25/n))"
                           "达标；通过率 ≥ 15/16（多次比较容差：尾部单点失守"
                           "允许 ≤1）；gated 总数 ≥2；n<50 稀疏 context 不计入"
                           "并如实单列"),
            "per_context": per_context,
            "curve_hand_marks": [c for c in curve if c["hand"] in
                                 (50, 100, 150, 200, 300, 400, n_hands)],
            "sparse_contexts_excluded": [c["context"] for c in per_context
                                         if c["gate"] != "hand200"
                                         and c["gate"] != "final"],
            "converged": bool(accepted)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=800)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=20261003)
    ap.add_argument("--stake", default=None)
    ap.add_argument("--archetype", default=None)
    ap.add_argument("--out",
                    default="docs/m3_opponent/adaptation_speed.json")
    args = ap.parse_args()

    jobs = []
    for stake in FACTION_SPECS:
        if args.stake and stake != args.stake:
            continue
        for arch in FACTION_SPECS[stake]["archetypes"]:
            # 代表性组合：每档的软端与硬端（fish/loose_passive_fish 与 reg）
            if arch not in ("fish", "loose_passive_fish", "reg"):
                continue
            if args.archetype and arch != args.archetype:
                continue
            jobs.append((stake, arch))

    t0 = time.perf_counter()
    sessions = [run_session(stake, arch, n_hands=args.hands,
                            seed_base=args.seed_base, horizon=args.horizon)
                for stake, arch in jobs]
    out = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
           "hands_per_session": args.hands, "horizon": args.horizon,
           "seed_base": args.seed_base,
           "all_converged": all(s["converged"] for s in sessions),
           "sessions": sessions,
           "wall_sec": round(time.perf_counter() - t0, 1)}
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    for s in sessions:
        det = "；".join(
            f"{c['context']}[{c['gate']}]:err={c['abs_err_gate']}"
            f"(n200={c['n_at_200']},first_touch={c['convergence_hand_first_touch']})"
            for c in s["per_context"] if c["gate_pass"] is not None)
        print(f"[{s['stake']}/{s['archetype']}] converged="
              f"{s['converged']} {det}")
    print(f"all_converged={out['all_converged']} -> {out_path}")
    return 0 if out["all_converged"] else 1


if __name__ == "__main__":
    sys.exit(main())
