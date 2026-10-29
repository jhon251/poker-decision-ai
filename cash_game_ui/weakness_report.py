"""长期弱点报告：聚合 data/play_history 全部会话 → 文本诊断报告。

只统计真实发生的对局事实与 AI 参考一致性（教学展示口径），
不做 EV 承诺、不做实时辅助。数据源为 play_history v1+ JSONL。
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict

from poker_decision_ai.play.history import load_session

_STREET_CN = {"preflop": "翻前", "flop": "翻牌", "turn": "转牌", "river": "河牌"}
_POS_BY_SIZE = {
    2: ("BTN/SB", "BB"), 3: ("BTN", "SB", "BB"),
    4: ("BTN", "SB", "BB", "UTG"),
    5: ("BTN", "SB", "BB", "UTG", "CO"),
    6: ("BTN", "SB", "BB", "UTG", "MP", "CO"),
}
_BLOCKS = "▁▂▃▄▅▆▇█"


def _spark(values: list[float]) -> str:
    """累计净收益曲线的文字迷你图（归一化到正负区间）。"""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    out = []
    for v in values:
        idx = int(round((v - lo) / span * (len(_BLOCKS) - 1)))
        out.append(_BLOCKS[max(0, min(len(_BLOCKS) - 1, idx))])
    return "".join(out)


def _hero_position(button: int, table_size: int) -> str:
    labels = _POS_BY_SIZE.get(table_size, _POS_BY_SIZE[6])
    return labels[(0 - button) % table_size]


def build_report(history_dir: str = "data/play_history",
                 source: str = "play") -> str:
    files = []
    if os.path.isdir(history_dir):
        files = sorted(f for f in os.listdir(history_dir)
                       if f.startswith("play_session_") and f.endswith(".jsonl"))
    if not files:
        return ("暂无数据：还没有完成过任何对打会话。\n"
                "打完一个会话后这里会生成你的画像与弱点诊断。")

    totals = {"sessions": 0, "hands": 0, "net": 0.0, "net_raked": 0.0,
              "decisions": 0, "adv": 0, "agree": 0}
    pos_stat: dict[str, dict] = defaultdict(lambda: {"hands": 0, "net": 0.0})
    street_dec: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "adv": 0, "agree": 0})
    hero_acts: dict[str, Counter] = defaultdict(Counter)
    adv_acts: dict[str, Counter] = defaultdict(Counter)
    worst: list[tuple[float, dict]] = []
    cum: list[float] = []
    net_running = 0.0

    for fname in files:
        try:
            data = load_session(os.path.join(history_dir, fname))
        except Exception:                         # noqa: BLE001
            continue
        if data["head"].get("source", "play") != source:
            continue                              # 只统计指定来源的会话
        totals["sessions"] += 1
        n_seats = int(data["head"].get("table_size", 6))
        for h in data["hands"]:
            if h.get("net_bb") is None:
                continue                          # 中止手不计入统计
            totals["hands"] += 1
            totals["net"] += h["net_bb"]
            totals["net_raked"] += h.get("net_bb_raked") or 0.0
            net_running += h["net_bb"]
            cum.append(net_running)
            pos = _hero_position(int(h["button"]), n_seats)
            pos_stat[pos]["hands"] += 1
            pos_stat[pos]["net"] += h["net_bb"]
            raked = h.get("net_bb_raked") or 0.0
            if raked < 0:
                worst.append((raked, h))
            for d in h.get("hero_decisions") or []:
                street = str(d.get("street") or "?")
                totals["decisions"] += 1
                street_dec[street]["n"] += 1
                hero_acts[street][str(d.get("action") or "?")] += 1
                adv = d.get("advisor") or {}
                if adv.get("available"):
                    totals["adv"] += 1
                    street_dec[street]["adv"] += 1
                    adv_acts[street][str(adv.get("action") or "?")] += 1
                    if adv.get("action") == d.get("action"):
                        totals["agree"] += 1
                        street_dec[street]["agree"] += 1

    if not totals["hands"]:
        return "暂无数据：历史会话里没有完成的手（全部中止或为空会话）。"

    L: list[str] = []
    src_label = ("AI 对打数据" if source == "play"
                 else "外部牌谱（CoinPoker 导入）")
    L.append(f"══════ 长期弱点报告 · {src_label} ══════")
    L.append("（教学展示口径，非 EV 承诺）")
    L.append(f"会话 {totals['sessions']} 个 ｜ 完成 {totals['hands']} 手 ｜ "
             f"累计 {totals['net']:+.1f}bb（含抽水 {totals['net_raked']:+.1f}bb）")
    done_n = max(1, totals["hands"])
    L.append(f"百手收益 {totals['net'] / done_n * 100:+.1f} bb/100 ｜ "
             f"hero 决策 {totals['decisions']} 次 ｜ "
             f"AI 参考一致 {totals['agree']}/{totals['adv']}"
             f"（{totals['agree'] / max(1, totals['adv']) * 100:.0f}%）")
    L.append("")
    L.append("── 按位置（净收益 bb ｜ 手数 ｜ 均值）──")
    for pos, st in sorted(pos_stat.items(), key=lambda kv: kv[1]["net"]):
        mean = st["net"] / max(1, st["hands"])
        mark = " ← 最需改进" if pos == min(
            pos_stat.items(), key=lambda kv: kv[1]["net"])[0] else ""
        L.append(f"  {pos:6s}  {st['net']:+7.1f}bb  {st['hands']:4d} 手  "
                 f"均值 {mean:+.2f}bb{mark}")
    L.append("")
    L.append("── 按街：AI 参考一致率 + 动作分布对比 ──")
    for street in ("preflop", "flop", "turn", "river"):
        st = street_dec.get(street)
        if not st or not st["n"]:
            continue
        agree_s = (f"{st['agree'] / st['adv'] * 100:.0f}%" if st["adv"] else "—")
        L.append(f"  {street:8s} 决策 {st['n']:3d} ｜ 一致率 {agree_s}")
        ha = hero_acts.get(street) or Counter()
        aa = adv_acts.get(street) or Counter()
        ha_tot = max(1, sum(ha.values()))
        aa_tot = max(1, sum(aa.values()))
        for a in sorted(set(ha) | set(aa)):
            L.append(f"    {a:8s} 你 {ha.get(a, 0) / ha_tot * 100:5.1f}%"
                     f"   AI 参考 {aa.get(a, 0) / aa_tot * 100:5.1f}%")
    L.append("")
    L.append("── 动作分布解读提示 ──")
    tips = []
    for street in ("preflop", "flop", "turn", "river"):
        ha, aa = hero_acts.get(street) or Counter(), adv_acts.get(street) or Counter()
        ha_tot, aa_tot = max(1, sum(ha.values())), max(1, sum(aa.values()))
        if "fold" in ha and ha["fold"] / ha_tot > (aa.get("fold", 0) / aa_tot + 0.10):
            tips.append(f"{_STREET_CN[street]}：你的弃牌率明显高于 AI 参考——"
                        f"可能被对手下注打跑的次数偏多")
        if "call" in ha and ha["call"] / ha_tot > (aa.get("call", 0) / aa_tot + 0.10):
            tips.append(f"{_STREET_CN[street]}：你的跟注率偏高——检查是否在用弱牌追牌")
    L.extend(f"  · {t}" for t in tips[:4] if t)
    if not tips:
        L.append("  · 暂无显著系统性偏差（样本继续积累中）")
    L.append("")
    L.append("── 最痛的五手（含抽水净亏损最大）──")
    for raked, h in sorted(worst, key=lambda x: x[0])[:5]:
        hero = " ".join(str(c) for c in h.get("hero_hole") or [])
        street_cn = _STREET_CN.get(h.get("street_reached") or "",
                                   h.get("street_reached"))
        L.append(f"  第{h['hand_index'] + 1}手 {hero}  "
                 f"到 {street_cn}  底池 {h.get('pot_chips', 0):.0f}ch  "
                 f"净 {raked:+.1f}bb（{h.get('terminal_reason')}）")
    L.append("")
    L.append("── 累计净收益曲线（每手）──")
    L.append("  " + _spark(cum) + f"  最新 {cum[-1]:+.1f}bb")
    L.append("")
    L.append("样本积累越多越准；本报告只对比「你 vs AI 参考」，"
             "绝不混淆两者记录。")
    return "\n".join(L) + "\n"
