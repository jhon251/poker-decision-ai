"""批量分析 JSON → 单文件可视化 HTML 报告（零依赖；双击即开）。

用法：
  python scripts/render_report_html.py <分析.json> <输出.html>
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

STREET_CN = {"preflop": "翻前", "flop": "翻牌", "turn": "转牌", "river": "河牌"}
POS_ORDER = ("BTN", "SB", "BB", "UTG", "MP", "CO")
ACT_CN = {"fold": "弃牌", "check": "过牌", "call": "跟注",
          "bet": "下注", "raise": "加注"}
GREEN = "#2dc653"
AMBER = "#f5a623"
RED = "#e05656"
BLUE = "#4a90d9"


def _bar(pct: float, color: str) -> str:
    return (f'<div style="background:#1a2530;border-radius:4px;width:100%">'
            f'<div style="background:{color};width:{pct:.0f}%;'
            f'border-radius:4px;height:14px"></div></div>')


def build_html(data: dict) -> str:
    rows = [r for r in data["rows"] if r.get("status") in ("ok", "degraded")]
    summary = data.get("summary") or {}
    n_valid = len(rows)
    n_ai = sum(1 for r in rows if r.get("ai"))
    n_agree = sum(1 for r in rows if r.get("agree"))

    by_street: dict[str, list] = defaultdict(list)
    by_pos: dict[str, list] = defaultdict(list)
    dist_hero: Counter = Counter()
    dist_ai: Counter = Counter()
    disagree: list[dict] = []
    for r in rows:
        if not r.get("ai"):
            continue
        by_street[r["street"]].append(r)
        by_pos[r.get("position") or "?"].append(r)
        dist_ai[r["ai"]] += 1
        dist_hero[r["hero"]] += 1
        if r.get("agree") is False:
            disagree.append(r)
    disagree.sort(key=lambda r: str(r["street"]))

    cards = f"""
    <div class="cards">
      <div class="card"><div class="num">{data.get('n_hands', '—')}</div>
        <div class="lab">手数</div></div>
      <div class="card"><div class="num">{len(rows)}</div>
        <div class="lab">AI 有效决策</div></div>
      <div class="card"><div class="num">{n_agree / max(1, n_ai) * 100:.0f}%</div>
        <div class="lab">与 AI 一致率</div></div>
      <div class="card"><div class="num">{len(disagree)}</div>
        <div class="lab">不一致决策</div></div>
    </div>"""

    def street_rows_html():
        out = []
        for st in ("preflop", "flop", "turn", "river"):
            rs = by_street.get(st, [])
            if not rs:
                continue
            ag = sum(1 for r in rs if r.get("agree"))
            pct = ag / len(rs) * 100
            color = GREEN if pct >= 70 else (AMBER if pct >= 55 else RED)
            out.append(
                f'<tr><td>{STREET_CN[st]}</td><td>{len(rs)}</td>'
                f'<td style="width:45%">{_bar(pct, color)}</td>'
                f'<td>{pct:.0f}%</td></tr>')
        return "".join(out)

    def pos_rows_html():
        out = []
        items = [p for p in POS_ORDER if p in by_pos] + \
                [p for p in sorted(by_pos) if p not in POS_ORDER]
        for pos in items:
            rs = by_pos[pos]
            ag = sum(1 for r in rs if r.get("agree"))
            pct = ag / len(rs) * 100
            color = GREEN if pct >= 70 else (AMBER if pct >= 55 else RED)
            out.append(
                f'<tr><td>{pos}</td><td>{len(rs)}</td>'
                f'<td style="width:45%">{_bar(pct, color)}</td>'
                f'<td>{pct:.0f}%</td></tr>')
        return "".join(out)

    actions = ("fold", "check", "call", "bet", "raise")
    dist_rows = []
    ha_tot = max(1, sum(dist_hero[a] for a in actions))
    aa_tot = max(1, sum(dist_ai[a] for a in actions))
    for a in actions:
        hp = dist_hero[a] / ha_tot * 100
        ap = dist_ai[a] / aa_tot * 100
        dist_rows.append(
            f'<tr><td>{ACT_CN[a]}</td>'
            f'<td style="width:38%">{_bar(hp, "#4a90d9")}</td>'
            f'<td>{dist_hero[a]}</td><td>{hp:.0f}%</td>'
            f'<td style="width:38%">{_bar(ap, "#caa539")}</td>'
            f'<td>{dist_ai[a]}</td><td>{ap:.0f}%</td></tr>')

    dis_rows = []
    for r in disagree:
        st = STREET_CN.get(r["street"], r["street"])
        size = (f" {r['ai_size']:.1f}bb"
                if isinstance(r.get("ai_size"), (int, float)) else "")
        amt = (f" {r['amount']}ch"
               if isinstance(r.get("amount"), (int, float)) else "")
        src = r.get("source") or ""
        badge = ("solver" if src == "solver_hit" else
                 "图表" if src == "preflop_chart" else "降级")
        dis_rows.append(
            f'<tr data-street="{r["street"]}" data-pos="{r.get("position", "?")}">'
            f'<td>{r["hand"]}</td><td>{st}</td><td>{r.get("position", "?")}</td>'
            f'<td>{ACT_CN.get(r["hero"], r["hero"])}{amt}</td>'
            f'<td>{ACT_CN.get(r["ai"], r["ai"])}{size}</td>'
            f'<td><span class="badge b-{badge}">{badge}</span></td></tr>')

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>poker-ai 牌谱分析报告</title>
<style>
 body {{ background:#0f1419; color:#c8d6e5; font-family:"Segoe UI",sans-serif;
        margin:0; padding:24px; }}
 h1 {{ font-size:20px; }} h2 {{ font-size:15px; color:#88a0b8; margin:28px 0 10px; }}
 .cards {{ display:flex; gap:14px; margin:14px 0; }}
 .card {{ flex:1; background:#111820; border:1px solid #223140;
         border-radius:10px; padding:14px; text-align:center; }}
 .num {{ font-size:26px; font-weight:700; color:#fff; }}
 .lab {{ font-size:12px; color:#88a0b8; margin-top:4px; }}
 table {{ border-collapse:collapse; width:100%; font-size:13px; }}
 td, th {{ padding:6px 10px; text-align:left; border-bottom:1px solid #1d2a38; }}
 th {{ color:#88a0b8; font-weight:600; }}
 .badge {{ font-size:11px; padding:2px 8px; border-radius:999px; }}
 .b-solver {{ background:#12314f; color:#9cc3e8; }}
 .b-图表 {{ background:#123a26; color:#9fd3b4; }}
 .b-降级 {{ background:#4a2a1a; color:#f0b48a; }}
 .filters button {{ background:#1a2530; color:#c8d6e5; border:1px solid #223140;
                   border-radius:999px; padding:4px 14px; margin-right:6px;
                   cursor:pointer; font-size:12px; }}
 .filters button.on {{ background:#2dc653; color:#0f1419; font-weight:700; }}
 .foot {{ color:#5c7186; font-size:11px; margin-top:24px; }}
</style></head><body>
<h1>poker-ai 牌谱分析报告</h1>
<p style="color:#88a0b8;font-size:12px">教学展示口径：AI 建议来自产品内置决策引擎
（solver/图表/频率先验，未命中场景如实标注"降级"）；对比不构成 EV 承诺。</p>
{cards}
<h2>按街一致率</h2>
<table><tr><th>街</th><th>决策数</th><th></th><th>一致率</th></tr>{street_rows_html()}</table>
<h2>按位置一致率</h2>
<table><tr><th>位置</th><th>决策数</th><th></th><th>一致率</th></tr>{pos_rows_html()}</table>
<h2>动作分布对比：你 vs AI 参考</h2>
<table><tr><th>动作</th><th></th><th>你(次)</th><th>你(%)</th><th></th>
<th>AI(次)</th><th>AI(%)</th></tr>{''.join(dist_rows)}</table>
<h2>不一致决策（{len(disagree)} 条）</h2>
<div class="filters" id="filters">
 <button class="on" onclick="flt('all')">全部</button>
 <button onclick="flt('preflop')">翻前</button>
 <button onclick="flt('flop')">翻牌</button>
 <button onclick="flt('turn')">转牌</button>
 <button onclick="flt('river')">河牌</button>
</div>
<table id="dis"><tr><th>手号</th><th>街</th><th>位置</th><th>你</th>
<th>AI</th><th>来源</th></tr>{''.join(dis_rows)}</table>
<div class="foot">样本积累越多越准。不一致 ≠ 你错：部分建议来自"诚实降级"路径，
请结合具体牌局自行判断。</div>
<script>
function flt(s) {{
  const btns = document.querySelectorAll('#filters button');
  btns.forEach(b => b.classList.toggle('on', b.textContent.trim() ===
    (s === 'all' ? '全部' : document.querySelector(
     `#filters button[onclick="flt('${{s}}')"]`).textContent.trim())));
  document.querySelectorAll('#dis tr[data-street]').forEach(tr => {{
    tr.style.display = (s === 'all' || tr.dataset.street === s) ? '' : 'none';
  }});
}}
</script>
</body></html>"""


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    html = build_html(data)
    Path(sys.argv[2]).write_text(html, encoding="utf-8")
    print(f"HTML 报告: {sys.argv[2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
