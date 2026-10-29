"""Phase 6C CI Boundary Audit（必做 3）：对 CI width ∈ [0.29, 0.31] 的记录用
Phase 5B 已验证的精确 Beta 分位数复核 reliability 最终分类。

只读复核，不修改画像、不修改阈值：
  reliable = samples>=30 AND shrinkage<=0.5 AND CI_width<=0.30
  marginal = samples>=15 AND shrinkage<=0.7
  otherwise unreliable

输出 data/uoft/models/phase6/ci_boundary_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from phase5b_math import beta_quantile  # noqa: E402
from phase6b_tiers import reliability_rule  # noqa: E402
from phase6c_profiles import pick_prior  # noqa: E402


def main() -> int:
    priors = json.load(open("data/uoft/models/phase6/population_priors.json",
                            encoding="utf-8"))
    stats = {}
    for line in open("data/uoft/stats/full_pool_suff_stats.jsonl",
                     encoding="utf-8"):
        r = json.loads(line)
        stats[r["player_id"]] = r

    lo_bound, hi_bound = 0.29, 0.31
    records = []
    n_boundary = 0
    n_flip = 0
    flip_dir = {}
    for line in open("data/uoft/models/phase6/profiles.jsonl", encoding="utf-8"):
        p = json.loads(line)
        r = stats[p["player_id"]]
        for name, f in p["features"].items():
            if name == "position_sensitivity" or f["zero_opportunity"]:
                continue
            width = f["ci95"][1] - f["ci95"][0]
            if not (lo_bound <= width <= hi_bound):
                continue
            n_boundary += 1
            S, n = r["features"][name]
            prior, _, _ = pick_prior(name, r["venue"], r["stakes_bucket"], priors)
            a, b = prior["a"] + S, prior["b"] + n - S
            elo = beta_quantile(a, b, 0.025)
            ehi = beta_quantile(a, b, 0.975)
            ewidth = ehi - elo
            rel_normal = f["reliability"]
            rel_exact = reliability_rule(f["samples"], f["shrinkage"], ewidth)
            rec = {
                "player_id": p["player_id"], "feature": name,
                "samples": f["samples"], "shrinkage": f["shrinkage"],
                "width_normal": round(width, 6), "width_exact": round(ewidth, 6),
                "reliability_normal": rel_normal,
                "reliability_exact": rel_exact,
                "flipped": rel_normal != rel_exact,
            }
            records.append(rec)
            if rel_normal != rel_exact:
                n_flip += 1
                flip_dir[f"{rel_normal}->{rel_exact}"] = \
                    flip_dir.get(f"{rel_normal}->{rel_exact}", 0) + 1

    out = {
        "band": [lo_bound, hi_bound],
        "n_boundary_records": n_boundary,
        "n_flipped": n_flip,
        "flip_direction": flip_dir,
        "note": "精确 CI 仅用于边界复核；画像未修改；阈值未调整",
        "records": records,
    }
    Path("data/uoft/models/phase6/ci_boundary_audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in
                      ("n_boundary_records", "n_flipped", "flip_direction")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
