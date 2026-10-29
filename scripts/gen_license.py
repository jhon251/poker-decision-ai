"""激活码签发工具（发行方内部使用）。

用法：
  python scripts/gen_license.py                          # 打印本机机器码
  python scripts/gen_license.py --machine <机器码> --plan month    # 月付（30 天）
  python scripts/gen_license.py --machine <机器码> --plan quarter   # 季付（90 天）
  python scripts/gen_license.py --machine <机器码> --plan year     # 年付（365 天）
  python scripts/gen_license.py --machine <机器码> --plan permanent  # 永久
  python scripts/gen_license.py --machine <机器码> --plan year --tier pro
  python scripts/gen_license.py --machine <机器码> --days 7        # 自定义天数
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cash_game_ui.license import (  # noqa: E402
    make_license_key, machine_id, tier_label)

PLANS = {
    "month": ("月付", 30),
    "quarter": ("季付", 90),
    "year": ("年付", 365),
    "permanent": ("永久", None),
}
TIERS = {"base": "基础版", "pro": "专业版", "max": "大师版"}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="签发 poker-ai 激活码（月付/季付/年付/永久）")
    ap.add_argument("--machine", help="用户机器码（缺省打印本机机器码后退出）")
    ap.add_argument("--plan", choices=PLANS, help="订阅计划")
    ap.add_argument("--days", type=int, help="自定义有效天数（覆盖 --plan）")
    ap.add_argument("--tier", choices=TIERS, default="base",
                    help="功能档位（默认基础版）")
    ap.add_argument("--ref", default="",
                    help="代理标记（A-Z0-9_-，≤16 字符；分账用）")
    args = ap.parse_args()

    machine = (args.machine or machine_id()).strip().upper()
    if not args.machine:
        print(f"本机机器码：{machine}")
        print("签发示例：python scripts/gen_license.py "
              "--machine <机器码> --plan year")
        return 0
    if args.days is not None:
        days, plan_name = args.days, f"自定义 {args.days} 天"
    else:
        plan_name, days = PLANS.get(args.plan, ("月付", 30))
    key = make_license_key(machine, days=days, tier=args.tier, ref=args.ref)
    print(f"机器码：{machine}")
    print(f"计划：{plan_name} ｜ 档位：{TIERS[args.tier]}"
          + (f" ｜ 代理：{args.ref}" if args.ref else ""))
    print(f"激活码：{key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
