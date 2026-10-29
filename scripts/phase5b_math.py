"""Phase 5B 数学工具（纯 Python，无第三方数值依赖）。

- wilson_interval: Wilson 比例区间（与冻结 stats.wilson_interval 同公式）
- betainc: 正则化不完全 Beta 函数（连分数，Numerical Recipes betacf）
- beta_quantile: Beta 分布分位数（二分求逆）
- pearson / spearman_brown
- bootstrap_pct: 玩家级 bootstrap 百分位 CI（种子可复现）

冻结边界：不 import 任何冻结模块（独立实现，供测试交叉验证用）。
"""
from __future__ import annotations

import math
import random


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def betainc(a: float, b: float, x: float, eps: float = 1e-14,
            max_iter: int = 400) -> float:
    """正则化不完全 Beta 函数 I_x(a,b)，连分数实现（Numerical Recipes betai）。"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    # 对称分支：保证连分数收敛（betacf 在 x 小于均值侧收敛）
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x, eps, max_iter) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x, eps, max_iter) / b


def _betacf(a: float, b: float, x: float, eps: float, max_iter: int) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def beta_quantile(a: float, b: float, p: float, max_iter: int = 100) -> float:
    """Beta(a,b) 的 p 分位数（二分）。a,b>0, 0<p<1。"""
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pearson(xs: list, ys: list) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def spearman_brown(r: float) -> float:
    """Split-half 校正到全长信度。r<=0 时无意义，返回原值。"""
    if r >= 1.0 or r <= 0.0:
        return r
    return 2.0 * r / (1.0 + r)


def bootstrap_pct(stat_fn, records: list, n_resample: int = 1000,
                  seed: int = 42, alpha: float = 0.05):
    """玩家级 bootstrap：重采样 records（长度保持），返回 (lo, hi, samples)。"""
    rng = random.Random(seed)
    k = len(records)
    stats = []
    for _ in range(n_resample):
        idx = [rng.randrange(k) for _ in range(k)]
        stats.append(stat_fn([records[i] for i in idx]))
    stats.sort()
    lo = stats[max(0, int(len(stats) * alpha / 2))]
    hi = stats[min(len(stats) - 1, int(len(stats) * (1 - alpha / 2)))]
    return lo, hi, stats


def beta_binomial_pmf(m: int, n: int, a: float, b: float) -> float:
    """Beta-Binomial(n, a, b) 的 PMF（log 域）。"""
    if m < 0 or m > n:
        return 0.0
    logp = (math.lgamma(n + 1) - math.lgamma(m + 1) - math.lgamma(n - m + 1)
            + math.lgamma(m + a) + math.lgamma(n - m + b)
            - math.lgamma(n + a + b)
            + math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
    return math.exp(logp)


def beta_binomial_quantile(a: float, b: float, n: int, p: float) -> int:
    """Beta-Binomial(n,a,b) 的 p 分位数（PMF 累加，n<=10000 可行）。"""
    target = p
    cdf = 0.0
    for m in range(n + 1):
        cdf += beta_binomial_pmf(m, n, a, b)
        if cdf >= target:
            return m
    return n


def beta_binomial_interval(a: float, b: float, n: int, level: float = 0.95
                           ) -> tuple[int, int]:
    """Beta-Binomial 预测区间（吸收未来 n 手观测的抽样噪声）。"""
    alpha = (1.0 - level) / 2.0
    lo = beta_binomial_quantile(a, b, n, alpha)
    hi = beta_binomial_quantile(a, b, n, 1.0 - alpha)
    return lo, hi


def mean_log_loss(post_mean: float, n_future: int, s_future: int) -> float:
    """常数预测 p 在 future 手上的平均对数损失。p∈(0,1)，n>0。"""
    p = min(max(post_mean, 1e-12), 1.0 - 1e-12)
    return -(s_future * math.log(p) + (n_future - s_future) * math.log(1.0 - p)) / n_future


def brier(post_mean: float, n_future: int, s_future: int) -> float:
    p = min(max(post_mean, 0.0), 1.0)
    return (s_future * (1.0 - p) ** 2 + (n_future - s_future) * p ** 2) / n_future
