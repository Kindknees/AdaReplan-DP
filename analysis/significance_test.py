"""
Pairwise significance test for two evaluation runs of the D3P replan agents.

Compares success rate and mean episode reward between two .npz outputs produced
by eval_d3p_replan_agent / eval_d3p_oracle_replan_agent (any agent that now
saves the per-episode arrays `episode_reward` and `episode_best_reward`).

For success rate (Bernoulli per episode):
  - Each method's point estimate + Wilson 95% CI
  - Two-proportion z-test (pooled SE) for the difference
  - Bootstrap 95% CI for the difference

For mean episode reward (continuous per episode):
  - Each method's mean + t-based 95% CI
  - Welch's t-test for the difference
  - Bootstrap 95% CI for the difference
  - Cohen's d effect size

Usage:
  python analysis/significance_test.py \
      $DPPO_LOG_DIR/.../eval_results_A.npz \
      $DPPO_LOG_DIR/.../eval_results_B.npz \
      --label-a "TD trigger" --label-b Oracle

Episodes are treated as independent samples (unpaired tests). With ~200
episodes per run this is statistically sound; the standard error of an
~0.5-success-rate method is ~0.035, so differences below ~0.07 are unlikely
to reach significance.
"""

import argparse
import numpy as np
from scipy import stats


def wilson_interval(k: int, n: int, confidence: float = 0.95):
    """Wilson score interval for a binomial proportion. More reliable than
    the normal approximation when p is close to 0 or 1, or n is moderate."""
    if n == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = k / n
    denom = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


def two_proportion_z_test(k1: int, n1: int, k2: int, n2: int):
    """Pooled two-proportion z-test. Returns (z, p_two_sided)."""
    if n1 == 0 or n2 == 0:
        return (0.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0.0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    p_two = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
    return float(z), float(p_two)


def bootstrap_diff_ci(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
):
    """Bootstrap CI for the difference in means: mean(A) − mean(B)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return (float("nan"), float("nan"))
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        ai = rng.integers(0, a.size, size=a.size)
        bi = rng.integers(0, b.size, size=b.size)
        diffs[i] = a[ai].mean() - b[bi].mean()
    alpha = (1 - confidence) / 2
    lo, hi = np.percentile(diffs, [100 * alpha, 100 * (1 - alpha)])
    return float(lo), float(hi)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with pooled standard deviation."""
    if a.size < 2 or b.size < 2:
        return float("nan")
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)
    pooled_sd = np.sqrt(
        ((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2)
    )
    if pooled_sd == 0.0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_sd)


def describe_effect(d: float) -> str:
    """Cohen's conventional labels."""
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz_a", help="path to first run's eval_results.npz")
    parser.add_argument("npz_b", help="path to second run's eval_results.npz")
    parser.add_argument("--label-a", default="A", help="display name for run A")
    parser.add_argument("--label-b", default="B", help="display name for run B")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="success threshold for best_reward (default: read from npz if present, else 1.0)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="bootstrap samples for difference CIs",
    )
    args = parser.parse_args()

    a = np.load(args.npz_a, allow_pickle=True)
    b = np.load(args.npz_b, allow_pickle=True)

    for d, p in [(a, args.npz_a), (b, args.npz_b)]:
        if "episode_reward" not in d.files or "episode_best_reward" not in d.files:
            raise KeyError(
                f"{p} is missing per-episode arrays. Re-run the eval after the "
                "agent was updated to save `episode_reward` / `episode_best_reward`."
            )

    rew_a = np.asarray(a["episode_reward"], dtype=np.float64)
    rew_b = np.asarray(b["episode_reward"], dtype=np.float64)
    best_a = np.asarray(a["episode_best_reward"], dtype=np.float64)
    best_b = np.asarray(b["episode_best_reward"], dtype=np.float64)

    # Resolve threshold: CLI > npz field > 1.0.
    if args.threshold is not None:
        threshold = float(args.threshold)
    elif "best_reward_threshold_for_success" in a.files:
        threshold = float(a["best_reward_threshold_for_success"])
    else:
        threshold = 1.0

    succ_a = (best_a >= threshold).astype(np.float64)
    succ_b = (best_b >= threshold).astype(np.float64)
    n_a, n_b = int(succ_a.size), int(succ_b.size)
    k_a, k_b = int(succ_a.sum()), int(succ_b.sum())

    label_a, label_b = args.label_a, args.label_b
    bar = "=" * 68

    print(f"\n{bar}")
    print(f"  Significance comparison:  {label_a}  vs  {label_b}")
    print(f"  (success threshold = {threshold})")
    print(bar)
    print(f"  {label_a}: n = {n_a:4d} episodes, k = {k_a:4d} successes")
    print(f"  {label_b}: n = {n_b:4d} episodes, k = {k_b:4d} successes\n")

    # ---- Success rate ----
    p_a = k_a / max(n_a, 1)
    p_b = k_b / max(n_b, 1)
    ci_a = wilson_interval(k_a, n_a)
    ci_b = wilson_interval(k_b, n_b)
    z, p_z = two_proportion_z_test(k_a, n_a, k_b, n_b)
    succ_diff_lo, succ_diff_hi = bootstrap_diff_ci(succ_a, succ_b, args.n_bootstrap)

    print("  Success rate")
    print("  ------------")
    print(f"    {label_a}: {p_a:.4f}   95% Wilson CI: [{ci_a[0]:.4f}, {ci_a[1]:.4f}]")
    print(f"    {label_b}: {p_b:.4f}   95% Wilson CI: [{ci_b[0]:.4f}, {ci_b[1]:.4f}]")
    print(f"    Difference ({label_a} − {label_b}): {p_a - p_b:+.4f}")
    print(f"        bootstrap 95% CI: [{succ_diff_lo:+.4f}, {succ_diff_hi:+.4f}]")
    print(f"    Two-proportion z-test:  z = {z:+.3f}   p = {p_z:.4g}")
    sig = "** SIGNIFICANT **" if p_z < 0.05 else "not significant"
    star = ("***" if p_z < 0.001 else "**" if p_z < 0.01 else "*" if p_z < 0.05 else "")
    print(f"      verdict at α=0.05: {sig}   {star}\n")

    # ---- Mean episode reward ----
    print("  Mean episode reward")
    print("  -------------------")
    mean_a, mean_b = rew_a.mean(), rew_b.mean()
    se_a = rew_a.std(ddof=1) / np.sqrt(n_a) if n_a > 1 else float("nan")
    se_b = rew_b.std(ddof=1) / np.sqrt(n_b) if n_b > 1 else float("nan")
    tcrit_a = stats.t.ppf(0.975, n_a - 1) if n_a > 1 else float("nan")
    tcrit_b = stats.t.ppf(0.975, n_b - 1) if n_b > 1 else float("nan")
    ci_rew_a = (mean_a - tcrit_a * se_a, mean_a + tcrit_a * se_a)
    ci_rew_b = (mean_b - tcrit_b * se_b, mean_b + tcrit_b * se_b)
    t_stat, p_t = stats.ttest_ind(rew_a, rew_b, equal_var=False)
    rew_diff_lo, rew_diff_hi = bootstrap_diff_ci(rew_a, rew_b, args.n_bootstrap)
    d = cohens_d(rew_a, rew_b)

    print(f"    {label_a}: {mean_a:.4f}   95% t CI: [{ci_rew_a[0]:.4f}, {ci_rew_a[1]:.4f}]")
    print(f"    {label_b}: {mean_b:.4f}   95% t CI: [{ci_rew_b[0]:.4f}, {ci_rew_b[1]:.4f}]")
    print(f"    Difference ({label_a} − {label_b}): {mean_a - mean_b:+.4f}")
    print(f"        bootstrap 95% CI: [{rew_diff_lo:+.4f}, {rew_diff_hi:+.4f}]")
    print(f"    Welch's t-test:  t = {t_stat:+.3f}   p = {p_t:.4g}")
    sig = "** SIGNIFICANT **" if p_t < 0.05 else "not significant"
    star = ("***" if p_t < 0.001 else "**" if p_t < 0.01 else "*" if p_t < 0.05 else "")
    print(f"      verdict at α=0.05: {sig}   {star}")
    if not np.isnan(d):
        print(f"    Cohen's d = {d:+.3f}  ({describe_effect(d)} effect)")

    print(f"{bar}\n")


if __name__ == "__main__":
    main()
