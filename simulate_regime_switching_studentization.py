# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
# ---

# %% [markdown]
# # Monte Carlo: regime-switching propensities and studentized inference
#
# This experiment uses adaptive, logged propensities that switch to a long-run
# regime (e_low or e_high) based on a noisy burn-in estimate. With tau=0, the
# regime is roughly a fair coin, so the limiting variance is random across
# replications (mixed-normal behavior). Studentization adapts to this randomness,
# while a fixed-variance CI (assuming a stabilized propensity) under-covers.
#
# Expected qualitative result: SN coverage near 0.95; Fixed coverage below 0.95.
# %%
from __future__ import annotations

import math
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import norm as scipy_norm

    norm_ppf = scipy_norm.ppf
except ImportError:  # pragma: no cover - fallback for environments without scipy
    norm_ppf = NormalDist().inv_cdf


# %% [markdown]
# ## Helpers and configuration
# %%
# Global so simulate_one_replication can report the selected regime without
# changing its required return signature.
LAST_REGIME_HIGH = None

# Baseline (incorrect) stabilized propensity used for the fixed-variance CI.
E_ASSUMED = 0.5


# %%
def standard_normal_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal PDF for plotting (vectorized)."""
    return np.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


# %%
def normal_pdf(x: np.ndarray, sd: float) -> np.ndarray:
    """Normal PDF with mean 0 and given sd (vectorized)."""
    return np.exp(-0.5 * (x / sd) ** 2) / (sd * math.sqrt(2.0 * math.pi))


# %%
def norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF with scipy fallback."""
    if "scipy_norm" in globals():
        return scipy_norm.cdf(x)
    return NormalDist().cdf(x)


# %%
def simulate_one_replication(
    n: int,
    n0: int,
    tau: float,
    sigma0: float,
    sigma1: float,
    e_low: float,
    e_high: float,
    e_burn: float,
    rng: np.random.Generator,
):
    """Simulate one replication and return tau_hat, se_SN, se_fixed, tstats."""
    global LAST_REGIME_HIGH

    if n <= n0:
        raise ValueError("n must exceed n0 for the regime-switching design.")

    # Burn-in phase: fixed propensity.
    A_burn = rng.binomial(1, e_burn, size=n0)
    eps1_burn = rng.normal(0.0, sigma1, size=n0)
    eps0_burn = rng.normal(0.0, sigma0, size=n0)
    Y_burn = np.where(A_burn == 1, tau + eps1_burn, eps0_burn)

    treated = A_burn == 1
    control = ~treated
    if treated.sum() == 0 or control.sum() == 0:
        tau_hat_burn = 0.0
    else:
        tau_hat_burn = Y_burn[treated].mean() - Y_burn[control].mean()

    # Regime choice makes the long-run propensity random across replications.
    regime_high = tau_hat_burn >= 0.0
    e_star = e_high if regime_high else e_low
    LAST_REGIME_HIGH = regime_high

    n_main = n - n0
    A_main = rng.binomial(1, e_star, size=n_main)
    eps1_main = rng.normal(0.0, sigma1, size=n_main)
    eps0_main = rng.normal(0.0, sigma0, size=n_main)
    Y_main = np.where(A_main == 1, tau + eps1_main, eps0_main)

    A = np.concatenate([A_burn, A_main])
    Y = np.concatenate([Y_burn, Y_main])
    e = np.full(n, e_star, dtype=float)
    e[:n0] = e_burn

    # IPW score with logged propensities.
    psi = A * Y / e - (1 - A) * Y / (1 - e)
    tau_hat = psi.mean()

    xi = psi - tau_hat
    V_hat = np.sum(xi**2)
    se_SN = math.sqrt(V_hat) / n
    tstat_SN = (tau_hat - tau) / se_SN

    var_fixed = sigma1**2 / E_ASSUMED + sigma0**2 / (1.0 - E_ASSUMED)
    se_fixed = math.sqrt(var_fixed / n)
    tstat_fixed = (tau_hat - tau) / se_fixed

    return tau_hat, se_SN, se_fixed, tstat_SN, tstat_fixed


# %%
def run_monte_carlo(
    n_list,
    R,
    seed,
    n0=50,
    tau=0.0,
    sigma0=1.0,
    sigma1=3.0,
    e_low=0.2,
    e_high=0.8,
    e_burn=0.5,
    alpha=0.05,
):
    """Run the Monte Carlo study and return results_df and t-stat dict."""
    results_rows = []
    tstats = {}
    z = float(norm_ppf(1.0 - alpha / 2.0))

    for idx, n in enumerate(n_list):
        rng = np.random.default_rng(seed + idx)

        tau_hats = np.empty(R)
        se_sn = np.empty(R)
        se_fixed = np.empty(R)
        tstat_sn = np.empty(R)
        tstat_fixed = np.empty(R)
        regime_high = np.empty(R, dtype=bool)
        vhat_over_n = np.empty(R)

        count_high = 0

        for r in range(R):
            (
                tau_hat,
                se_SN,
                se_F,
                t_SN,
                t_F,
            ) = simulate_one_replication(
                n=n,
                n0=n0,
                tau=tau,
                sigma0=sigma0,
                sigma1=sigma1,
                e_low=e_low,
                e_high=e_high,
                e_burn=e_burn,
                rng=rng,
            )

            tau_hats[r] = tau_hat
            se_sn[r] = se_SN
            se_fixed[r] = se_F
            tstat_sn[r] = t_SN
            tstat_fixed[r] = t_F

            if LAST_REGIME_HIGH is None:
                raise RuntimeError("Regime indicator was not set.")
            regime_high[r] = LAST_REGIME_HIGH
            count_high += int(LAST_REGIME_HIGH)
            vhat_over_n[r] = (se_SN * n) ** 2 / n

        count_low = R - count_high
        freq_high = count_high / R
        freq_low = count_low / R
        mean_tau = tau_hats.mean()
        sd_tau = tau_hats.std(ddof=1)

        print(
            f"n={n}: regime high={freq_high:.3f}, low={freq_low:.3f}; "
            f"mean tau_hat={mean_tau:.4f}, sd={sd_tau:.4f}"
        )

        ci_lower_sn = tau_hats - z * se_sn
        ci_upper_sn = tau_hats + z * se_sn
        coverage_sn = np.mean((ci_lower_sn <= tau) & (tau <= ci_upper_sn))
        avg_length_sn = np.mean(ci_upper_sn - ci_lower_sn)
        reject_sn = np.mean(np.abs(tstat_sn) > z)

        ci_lower_fixed = tau_hats - z * se_fixed
        ci_upper_fixed = tau_hats + z * se_fixed
        coverage_fixed = np.mean((ci_lower_fixed <= tau) & (tau <= ci_upper_fixed))
        avg_length_fixed = np.mean(ci_upper_fixed - ci_lower_fixed)
        reject_fixed = np.mean(np.abs(tstat_fixed) > z)

        results_rows.append(
            {
                "n": n,
                "method": "SN",
                "coverage": coverage_sn,
                "avg_length": avg_length_sn,
                "reject_rate": reject_sn,
                "mean_tau_hat": mean_tau,
                "sd_tau_hat": sd_tau,
            }
        )
        results_rows.append(
            {
                "n": n,
                "method": "Fixed",
                "coverage": coverage_fixed,
                "avg_length": avg_length_fixed,
                "reject_rate": reject_fixed,
                "mean_tau_hat": mean_tau,
                "sd_tau_hat": sd_tau,
            }
        )

        tstats[n] = {
            "SN": tstat_sn,
            "Fixed": tstat_fixed,
            "tau_hat": tau_hats,
            "se_sn": se_sn,
            "se_fixed": se_fixed,
            "regime_high": regime_high,
            "vhat_over_n": vhat_over_n,
        }

    results_df = pd.DataFrame(results_rows)
    return results_df, tstats


# %%
def plot_tstat_hist(tstats, n, method, output_path):
    """Plot histogram of t-stats with standard normal overlay."""
    color = "steelblue" if method == "SN" else "tomato"
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(tstats, bins=50, density=True, alpha=0.65, color=color, edgecolor="white")

    x = np.linspace(-4.5, 4.5, 400)
    ax.plot(x, standard_normal_pdf(x), color="black", linewidth=2.0, label="N(0,1)")
    ax.set_title(f"t-statistics: {method}, n={n}")
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# %%
def summarize_unconditional(results_df):
    """Return unconditional results with consistent ordering."""
    return results_df.sort_values(["n", "method"]).reset_index(drop=True)


# %%
def summarize_conditional(tstats, n_list, tau, alpha):
    """Compute conditional performance by regime for each method."""
    z = float(norm_ppf(1.0 - alpha / 2.0))
    rows = []

    for n in n_list:
        data = tstats[n]
        regime_high = data["regime_high"]
        tau_hat = data["tau_hat"]
        se_sn = data["se_sn"]
        se_fixed = data["se_fixed"]
        tstat_sn = data["SN"]
        tstat_fixed = data["Fixed"]

        for regime_name, mask in [
            ("high", regime_high),
            ("low", ~regime_high),
        ]:
            count = int(mask.sum())
            if count == 0:
                continue

            ci_lower_sn = tau_hat[mask] - z * se_sn[mask]
            ci_upper_sn = tau_hat[mask] + z * se_sn[mask]
            coverage_sn = np.mean((ci_lower_sn <= tau) & (tau <= ci_upper_sn))
            avg_length_sn = np.mean(ci_upper_sn - ci_lower_sn)
            reject_sn = np.mean(np.abs(tstat_sn[mask]) > z)

            rows.append(
                {
                    "n": n,
                    "method": "SN",
                    "regime": regime_name,
                    "coverage": coverage_sn,
                    "avg_length": avg_length_sn,
                    "reject_rate": reject_sn,
                    "count": count,
                }
            )

            ci_lower_fixed = tau_hat[mask] - z * se_fixed[mask]
            ci_upper_fixed = tau_hat[mask] + z * se_fixed[mask]
            coverage_fixed = np.mean((ci_lower_fixed <= tau) & (tau <= ci_upper_fixed))
            avg_length_fixed = np.mean(ci_upper_fixed - ci_lower_fixed)
            reject_fixed = np.mean(np.abs(tstat_fixed[mask]) > z)

            rows.append(
                {
                    "n": n,
                    "method": "Fixed",
                    "regime": regime_name,
                    "coverage": coverage_fixed,
                    "avg_length": avg_length_fixed,
                    "reject_rate": reject_fixed,
                    "count": count,
                }
            )

    results = pd.DataFrame(rows)
    return results.sort_values(["n", "method", "regime"]).reset_index(drop=True)


# %%
def plot_vhat_over_n(vhat_over_n, n, v_high, v_low, output_path):
    """Plot histogram of V_hat / n with theoretical regime variances."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(vhat_over_n, bins=40, density=True, alpha=0.7, color="slategray")
    ax.axvline(v_high, color="steelblue", linestyle="--", linewidth=2.0, label="v_high")
    ax.axvline(v_low, color="tomato", linestyle="--", linewidth=2.0, label="v_low")
    ax.set_title(f"V_hat / n distribution, n={n}")
    ax.set_xlabel("V_hat / n")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# %%
def plot_tstat_fixed_mixture(tstat_fixed, n, sd_high, sd_low, output_path):
    """Plot fixed t-stat histogram with standard normal and mixture overlay."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.hist(
        tstat_fixed,
        bins=50,
        density=True,
        alpha=0.65,
        color="tomato",
        edgecolor="white",
    )

    x = np.linspace(-6.0, 6.0, 500)
    ax.plot(x, standard_normal_pdf(x), color="black", linewidth=2.0, label="N(0,1)")
    mix_density = 0.5 * normal_pdf(x, sd_high) + 0.5 * normal_pdf(x, sd_low)
    ax.plot(x, mix_density, color="purple", linewidth=2.0, label="Mixture")
    ax.set_title(f"Fixed t-stat with mixture overlay, n={n}")
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# %%
def make_booktabs_table(df, float_format="%.3f"):
    """Create a LaTeX booktabs table string from a DataFrame."""
    columns = list(df.columns)
    alignments = []
    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            alignments.append("r")
        else:
            alignments.append("l")

    lines = []
    lines.append("\\begin{tabular}{" + "".join(alignments) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(columns) + " \\\\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        formatted = []
        for col in columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                formatted.append(float_format % float(val))
            else:
                formatted.append(str(val))
        lines.append(" & ".join(formatted) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


# %%
def main():
    # Default parameters (edit here or pass via a wrapper).
    n_list = [500, 2000]
    R = 3000
    seed = 12345

    n0 = 50
    tau = 0.0
    sigma0 = 1.0
    sigma1 = 3.0
    e_burn = 0.5
    e_low = 0.2
    e_high = 0.8
    alpha = 0.05

    # Expected qualitative result: SN coverage near 0.95; Fixed coverage below 0.95.
    results_df, tstats = run_monte_carlo(
        n_list=n_list,
        R=R,
        seed=seed,
        n0=n0,
        tau=tau,
        sigma0=sigma0,
        sigma1=sigma1,
        e_low=e_low,
        e_high=e_high,
        e_burn=e_burn,
        alpha=alpha,
    )

    results_df = summarize_unconditional(results_df)
    results_df.to_csv("results_regime_switching.csv", index=False)
    print("\nMonte Carlo results")
    print(results_df.to_string(index=False, float_format="{:.4f}".format))

    cond_df = summarize_conditional(tstats, n_list, tau=tau, alpha=alpha)
    cond_df.to_csv("results_regime_conditional.csv", index=False)
    print("\nConditional performance by regime")
    print(cond_df.to_string(index=False, float_format="{:.4f}".format))

    v_high = sigma1**2 / e_high + sigma0**2 / (1.0 - e_high)
    v_low = sigma1**2 / e_low + sigma0**2 / (1.0 - e_low)
    var_fixed = sigma1**2 / E_ASSUMED + sigma0**2 / (1.0 - E_ASSUMED)
    sd_high = math.sqrt(v_high / var_fixed)
    sd_low = math.sqrt(v_low / var_fixed)
    z = float(norm_ppf(1.0 - alpha / 2.0))
    coverage_fixed_pred = 0.5 * (2 * norm_cdf(z / sd_high) - 1) + 0.5 * (
        2 * norm_cdf(z / sd_low) - 1
    )
    reject_fixed_pred = 1.0 - coverage_fixed_pred

    print("\nFixed-method mixture prediction (asymptotic)")
    print(f"predicted coverage={coverage_fixed_pred:.4f}")
    print(f"predicted reject_rate={reject_fixed_pred:.4f}")
    fixed_rows = results_df[results_df["method"] == "Fixed"]
    for _, row in fixed_rows.iterrows():
        print(
            f"n={int(row['n'])}: "
            f"sim coverage={row['coverage']:.4f}, "
            f"sim reject_rate={row['reject_rate']:.4f}"
        )

    for n in n_list:
        plot_tstat_hist(
            tstats[n]["SN"],
            n,
            "SN",
            f"tstat_hist_SN_n{n}.png",
        )
        plot_tstat_hist(
            tstats[n]["Fixed"],
            n,
            "Fixed",
            f"tstat_hist_Fixed_n{n}.png",
        )
        plot_vhat_over_n(
            tstats[n]["vhat_over_n"],
            n,
            v_high,
            v_low,
            f"Vhat_over_n_hist_n{n}.png",
        )
        plot_tstat_fixed_mixture(
            tstats[n]["Fixed"],
            n,
            sd_high,
            sd_low,
            f"tstat_fixed_mixture_overlay_n{n}.png",
        )

    table_main = make_booktabs_table(results_df, float_format="%.3f")
    with open("table_main.tex", "w", encoding="utf-8") as f:
        f.write(table_main)

    cond_df_out = cond_df.copy()
    cond_df_out["count"] = cond_df_out["count"].astype(int)
    table_conditional = make_booktabs_table(cond_df_out, float_format="%.3f")
    with open("table_conditional.tex", "w", encoding="utf-8") as f:
        f.write(table_conditional)


# %%
if __name__ == "__main__":
    main()
