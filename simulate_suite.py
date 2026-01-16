#!/usr/bin/env python3
"""
Monte Carlo suite for Designs A/B/C in the martingale-DML paper.

Design A: regime-switching propensities with random long-run variance (no
stabilization); Design B: stabilization benchmark; Design C: covariates with
predictable forward cross-fitting and predictability-violation control.

Outputs (CSV + LaTeX): results_designA_main.csv, results_designA_conditional.csv,
results_designB.csv, results_designC.csv, table_designA.tex,
table_designA_conditional.tex, table_designB.tex, table_designC.tex.

Reproduce (example): python simulate_suite.py --design all --n_list 500,2000 --R 500 --seed 12345
"""
from __future__ import annotations

import argparse
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm as scipy_norm

    norm_ppf = scipy_norm.ppf
except ImportError:  # pragma: no cover - fallback for environments without scipy
    norm_ppf = NormalDist().inv_cdf


def parse_n_list(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("n_list must be a comma-separated list like '500,2000'.")
    return [int(p) for p in parts]


def expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def mcse(p_hat: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / n)


def make_booktabs_table(df: pd.DataFrame, float_format: str = "%.3f") -> str:
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


def rep_rng(seed: int, design_id: int, n_index: int, rep_index: int) -> np.random.Generator:
    base = seed + design_id * 1_000_000 + n_index * 10_000 + rep_index
    return np.random.default_rng(base)


def simulate_design_ab_replication(
    n: int,
    n0: int,
    tau: float,
    sigma0: float,
    sigma1: float,
    pi_low: float,
    pi_high: float,
    pi_burn: float,
    fixed_pi: float,
    fixed_tau: float,
    rng: np.random.Generator,
):
    if n <= n0:
        raise ValueError("n must exceed n0 for the regime-switching design.")

    A_burn = rng.binomial(1, pi_burn, size=n0)
    eps1_burn = rng.normal(0.0, sigma1, size=n0)
    eps0_burn = rng.normal(0.0, sigma0, size=n0)
    Y_burn = np.where(A_burn == 1, tau + eps1_burn, eps0_burn)

    treated = A_burn == 1
    control = ~treated
    if treated.sum() == 0 or control.sum() == 0:
        tau_hat_burn = 0.0
    else:
        tau_hat_burn = Y_burn[treated].mean() - Y_burn[control].mean()

    regime_high = tau_hat_burn >= 0.0
    pi_star = pi_high if regime_high else pi_low

    n_main = n - n0
    A_main = rng.binomial(1, pi_star, size=n_main)
    eps1_main = rng.normal(0.0, sigma1, size=n_main)
    eps0_main = rng.normal(0.0, sigma0, size=n_main)
    Y_main = np.where(A_main == 1, tau + eps1_main, eps0_main)

    A = np.concatenate([A_burn, A_main])
    Y = np.concatenate([Y_burn, Y_main])
    pi = np.full(n, pi_star, dtype=float)
    pi[:n0] = pi_burn

    A_T = A[n0:]
    Y_T = Y[n0:]
    pi_T = pi[n0:]
    n_eff = n - n0

    psi = A_T * Y_T / pi_T - (1 - A_T) * Y_T / (1 - pi_T)
    tau_hat = psi.mean()
    xi = psi - tau_hat
    V_hat = np.sum(xi**2)
    se_sn = math.sqrt(V_hat) / n_eff

    var_fixed = (sigma1**2 + fixed_tau**2) / fixed_pi + sigma0**2 / (1.0 - fixed_pi) - fixed_tau**2
    se_fixed = math.sqrt(var_fixed / n_eff)

    return tau_hat, se_sn, se_fixed, regime_high


def run_design_a(
    n_list: list[int],
    R: int,
    seed: int,
    n0: int,
    tau: float,
    sigma0: float,
    sigma1: float,
    pi_low: float,
    pi_high: float,
    pi_burn: float,
    fixed_pi: float,
    alpha: float,
    assert_patterns: bool,
):
    z = float(norm_ppf(1.0 - alpha / 2.0))
    rows_main: list[dict] = []
    rows_cond: list[dict] = []

    for n_index, n in enumerate(n_list):
        tau_hat = np.empty(R)
        se_sn = np.empty(R)
        se_fixed = np.empty(R)
        regime_high = np.empty(R, dtype=bool)

        for r in range(R):
            rng = rep_rng(seed, design_id=1, n_index=n_index, rep_index=r)
            tau_hat[r], se_sn[r], se_fixed[r], regime_high[r] = simulate_design_ab_replication(
                n=n,
                n0=n0,
                tau=tau,
                sigma0=sigma0,
                sigma1=sigma1,
                pi_low=pi_low,
                pi_high=pi_high,
                pi_burn=pi_burn,
                fixed_pi=fixed_pi,
                fixed_tau=tau,
                rng=rng,
            )

        ci_lower_sn = tau_hat - z * se_sn
        ci_upper_sn = tau_hat + z * se_sn
        coverage_sn = np.mean((ci_lower_sn <= tau) & (tau <= ci_upper_sn))
        avg_len_sn = np.mean(ci_upper_sn - ci_lower_sn)

        ci_lower_fixed = tau_hat - z * se_fixed
        ci_upper_fixed = tau_hat + z * se_fixed
        coverage_fixed = np.mean((ci_lower_fixed <= tau) & (tau <= ci_upper_fixed))
        avg_len_fixed = np.mean(ci_upper_fixed - ci_lower_fixed)

        rows_main.append(
            {
                "n": n,
                "method": "SN",
                "coverage": coverage_sn,
                "mcse": mcse(coverage_sn, R),
                "avg_len": avg_len_sn,
            }
        )
        rows_main.append(
            {
                "n": n,
                "method": "Fixed",
                "coverage": coverage_fixed,
                "mcse": mcse(coverage_fixed, R),
                "avg_len": avg_len_fixed,
            }
        )

        for regime_name, mask in [("high", regime_high), ("low", ~regime_high)]:
            count = int(mask.sum())
            if count == 0:
                continue

            ci_lower_sn = tau_hat[mask] - z * se_sn[mask]
            ci_upper_sn = tau_hat[mask] + z * se_sn[mask]
            cov_sn = np.mean((ci_lower_sn <= tau) & (tau <= ci_upper_sn))
            avg_sn = np.mean(ci_upper_sn - ci_lower_sn)

            rows_cond.append(
                {
                    "n": n,
                    "method": "SN",
                    "regime": regime_name,
                    "coverage": cov_sn,
                    "mcse": mcse(cov_sn, count),
                    "avg_len": avg_sn,
                    "count": count,
                }
            )

            ci_lower_fixed = tau_hat[mask] - z * se_fixed[mask]
            ci_upper_fixed = tau_hat[mask] + z * se_fixed[mask]
            cov_fixed = np.mean((ci_lower_fixed <= tau) & (tau <= ci_upper_fixed))
            avg_fixed = np.mean(ci_upper_fixed - ci_lower_fixed)

            rows_cond.append(
                {
                    "n": n,
                    "method": "Fixed",
                    "regime": regime_name,
                    "coverage": cov_fixed,
                    "mcse": mcse(cov_fixed, count),
                    "avg_len": avg_fixed,
                    "count": count,
                }
            )

        if assert_patterns:
            tol = 0.20
            cov_sn_high = rows_cond[-4]["coverage"]
            cov_sn_low = rows_cond[-2]["coverage"]
            cov_fixed_high = rows_cond[-3]["coverage"]
            cov_fixed_low = rows_cond[-1]["coverage"]
            if abs(cov_sn_high - (1.0 - alpha)) > tol or abs(cov_sn_low - (1.0 - alpha)) > tol:
                raise AssertionError("Design A: SN coverage too far from nominal in smoke test.")
            if cov_fixed_low >= cov_fixed_high:
                raise AssertionError("Design A: Fixed does not under-cover in low regime.")

    results_main = pd.DataFrame(rows_main).sort_values(["n", "method"]).reset_index(drop=True)
    results_cond = (
        pd.DataFrame(rows_cond).sort_values(["n", "method", "regime"]).reset_index(drop=True)
    )
    return results_main, results_cond


def run_design_b(
    n_list: list[int],
    R: int,
    seed: int,
    tau: float,
    sigma0: float,
    sigma1: float,
    pi_low: float,
    pi_high: float,
    pi_burn: float,
    fixed_pi: float,
    alpha: float,
):
    z = float(norm_ppf(1.0 - alpha / 2.0))
    rows: list[dict] = []

    for n_index, n in enumerate(n_list):
        n0 = int(math.ceil(math.sqrt(n)))
        tau_hat = np.empty(R)
        se_sn = np.empty(R)
        se_fixed = np.empty(R)

        for r in range(R):
            rng = rep_rng(seed, design_id=2, n_index=n_index, rep_index=r)
            tau_hat[r], se_sn[r], se_fixed[r], _ = simulate_design_ab_replication(
                n=n,
                n0=n0,
                tau=tau,
                sigma0=sigma0,
                sigma1=sigma1,
                pi_low=pi_low,
                pi_high=pi_high,
                pi_burn=pi_burn,
                fixed_pi=fixed_pi,
                fixed_tau=tau,
                rng=rng,
            )

        ci_lower_sn = tau_hat - z * se_sn
        ci_upper_sn = tau_hat + z * se_sn
        coverage_sn = np.mean((ci_lower_sn <= tau) & (tau <= ci_upper_sn))
        avg_len_sn = np.mean(ci_upper_sn - ci_lower_sn)

        ci_lower_fixed = tau_hat - z * se_fixed
        ci_upper_fixed = tau_hat + z * se_fixed
        coverage_fixed = np.mean((ci_lower_fixed <= tau) & (tau <= ci_upper_fixed))
        avg_len_fixed = np.mean(ci_upper_fixed - ci_lower_fixed)

        rows.append(
            {
                "n": n,
                "method": "SN",
                "coverage": coverage_sn,
                "mcse": mcse(coverage_sn, R),
                "avg_len": avg_len_sn,
            }
        )
        rows.append(
            {
                "n": n,
                "method": "Fixed",
                "coverage": coverage_fixed,
                "mcse": mcse(coverage_fixed, R),
                "avg_len": avg_len_fixed,
            }
        )

    return pd.DataFrame(rows).sort_values(["n", "method"]).reset_index(drop=True)


def t_error(rng: np.random.Generator, df: int, size: int) -> np.ndarray:
    scale = math.sqrt((df - 2.0) / df)
    return rng.standard_t(df, size=size) * scale


def m0_star(x: np.ndarray) -> np.ndarray:
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    return 0.5 * x1 + 0.25 * x2**2 - 0.25 + 0.5 * np.sin(x3)


def tau_x(x: np.ndarray, tau0: float, delta: float) -> np.ndarray:
    return tau0 + delta * np.sin(x[:, 0])


def build_features_predictable(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    cols = [
        np.ones(n),
        x1,
        x2**2,
        np.sin(x3),
        np.sin(x1),
    ]
    return np.column_stack(cols)


def build_features_leaky(x: np.ndarray) -> np.ndarray:
    n, p = x.shape
    cols = [np.ones(n)]
    cols.extend([x[:, j] for j in range(p)])
    cols.extend([x[:, j] ** 2 for j in range(p)])
    cols.extend([x[:, j] ** 3 for j in range(p)])
    cols.append(np.sin(x[:, 0]))
    cols.append(np.sin(x[:, 1]))
    cols.append(np.sin(x[:, 2]))
    max_inter = min(p, 10)
    for i in range(max_inter):
        for j in range(i + 1, max_inter):
            cols.append(x[:, i] * x[:, j])
    return np.column_stack(cols)


def ridge_fit(phi: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    if y.size == 0:
        return np.zeros(phi.shape[1])
    xtx = phi.T @ phi
    d = xtx.shape[0]
    xtx.flat[:: d + 1] += ridge
    return np.linalg.solve(xtx, phi.T @ y)


def aipw_score(
    y: np.ndarray, a: np.ndarray, pi: np.ndarray, m0: np.ndarray, m1: np.ndarray
) -> np.ndarray:
    return (m1 - m0) + a * (y - m1) / pi - (1 - a) * (y - m0) / (1 - pi)


def simulate_design_c_replication(
    n: int,
    p: int,
    K: int,
    tau0: float,
    delta: float,
    eps: float,
    lam: float,
    ridge: float,
    ridge_leaky: float,
    df: int,
    rng: np.random.Generator,
):
    x = rng.normal(0.0, 1.0, size=(n, p))
    m0_true = m0_star(x)
    tau_true = tau_x(x, tau0=tau0, delta=delta)
    m1_true = m0_true + tau_true

    eps0 = t_error(rng, df=df, size=n)
    eps1 = t_error(rng, df=df, size=n)

    pi = np.empty(n)
    a = np.empty(n, dtype=int)
    y = np.empty(n)
    mhat0_pred = np.zeros(n)
    mhat1_pred = np.zeros(n)

    phi_pred = build_features_predictable(x)
    phi_leaky = build_features_leaky(x)
    blocks = np.array_split(np.arange(n), K)

    burn_idx = blocks[0]
    pi[burn_idx] = 0.5
    a[burn_idx] = rng.binomial(1, pi[burn_idx])
    y[burn_idx] = np.where(
        a[burn_idx] == 1,
        m1_true[burn_idx] + eps1[burn_idx],
        m0_true[burn_idx] + eps0[burn_idx],
    )

    for k in range(1, K):
        past_idx = np.concatenate(blocks[:k])
        cur_idx = blocks[k]

        past_phi = phi_pred[past_idx]
        past_y = y[past_idx]
        past_a = a[past_idx]

        beta0 = ridge_fit(past_phi[past_a == 0], past_y[past_a == 0], ridge)
        beta1 = ridge_fit(past_phi[past_a == 1], past_y[past_a == 1], ridge)

        mhat0_pred[cur_idx] = phi_pred[cur_idx] @ beta0
        mhat1_pred[cur_idx] = phi_pred[cur_idx] @ beta1

        logits = lam * (mhat1_pred[cur_idx] - mhat0_pred[cur_idx])
        pi[cur_idx] = eps + (1.0 - 2.0 * eps) * expit(logits)
        a[cur_idx] = rng.binomial(1, pi[cur_idx])
        y[cur_idx] = np.where(
            a[cur_idx] == 1,
            m1_true[cur_idx] + eps1[cur_idx],
            m0_true[cur_idx] + eps0[cur_idx],
        )

    full_beta0 = ridge_fit(phi_leaky[a == 0], y[a == 0], ridge_leaky)
    full_beta1 = ridge_fit(phi_leaky[a == 1], y[a == 1], ridge_leaky)
    mhat0_full = phi_leaky @ full_beta0
    mhat1_full = phi_leaky @ full_beta1

    scored_idx = np.concatenate(blocks[1:])
    n_eff = scored_idx.size

    return {
        "y": y,
        "a": a,
        "pi": pi,
        "m0_true": m0_true,
        "m1_true": m1_true,
        "mhat0_pred": mhat0_pred,
        "mhat1_pred": mhat1_pred,
        "mhat0_full": mhat0_full,
        "mhat1_full": mhat1_full,
        "scored_idx": scored_idx,
        "n_eff": n_eff,
    }


def run_design_c(
    n_list: list[int],
    R: int,
    seed: int,
    p: int,
    K: int,
    tau0: float,
    delta: float,
    eps: float,
    lam: float,
    ridge: float,
    ridge_leaky: float,
    df: int,
    alpha: float,
    assert_patterns: bool,
):
    z = float(norm_ppf(1.0 - alpha / 2.0))
    rows: list[dict] = []

    method_order = ["SN-AIPW", "SN-IPW", "SN-Oracle", "Leaky-AIPW"]

    for n_index, n in enumerate(n_list):
        stats = {m: {"tau_hat": np.empty(R), "se": np.empty(R)} for m in method_order}

        for r in range(R):
            rng = rep_rng(seed, design_id=3, n_index=n_index, rep_index=r)
            data = simulate_design_c_replication(
                n=n,
                p=p,
                K=K,
                tau0=tau0,
                delta=delta,
                eps=eps,
                lam=lam,
                ridge=ridge,
                ridge_leaky=ridge_leaky,
                df=df,
                rng=rng,
            )

            idx = data["scored_idx"]
            y = data["y"][idx]
            a = data["a"][idx]
            pi = data["pi"][idx]
            n_eff = data["n_eff"]

            m0_pred = data["mhat0_pred"][idx]
            m1_pred = data["mhat1_pred"][idx]
            m0_oracle = data["m0_true"][idx]
            m1_oracle = data["m1_true"][idx]
            m0_full = data["mhat0_full"][idx]
            m1_full = data["mhat1_full"][idx]

            psi_aipw = aipw_score(y, a, pi, m0_pred, m1_pred)
            psi_ipw = aipw_score(y, a, pi, np.zeros_like(m0_pred), np.zeros_like(m1_pred))
            psi_oracle = aipw_score(y, a, pi, m0_oracle, m1_oracle)
            psi_leaky = aipw_score(y, a, pi, m0_full, m1_full)

            for method, psi in [
                ("SN-AIPW", psi_aipw),
                ("SN-IPW", psi_ipw),
                ("SN-Oracle", psi_oracle),
                ("Leaky-AIPW", psi_leaky),
            ]:
                tau_hat = psi.mean()
                xi = psi - tau_hat
                V_hat = np.sum(xi**2)
                se = math.sqrt(V_hat) / n_eff
                stats[method]["tau_hat"][r] = tau_hat
                stats[method]["se"][r] = se

        for method in method_order:
            tau_hat = stats[method]["tau_hat"]
            se = stats[method]["se"]
            ci_lower = tau_hat - z * se
            ci_upper = tau_hat + z * se
            coverage = np.mean((ci_lower <= tau0) & (tau0 <= ci_upper))
            avg_len = np.mean(ci_upper - ci_lower)
            bias = np.mean(tau_hat - tau0)
            rmse = math.sqrt(np.mean((tau_hat - tau0) ** 2))

            rows.append(
                {
                    "n": n,
                    "method": method,
                    "coverage": coverage,
                    "mcse": mcse(coverage, R),
                    "avg_len": avg_len,
                    "bias": bias,
                    "rmse": rmse,
                }
            )

        if assert_patterns:
            cov_lookup = {row["method"]: row["coverage"] for row in rows if row["n"] == n}
            if cov_lookup["Leaky-AIPW"] >= cov_lookup["SN-AIPW"]:
                raise AssertionError("Design C: leaky AIPW is not more anti-conservative.")

    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    return df.sort_values(["n", "method"]).reset_index(drop=True)


def write_table(df: pd.DataFrame, path: str, float_format: str = "%.3f") -> None:
    table = make_booktabs_table(df, float_format=float_format)
    with open(path, "w", encoding="utf-8") as f:
        f.write(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo simulation suite.")
    parser.add_argument("--design", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--n_list", type=str, default="500,2000")
    parser.add_argument("--R", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--alpha", type=float, default=0.05)

    parser.add_argument("--n0", type=int, default=50)
    parser.add_argument("--pi_low", type=float, default=0.2)
    parser.add_argument("--pi_high", type=float, default=0.8)
    parser.add_argument("--pi_burn", type=float, default=0.5)
    parser.add_argument("--tau_A", type=float, default=0.0)
    parser.add_argument("--tau_B", type=float, default=2.0)
    parser.add_argument("--sigma0", type=float, default=1.0)
    parser.add_argument("--sigma1", type=float, default=3.0)
    parser.add_argument("--fixed_pi_A", type=float, default=0.5)
    parser.add_argument("--fixed_pi_B", type=float, default=0.8)

    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--tau0", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=0.2)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--lambda_", type=float, default=2.5)
    parser.add_argument("--ridge", type=float, default=0.1)
    parser.add_argument("--ridge_leaky", type=float, default=1e-6)
    parser.add_argument("--df", type=int, default=5)

    parser.add_argument("--smoke", action="store_true", help="Run smoke assertions and print diagnostics.")

    args = parser.parse_args()
    n_list = parse_n_list(args.n_list)

    if args.design in ("A", "all"):
        results_main, results_cond = run_design_a(
            n_list=n_list,
            R=args.R,
            seed=args.seed,
            n0=args.n0,
            tau=args.tau_A,
            sigma0=args.sigma0,
            sigma1=args.sigma1,
            pi_low=args.pi_low,
            pi_high=args.pi_high,
            pi_burn=args.pi_burn,
            fixed_pi=args.fixed_pi_A,
            alpha=args.alpha,
            assert_patterns=args.smoke,
        )

        results_main.to_csv("results_designA_main.csv", index=False)
        results_cond.to_csv("results_designA_conditional.csv", index=False)
        write_table(results_main, "table_designA.tex")
        write_table(results_cond, "table_designA_conditional.tex")

        if args.smoke:
            for n in n_list:
                subset = results_cond[results_cond["n"] == n]
                counts = subset[["regime", "count"]].drop_duplicates()
                print(f"Design A n={n}: regime counts {counts.to_dict(orient='records')}")
                summary = results_main[results_main["n"] == n]
                print(summary.to_string(index=False, float_format="{:.3f}".format))

    if args.design in ("B", "all"):
        results_b = run_design_b(
            n_list=n_list,
            R=args.R,
            seed=args.seed,
            tau=args.tau_B,
            sigma0=args.sigma0,
            sigma1=args.sigma1,
            pi_low=args.pi_low,
            pi_high=args.pi_high,
            pi_burn=args.pi_burn,
            fixed_pi=args.fixed_pi_B,
            alpha=args.alpha,
        )

        results_b.to_csv("results_designB.csv", index=False)
        write_table(results_b, "table_designB.tex")

        if args.smoke:
            for n in n_list:
                summary = results_b[results_b["n"] == n]
                print(f"Design B n={n}")
                print(summary.to_string(index=False, float_format="{:.3f}".format))

    if args.design in ("C", "all"):
        results_c = run_design_c(
            n_list=n_list,
            R=args.R,
            seed=args.seed,
            p=args.p,
            K=args.K,
            tau0=args.tau0,
            delta=args.delta,
            eps=args.eps,
            lam=args.lambda_,
            ridge=args.ridge,
            ridge_leaky=args.ridge_leaky,
            df=args.df,
            alpha=args.alpha,
            assert_patterns=args.smoke,
        )

        results_c.to_csv("results_designC.csv", index=False)
        write_table(results_c, "table_designC.tex")

        if args.smoke:
            for n in n_list:
                summary = results_c[results_c["n"] == n]
                print(f"Design C n={n}")
                print(summary.to_string(index=False, float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
