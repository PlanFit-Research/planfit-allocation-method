#!/usr/bin/env python3
"""
run_backtests.py — PlanFit vs. 60/40 stress‑test harness
=======================================================
*Purpose* — Compare how much starting capital a household needs under
PlanFit’s aggregate SAA versus a conventional 60/40 mix when both are
subjected to historical sequence risk and a bootstrap robustness check.

Usage
-----
```powershell
python Tests\run_backtests.py data\returns_1950_2022.csv \
                            data\withdrawals.csv         \
                            --capital 3300000
```

Outputs: prints two summary tables and writes `results_summary.csv`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.optimize import brentq

SEED = 42
RNG = default_rng(SEED)

# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------
@dataclass
class Strategy:
    name: str
    weights: tuple[float, float, float]  # (stocks, bonds, cash)
    start_capital: float                 # to be solved for benchmark

# ------------------------------------------------------------------
# Loader helpers
# ------------------------------------------------------------------

def load_returns(path: Path) -> np.ndarray:
    """Return an N×3 ndarray of float real returns."""
    df = pd.read_csv(path)
    req = {"Stocks", "Bonds", "Cash"}
    if not req.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {req}")
    return df[["Stocks", "Bonds", "Cash"]].astype(float).to_numpy()


def load_withdrawals(path: Path) -> np.ndarray:
    """Return a 1‑D float vector of real withdrawals (positive = outflow)."""
    df = pd.read_csv(path).sort_values("Year")
    col = (
        df["RealWithdrawal"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)        # drop $ and commas
        .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)  # (1000) → -1000
        .str.strip()
        .astype(float)
    )
    return col.to_numpy()

# ------------------------------------------------------------------
# Core simulation helpers
# ------------------------------------------------------------------

def simulate_path(
    returns_block: np.ndarray,  # shape (H,3)
    weights: tuple[float, float, float],
    capital: float,
    cashflows: np.ndarray,     # length H
) -> tuple[bool, float]:
    """Run one H‑year path; return (success?, ending wealth)."""
    wealth = capital
    for r_vec, cf in zip(returns_block, cashflows):
        wealth = wealth * (1 + np.dot(weights, r_vec)) - cf
        if wealth < 0:
            return False, wealth
    return True, wealth


def rolling_slices(mat: np.ndarray, h: int) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx) inclusive‑exclusive windows."""
    n = mat.shape[0]
    return [(i, i + h) for i in range(0, n - h + 1)]


def solve_benchmark_capital(
    returns_mat: np.ndarray,
    cashflows: np.ndarray,
    weights: tuple[float, float, float],
    target_success: float = 0.95,
):
    H = len(cashflows)
    slices = rolling_slices(returns_mat, H)

    def success_gap(capital: float):
        succ = sum(
            simulate_path(returns_mat[s:e], weights, capital, cashflows)[0]
            for s, e in slices
        )
        return succ / len(slices) - target_success

    return brentq(success_gap, 1e5, 3e7)


def bootstrap_paths(mat: np.ndarray, h: int, block: int = 5, n: int = 5000):
    """Yield bootstrap resamples built from (h/block) non‑overlapping blocks."""
    n_rows = mat.shape[0]
    n_blocks = h // block
    for _ in range(n):
        idx_blocks = RNG.choice(range(n_rows - block + 1), size=n_blocks)
        rows = np.concatenate([mat[i : i + block] for i in idx_blocks], axis=0)
        yield rows

# ------------------------------------------------------------------
# Main driver
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PlanFit benchmark & stress‑tests")
    parser.add_argument("returns", type=Path, help="CSV of annual real returns")
    parser.add_argument("withdrawals", type=Path, help="CSV of real withdrawals")
    parser.add_argument("--capital", type=float, default=3.3e6, help="PlanFit starting capital")
    args = parser.parse_args()

    ret_mat = load_returns(args.returns)
    cashflows = load_withdrawals(args.withdrawals)
    H = len(cashflows)

    # --- define strategies ---
    planfit = Strategy("PlanFit", (0.37, 0.0, 0.63), args.capital)
    benchmark = Strategy("60/40", (0.6, 0.4, 0.0), 0.0)

    # --- solve benchmark start capital ---
    benchmark.start_capital = solve_benchmark_capital(ret_mat, cashflows, benchmark.weights)

    # --- deterministic rolling-window test ---
    rows = []
    for strat in [planfit, benchmark]:
        endings = [
            simulate_path(ret_mat[s:e], strat.weights, strat.start_capital, cashflows)[1]
            for s, e in rolling_slices(ret_mat, H)
        ]
        endings = np.array(endings)
        cvar5 = endings[np.argsort(endings)][: max(1, int(0.05 * len(endings)))].mean()
        rows.append(
            {
                "Strategy": strat.name,
                "StartCap": strat.start_capital,
                "CVaR5": cvar5,
                "MeanEnd": endings.mean(),
            }
        )
    det_summary = pd.DataFrame(rows)

    # --- bootstrap capital inflation ---
    cap_infl = {planfit.name: [], benchmark.name: []}
    for path in bootstrap_paths(ret_mat, H):
        for strat in [planfit, benchmark]:
            ok, _ = simulate_path(path, strat.weights, strat.start_capital, cashflows)
            if ok:
                cap_infl[strat.name].append(0.0)
            else:
                # how much extra capital until success?
                def gap(x):
                    return simulate_path(path, strat.weights, strat.start_capital + x, cashflows)[0] - 1

                extra = brentq(gap, 0, 2e6)
                cap_infl[strat.name].append(extra / strat.start_capital)

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    pd.set_option("display.float_format", "{:.0f}".format)
    print("\n=== Deterministic Worst‑Block Test ===")
    print(det_summary.to_string(index=False))

    print("\n=== Bootstrap Capital Inflation ===")
    for name, lst in cap_infl.items():
        print(f"{name:8s}: {np.mean(lst)*100:.2f}%")

    det_summary.to_csv("results_summary.csv", index=False)


if __name__ == "__main__":
    main()
