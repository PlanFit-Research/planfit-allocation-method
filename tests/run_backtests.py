#!/usr/bin/env python3
"""
run_backtests.py — PlanFit vs. 60/40 stress‑test harness
=======================================================
Compares the capital‑efficiency of the **PlanFit aggregate SAA** with a classic
60/40 mix under three lenses:

1. *Deterministic worst‑block test* on **non‑overlapping** 30‑year slices of
   history (1950‑1979, 1980‑2009).
2. *5 000 bootstrap* resamples of 30 annual return draws (with replacement).
3. *Five‑year inflation shock* (optional stub — left for future work).

Input files
-----------
* `returns_1950_2022.csv` — columns **Year, Stocks, Bonds, Cash** (real)
* `withdrawals.csv`      — columns **Year, RealWithdrawal** (real dollars)

Usage
-----
```bash
python Tests/run_backtests.py data/returns_1950_2022.csv \
                             data/withdrawals.csv        \
                             --capital 3300000
```
The script prints a summary table and writes `results_summary.csv` for the
manuscript.
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

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    name: str
    weights: tuple[float, float, float]  # (stocks, bonds, cash)
    start_capital: float                 # gets solved for benchmark


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_returns(path: Path) -> np.ndarray:
    """Return Nx3 ndarray of real returns for Stocks, Bonds, Cash."""
    df = pd.read_csv(path)
    req = {"Stocks", "Bonds", "Cash"}
    if not req.issubset(df.columns):
        raise ValueError(f"{path} must have cols {req}")
    return df["Stocks"].to_numpy(), df["Bonds"].to_numpy(), df["Cash"].to_numpy()


def load_withdrawals(path: Path) -> np.ndarray:
    df = pd.read_csv(path).sort_values("Year")
    if "RealWithdrawal" not in df.columns:
        raise ValueError("withdrawals CSV must have column RealWithdrawal")
    return df["RealWithdrawal"].to_numpy()

# ---------------------------------------------------------------------------
# Core mechanics (position‑based)
# ---------------------------------------------------------------------------

def simulate_path(
    returns_block: np.ndarray,
    weights: tuple[float, float, float],
    capital: float,
    cashflows: np.ndarray,
) -> tuple[bool, float]:
    """Run one path; returns_block shape = (H,3). cashflows len = H."""
    wealth = capital
    for r_vec, cf in zip(returns_block, cashflows):
        r = np.dot(weights, r_vec)
        wealth = wealth * (1 + r) - cf
        if wealth < 0:
            return False, wealth
    return True, wealth


def worst_block_slices(mat: np.ndarray, block_len: int) -> list[tuple[int, int]]:
    """Yield non‑overlapping POS slices (start, end) length *block_len*."""
    n = mat.shape[0]
    return [(i, i + block_len) for i in range(0, n - block_len + 1, block_len)]

# ---------------------------------------------------------------------------
# Capital solver for benchmark 60/40
# ---------------------------------------------------------------------------

def solve_benchmark_capital(
    ret_mat: np.ndarray,
    cashflows: np.ndarray,
    weights: tuple[float, float, float],
    target_success: float = 0.95,
):
    slices = worst_block_slices(ret_mat, len(cashflows))

    def success_gap(capital: float):
        succ = sum(
            simulate_path(ret_mat[s:e], weights, capital, cashflows)[0]
            for s, e in slices
        )
        return succ / len(slices) - target_success

    return brentq(success_gap, 0.1e6, 20e6)

# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------

def bootstrap_paths(ret_mat: np.ndarray, horizon: int, n: int = 5000):
    rows = ret_mat.shape[0]
    for _ in range(n):
        idx = RNG.choice(rows, size=horizon, replace=True)
        yield ret_mat[idx]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="PlanFit benchmark & stress‑tests")
    p.add_argument("returns", type=Path)
    p.add_argument("withdrawals", type=Path)
    p.add_argument("--capital", type=float, default=3.3e6,
                   help="PlanFit starting capital")
    args = p.parse_args()

    # ---- load data ----
    stocks, bonds, cash = load_returns(args.returns)
    ret_mat = np.column_stack([stocks, bonds, cash])
    cashflows = load_withdrawals(args.withdrawals)
    H = len(cashflows)

    # ---- define strategies ----
    planfit = Strategy("PlanFit", (0.37, 0.00, 0.63), args.capital)
    benchmark = Strategy("60/40", (0.60, 0.40, 0.00), 0.0)

    # ---- solve benchmark capital ----
    benchmark.start_capital = solve_benchmark_capital(ret_mat, cashflows, benchmark.weights)

    # ---- deterministic test ----
    slices = worst_block_slices(ret_mat, H)
    summary_rows = []
    for strat in [planfit, benchmark]:
        endings = [
            simulate_path(ret_mat[s:e], strat.weights, strat.start_capital, cashflows)[1]
            for s, e in slices
        ]
        endings = np.array(endings)
        cvar5 = endings[np.argsort(endings)][: max(1, int(0.05 * len(endings)))].mean()
        summary_rows.append({
            "Strategy": strat.name,
            "StartCapital": strat.start_capital,
            "CVaR5": cvar5,
            "MeanEnd": endings.mean(),
        })

    summary = pd.DataFrame(summary_rows)
    print("\n=== Deterministic Worst‑Block Test ===")
    print(summary.to_string(index=False, float_format="%.0f"))

    # ---- bootstrap ----
    cap_infl = {planfit.name: [], benchmark.name: []}
    for bpath in bootstrap_paths(ret_mat, H, n=5000):
        for strat in [planfit, benchmark]:
            ok, w_end = simulate_path(bpath, strat.weights, strat.start_capital, cashflows)
            if ok:
                cap_infl[strat.name].append(0.0)
            else:
                extra = brentq(
                    lambda x: simulate_path(bpath, strat.weights, strat.start_capital + x, cashflows)[0] - 1,
                    0, 2e6,
                )
                cap_infl[strat.name].append(extra / strat.start_capital)

    print("\n=== Bootstrap Capital Inflation (5000 resamples) ===")
    for k, lst in cap_infl.items():
        print(f"{k}: {np.mean(lst)*100:.2f}% avg cap increase")

    summary.to_csv("results_summary.csv", index=False)
    print("\nWrote results_summary.csv")


if __name__ == "__main__":
    main()
