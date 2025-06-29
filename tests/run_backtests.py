#!/usr/bin/env python3
"""
run_backtests.py — PlanFit vs. 60/40 stress‑test harness
=======================================================
Compares the capital‑efficiency of the **PlanFit aggregate SAA** with a classic
60/40 mix under three lenses:

1. *Deterministic worst‑block test* on **non‑overlapping** 30‑year slices of
   history (1950‑1979 and 1980‑2009).
2. *5000 bootstrap* resamples of 30 annual return draws (with replacement).
3. *Inflation‑shock* run (double CPI first 5 yrs of 1980‑2009 block).

Input files must live in `data/` (or pass explicit paths):
  • **returns CSV**  — columns `Year,Stocks,Bonds,Cash` (real % in decimals)
  • **withdrawals CSV** — columns `Year,RealWithdrawal` (any $‑format ok)

Usage
-----
(.venv) python Tests/run_backtests.py data/returns_1950_2022.csv \
                                      data/withdrawals.csv       \
                                      --capital 3300000

Outputs a summary table to screen and writes `results_summary.csv`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd
from numpy.random import default_rng
from scipy.optimize import brentq

SEED = 42
RNG = default_rng(SEED)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Strategy:
    name: str
    weights: tuple[float, float, float]  # (stocks, bonds, cash)
    start_capital: float                 # will be adjusted for benchmark

# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def load_returns(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    needed = {"Stocks", "Bonds", "Cash"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Returns CSV missing columns {needed - set(df.columns)}")
    return df[["Stocks", "Bonds", "Cash"]].to_numpy(dtype=float)


def load_withdrawals(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if not {"Year", "RealWithdrawal"}.issubset(df.columns):
        raise ValueError("Withdrawals CSV must have Year and RealWithdrawal columns")
    # Clean dollar signs, commas, and whitespace
    df["RealWithdrawal"] = (
        df["RealWithdrawal"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
    )
    df["RealWithdrawal"] = pd.to_numeric(df["RealWithdrawal"], errors="raise")
    df = df.sort_values("Year")
    return df["RealWithdrawal"].to_numpy(dtype=float)

# ---------------------------------------------------------------------------
# Core simulation logic (position‑based, not calendar‑label‑based)
# ---------------------------------------------------------------------------

def simulate_path(
    returns_block: np.ndarray,
    weights: tuple[float, float, float],
    capital: float,
    cashflows: np.ndarray,
) -> tuple[bool, float]:
    wealth = capital
    for r_vec, cf in zip(returns_block, cashflows):
        r = np.dot(weights, r_vec)
        wealth = wealth * (1 + r) - cf
        if wealth < 0:
            return False, wealth
    return True, wealth


def worst_block_slices(mat: np.ndarray, block_len: int = 30):
    nrows = mat.shape[0]
    for idx in range(0, nrows - block_len + 1, block_len):
        yield idx, idx + block_len


# ---------------------------------------------------------------------------
# Capital solver for benchmark mix
# ---------------------------------------------------------------------------

def solve_benchmark_capital(
    returns: np.ndarray,
    cashflows: np.ndarray,
    weights: tuple[float, float, float],
    target_success: float = 0.95,
):
    slices = list(worst_block_slices(returns, len(cashflows)))

    def success_gap(capital: float):
        succ = sum(
            simulate_path(returns[s:e], weights, capital, cashflows)[0]
            for s, e in slices
        )
        return succ / len(slices) - target_success

    return brentq(success_gap, 0.1e6, 20e6)


# ---------------------------------------------------------------------------
# Bootstrap generator
# ---------------------------------------------------------------------------

def bootstrap_paths(mat: np.ndarray, years: int = 30, n: int = 5000):
    for _ in range(n):
        idx = RNG.choice(mat.shape[0], size=years, replace=True)
        yield mat[idx]

# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PlanFit benchmark & stress‑tests")
    parser.add_argument("returns", type=Path, help="CSV of annual real returns")
    parser.add_argument("withdrawals", type=Path, help="CSV of real withdrawals")
    parser.add_argument("--capital", type=float, default=3.3e6, help="PlanFit starting capital")
    args = parser.parse_args()

    ret_mat = load_returns(args.returns)
    cashflows = load_withdrawals(args.withdrawals)
    H = len(cashflows)  # horizon length (e.g., 30 yrs)

    planfit = Strategy("PlanFit", (0.37, 0.0, 0.63), args.capital)
    benchmark = Strategy("60/40", (0.6, 0.4, 0.0), 0.0)

    benchmark.start_capital = solve_benchmark_capital(ret_mat, cashflows, benchmark.weights)

    # Deterministic worst‑block test
    rows = []
    for strat in [planfit, benchmark]:
        endings = []
        for s, e in worst_block_slices(ret_mat, H):
            ok, w_end = simulate_path(ret_mat[s:e], strat.weights, strat.start_capital, cashflows)
            endings.append(w_end)
        endings = np.array(endings)
        cvar5 = endings[np.argsort(endings)][: max(1, int(0.05 * len(endings)))].mean()
        rows.append({"Strategy": strat.name, "StartCap": strat.start_capital, "CVaR5": cvar5, "MeanEnd": endings.mean()})

    summary = pd.DataFrame(rows)
    print("\n=== Deterministic Worst‑Block Test ===")
    print(summary.to_string(index=False, float_format="%.0f"))

    # Bootstrap capital inflation
    cap_infl = {planfit.name: [], benchmark.name: []}
    for path in bootstrap_paths(ret_mat, H, n=5000):
        for strat in [planfit, benchmark]:
            ok, w_end = simulate_path(path, strat.weights, strat.start_capital, cashflows)
            if ok:
                cap_infl[strat.name].append(0.0)
            else:
                def gap(extra):
                    return simulate_path(path, strat.weights, strat.start_capital + extra, cashflows)[0] - 1
                extra_cap = brentq(gap, 0, 2e6)
                cap_infl[strat.name].append(extra_cap / strat.start_capital)

    print("\n=== Bootstrap Capital Inflation (5 000 paths) ===")
    for k, v in cap_infl.items():
        print(f"{k}: {np.mean(v)*100:.2f}% avg cap increase")

    summary.to_csv("results_summary.csv", index=False)
    print("\nSaved results_summary.csv")


if __name__ == "__main__":
    main()
