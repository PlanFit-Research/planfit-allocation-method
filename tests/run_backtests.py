#!/usr/bin/env python3
"""
run_backtests.py — PlanFit vs. 60/40 stress‑test harness
=======================================================
Compares the capital‑efficiency of the **PlanFit aggregate SAA** with a classic
60/40 mix.  The test uses

* **44 rolling 30‑year windows** (1950‑79 … 1993‑2022) for deterministic
  sizing.
* **5 000 bootstrap paths** created from six 5‑year blocks (with replacement)
  to retain short‑run serial structure.

Outputs a summary print‑table and writes `results_summary.csv`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq

RNG = np.random.default_rng(42)
HORIZON = 30        # yrs
BOOT_N = 5_000
BLOCK = 5           # yrs per bootstrap block

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_returns(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if not {"Stocks", "Bonds", "Cash"}.issubset(df.columns):
        raise ValueError("returns CSV missing Stocks/Bonds/Cash columns")
    return df[["Stocks", "Bonds", "Cash"]].to_numpy(dtype=float)


def load_withdrawals(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if not {"Year", "RealWithdrawal"}.issubset(df.columns):
        raise ValueError("withdrawals CSV needs Year & RealWithdrawal")
    col = (
        df["RealWithdrawal"].astype(str)
        .str.replace(r"[\\$,]", "", regex=True)      # drop $ and commas
        .str.replace(r"\\(([^)]+)\\)", r"-\\1", regex=True)  # (10000) → -10000
        .str.strip()
        .astype(float)
    )
    return col.to_numpy(dtype=float)

# ---------------------------------------------------------------------------
# Path simulation
# ---------------------------------------------------------------------------

def simulate_path(ret_block: np.ndarray, weights: tuple[float, float, float],
                  capital: float, cashflows: np.ndarray) -> tuple[bool, float]:
    wealth = capital
    for r_vec, cf in zip(ret_block, cashflows):
        r = np.dot(weights, r_vec)
        wealth = wealth * (1 + r) - cf
        if wealth < 0:
            return False, wealth
    return True, wealth

# ---------------------------------------------------------------------------
# Rolling windows & bootstrap generators
# ---------------------------------------------------------------------------

def rolling_windows(mat: np.ndarray, h: int = HORIZON):
    n = mat.shape[0]
    for start in range(0, n - h + 1):
        yield start, start + h


def bootstrap_paths(mat: np.ndarray, block: int = BLOCK, h: int = HORIZON,
                    n: int = BOOT_N):
    n_blocks = h // block
    nrows = mat.shape[0]
    for _ in range(n):
        idx_blocks = RNG.choice(range(nrows - block + 1), size=n_blocks)
        rows = np.concatenate([mat[i : i + block] for i in idx_blocks], axis=0)
        yield rows

# ---------------------------------------------------------------------------
# Capital sizing helpers
# ---------------------------------------------------------------------------

def solve_benchmark_capital(returns: np.ndarray, cashflows: np.ndarray,
                             weights: tuple[float, float, float]) -> float:
    slices = list(rolling_windows(returns))

    def success_rate(cap):
        succ = sum(
            simulate_path(returns[s:e], weights, cap, cashflows)[0]
            for s, e in slices
        )
        return succ / len(slices) - 0.95

    return brentq(success_rate, 0.1e6, 20e6)

# ---------------------------------------------------------------------------
# Dataclass for strategy container
# ---------------------------------------------------------------------------

dataclass
class Strategy:
    name: str
    weights: tuple[float, float, float]
    start_capital: float
    cvar5: float | None = None
    mean_end: float | None = None
    boot_inflation: float | None = None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PlanFit vs 60/40 test")
    parser.add_argument("returns", type=Path, help="returns CSV path")
    parser.add_argument("withdrawals", type=Path, help="withdrawals CSV path")
    parser.add_argument("--capital", type=float, default=3.3e6,
                        help="PlanFit starting capital (default 3.3M)")
    args = parser.parse_args()

    ret_mat = load_returns(args.returns)
    cashflows = load_withdrawals(args.withdrawals)

    planfit = Strategy("PlanFit", (0.37, 0.0, 0.63), args.capital)
    benchmark = Strategy("60/40", (0.6, 0.4, 0.0), 0.0)
    benchmark.start_capital = solve_benchmark_capital(ret_mat, cashflows,
                                                     benchmark.weights)

    # ---------- deterministic metrics ----------
    slices = list(rolling_windows(ret_mat))
    for strat in (planfit, benchmark):
        endings = [simulate_path(ret_mat[s:e], strat.weights,
                                 strat.start_capital, cashflows)[1]
                    for s, e in slices]
        cutoff = np.percentile(endings, 5)
        strat.cvar5 = np.mean([v for v in endings if v <= cutoff])
        strat.mean_end = np.mean(endings)

    # ---------- bootstrap robustness ----------
    cap_increases = {"PlanFit": [], "60/40": []}

    for path in bootstrap_paths(ret_mat):
        for strat in (planfit, benchmark):
            extra = 0.0
            while not simulate_path(path, strat.weights,
                                     strat.start_capital + extra,
                                     cashflows)[0]:
                extra += 0.05 * strat.start_capital  # 5 % increments
                if extra > 2e7:
                    raise RuntimeError("Cap solve blew up > $20M")
            cap_increases[strat.name].append(extra / strat.start_capital)

    planfit.boot_inflation = 100 * np.median(cap_increases["PlanFit"])
    benchmark.boot_inflation = 100 * np.median(cap_increases["60/40"])

    # ---------- print summary ----------
    print("\n=== Deterministic Worst‑Block Test ===")
    print(f"Strategy  StartCap   CVaR5  MeanEnd")
    for s in (planfit, benchmark):
        print(f"{s.name:8} {s.start_capital:9,.0f} {s.cvar5:7,.0f} {s.mean_end:9,.0f}")

    print("\n=== Bootstrap Capital Inflation ===")
    print(f"PlanFit : {planfit.boot_inflation:.2f}%")
    print(f"60/40   : {benchmark.boot_inflation:.2f}%")

    # optional CSV
    pd.DataFrame([
        {
            "Strategy": s.name,
            "StartCap": s.start_capital,
            "CVaR5": s.cvar5,
            "MeanEnd": s.mean_end,
            "Boot_Median_%": s.boot_inflation,
        }
        for s in (planfit, benchmark)
    ]).to_csv("results_summary.csv", index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
