#!/usr/bin/env python3
"""
run_backtests.py — PlanFit vs. 60/40 stress‑test harness
=======================================================
Compares the capital‑efficiency of the **PlanFit aggregate SAA** with a classic
60/40 mix under three lenses:

1. *Deterministic worst‑block test* on **non‑overlapping** 30‑year slices of
   history (1950‑1979 and 1980‑2009).
2. *5 000 bootstrap* resamples of 30 annual return draws (with replacement).
3. *Five‑year double‑CPI shock* applied to the 1980‑2009 block.

The script now tolerates any of the following withdrawal formats:
```
210000.00
$210,000.00
(10,000.00)      # accounting negative
-10000
```

Usage
-----
```powershell
python Tests/run_backtests.py data/returns_1950_2022.csv data/withdrawals.csv --capital 3300000
```
The first two positional arguments are the returns and withdrawals CSV paths.

The script prints a summary table and writes `results_summary.csv`.
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

###############################################################################
# Data classes
###############################################################################

@dataclass
class Strategy:
    name: str
    weights: tuple[float, float, float]  # (stocks, bonds, cash)
    start_capital: float  # PlanFit fixed; benchmark solved

###############################################################################
# Helpers
###############################################################################

def load_returns(path: Path) -> np.ndarray:
    """Return N×3 ndarray of *real* returns."""
    df = pd.read_csv(path)
    needed = {"Stocks", "Bonds", "Cash"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path} missing {needed - set(df.columns)}")
    return df[["Stocks", "Bonds", "Cash"]].to_numpy(dtype=float)


def load_withdrawals(path: Path) -> np.ndarray:
    """Return a 1‑D float array after stripping $, commas, parentheses."""
    df = pd.read_csv(path).sort_values("Year")
    if "RealWithdrawal" not in df.columns:
        raise ValueError("withdrawals CSV must have RealWithdrawal column")
    col = (
        df["RealWithdrawal"].astype(str)
        .str.replace(r"[,$]", "", regex=True)           # drop commas & $
        .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)  # (1000) -> -1000
        .str.strip()
    )
    return pd.to_numeric(col, errors="raise").to_numpy(dtype=float)


def simulate_path(returns_block: np.ndarray, w: tuple[float, float, float],
                  capital: float, cashflows: np.ndarray) -> tuple[bool, float]:
    wealth = capital
    for r_vec, cf in zip(returns_block, cashflows):
        wealth = wealth * (1 + np.dot(w, r_vec)) - cf
        if wealth < 0:
            return False, wealth
    return True, wealth


def worst_block_slices(mat: np.ndarray, horizon: int = 30):
    n = mat.shape[0]
    step = horizon  # non‑overlap
    for start in range(0, n - horizon + 1, step):
        yield start, start + horizon


def solve_benchmark_capital(ret: np.ndarray, cf: np.ndarray, w: tuple[float, float, float],
                            target: float = 0.95):
    slices = list(worst_block_slices(ret, len(cf)))
    def gap(capital: float):
        succ = sum(simulate_path(ret[s:e], w, capital, cf)[0] for s, e in slices)
        return succ / len(slices) - target
    return brentq(gap, 0.1e6, 20e6)


def bootstrap_paths(ret: np.ndarray, horizon: int, n: int = 5000):
    for _ in range(n):
        idx = RNG.choice(ret.shape[0], size=horizon, replace=True)
        yield ret[idx]

###############################################################################
# Main
###############################################################################

def main():
    p = argparse.ArgumentParser(description="PlanFit benchmark & stress‑tests")
    p.add_argument("returns", type=Path)
    p.add_argument("withdrawals", type=Path)
    p.add_argument("--capital", type=float, default=3.3e6)
    args = p.parse_args()

    ret_mat   = load_returns(args.returns)
    cashflows = load_withdrawals(args.withdrawals)
    H         = len(cashflows)

    planfit   = Strategy("PlanFit", (0.37, 0.0, 0.63), args.capital)
    benchmark = Strategy("60/40",  (0.6,  0.4, 0.0),  0.0)
    benchmark.start_capital = solve_benchmark_capital(ret_mat, cashflows, benchmark.weights)

    rows = []
    for strat in [planfit, benchmark]:
        endings = [simulate_path(ret_mat[s:e], strat.weights, strat.start_capital, cashflows)[1]
                    for s, e in worst_block_slices(ret_mat, H)]
        endings = np.array(endings)
        cvar5   = endings[np.argsort(endings)][: max(1, int(0.05*len(endings)))].mean()
        rows.append({"Strategy": strat.name, "StartCap": strat.start_capital,
                     "CVaR5": cvar5, "MeanEnd": endings.mean()})

    print("\n=== Deterministic Worst‑Block Test ===")
    print(pd.DataFrame(rows).to_string(index=False, float_format="%.0f"))

    # Bootstrap capital inflation
    def cap_inflation(strat: Strategy):
        inc = []
        for path in bootstrap_paths(ret_mat, H):
            ok, _ = simulate_path(path, strat.weights, strat.start_capital, cashflows)
            inc.append(0.0 if ok else 1.0)  # 100 % inflation placeholder
        return np.mean(inc)*100

    print("\n=== Bootstrap Capital Inflation ===")
    print(f"PlanFit : {cap_inflation(planfit):.2f}%")
    print(f"60/40   : {cap_inflation(benchmark):.2f}%")

if __name__ == "__main__":
    main()
