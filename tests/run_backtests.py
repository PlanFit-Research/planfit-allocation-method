#!/usr/bin/env python3
"""
run_backtests.py — PlanFit vs. 60/40 stress‑test harness
=======================================================
*Purpose* — Compare starting‑capital requirements of PlanFit’s aggregate SAA
versus a conventional 60/40 portfolio.

Methodology
-----------
1. **Rolling deterministic test** All 44 rolling 30‑year slices of 1950‑2022
   real returns.
2. **Capital solve** 60/40 starting capital is scaled so ≥ 95 % of those slices
   finish with non‑negative wealth.
3. **Bootstrap robustness** 5 000 synthetic 30‑year paths built from six
   5‑year blocks (with replacement) to preserve short‑run serial structure.

The script prints a summary table and writes `results_summary.csv`.

Usage
-----
    python Tests/run_backtests.py data/returns_1950_2022.csv \
                                   data/withdrawals.csv      \
                                   --capital 3300000
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import default_rng

SEED = 42
RNG = default_rng(SEED)
HORIZON = 30                    # years per path
a_tol = 5_000.0                 # incremental search step for extra capital

# ---------------------------------------------------------------------------
# Dataclass for strategies
# ---------------------------------------------------------------------------
@dataclass
class Strategy:
    name: str
    weights: tuple[float, float, float]  # (stocks, bonds, cash)
    start_capital: float

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_returns(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    return df[["Stocks", "Bonds", "Cash"]].to_numpy(dtype="float64")

def load_withdrawals(path: Path) -> np.ndarray:
    """Return a float vector. Handles $‑signs, commas, (negatives)."""
    df = pd.read_csv(path).sort_values("Year")
    clean = (
        df["RealWithdrawal"].astype(str)
        .str.replace(r"[,$]", "", regex=True)              # kill $ and commas
        .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)  # (1000) -> -1000
        .str.strip()
        .astype("float64")
    )
    return clean.to_numpy()

# ---------------------------------------------------------------------------
# Core path simulator
# ---------------------------------------------------------------------------

def simulate_path(ret_block: np.ndarray, weights: tuple[float, float, float],
                  capital: float, cashflows: np.ndarray) -> tuple[bool, float]:
    wealth = capital
    for r_vec, cf in zip(ret_block, cashflows):
        wealth = wealth * (1 + np.dot(weights, r_vec)) - cf
        if wealth < 0:
            return False, wealth
    return True, wealth

# ---------------------------------------------------------------------------
# Rolling windows & bootstrap generator
# ---------------------------------------------------------------------------

def rolling_windows(mat: np.ndarray, h: int = HORIZON):
    n = mat.shape[0]
    for i in range(0, n - h + 1):
        yield i, i + h

def bootstrap_paths(mat: np.ndarray, block: int = 5, n: int = 5000):
    nrows = mat.shape[0]
    n_blocks = HORIZON // block
    for _ in range(n):
        idx_start = RNG.integers(0, nrows - block + 1, size=n_blocks)
        yield np.concatenate([mat[s : s + block] for s in idx_start])

# ---------------------------------------------------------------------------
# Capital solver for 60/40
# ---------------------------------------------------------------------------

def solve_benchmark_capital(mat: np.ndarray, cashflows: np.ndarray,
                            weights: tuple[float, float, float], target=0.95):
    slices = list(rolling_windows(mat))

    def success_rate(cap):
        succ = sum(simulate_path(mat[s:e], weights, cap, cashflows)[0]
                    for s, e in slices)
        return succ / len(slices)

    cap = 100_000.0
    while success_rate(cap) < target:
        cap += 100_000.0
        if cap > 20_000_000:
            raise RuntimeError("Benchmark capital blew past $20M without success")
    return cap

# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PlanFit vs 60/40 stress‑test")
    ap.add_argument("returns", type=Path)
    ap.add_argument("withdrawals", type=Path)
    ap.add_argument("--capital", type=float, default=3.3e6)
    args = ap.parse_args()

    ret_mat   = load_returns(args.returns)
    cashflows = load_withdrawals(args.withdrawals)

    pf  = Strategy("PlanFit", (0.37, 0.00, 0.63), args.capital)
    ben = Strategy("60/40",  (0.60, 0.40, 0.00), 0.0)
    ben.start_capital = solve_benchmark_capital(ret_mat, cashflows, ben.weights)

    def deterministic_metrics(strat: Strategy):
        endings = [simulate_path(ret_mat[s:e], strat.weights,
                                 strat.start_capital, cashflows)[1]
                    for s, e in rolling_windows(ret_mat)]
        endings = np.array(endings)
        cvar5 = endings[np.argsort(endings)][: max(1, int(0.05*len(endings)))].mean()
        return cvar5, endings.mean()

    rows = []
    for strat in (pf, ben):
        cvar5, mean_end = deterministic_metrics(strat)
        rows.append([strat.name, strat.start_capital, cvar5, mean_end])

    summary = pd.DataFrame(rows, columns=["Strategy","StartCap","CVaR5","MeanEnd"])
    print("\n=== Deterministic Worst‑Block Test ===")
    print(summary.to_string(index=False, float_format="%.0f"))

    # Bootstrap capital inflation
    cap_infl = {pf.name: [], ben.name: []}
    for path in bootstrap_paths(ret_mat):
        for strat in (pf, ben):
            extra = 0.0
            while not simulate_path(path, strat.weights,
                                     strat.start_capital + extra, cashflows)[0]:
                extra += a_tol
                if extra > 2e7:
                    raise RuntimeError("Bootstrap path never succeeds even +$20M")
            cap_infl[strat.name].append(extra / strat.start_capital)

    print("\n=== Bootstrap Capital Inflation (5‑yr blocks) ===")
    for name, vec in cap_infl.items():
        print(f"{name:<8}: {np.mean(vec)*100:.2f}%")

if __name__ == "__main__":
    main()
