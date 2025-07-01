#!/usr/bin/env python3
"""
run_backtests.py  –  PlanFit PoC   ·   60/40 Bonds   ·   60/40 Cash
====================================================================
• Dynamic sleeve‑spend engine for PlanFit (cash first, then equity in each
  horizon bucket; surplus/deficit rolls to the next sleeve).
• Annual rebalancing for comparison mixes.
• 44 rolling 30‑year windows (1950‑79 … 1993‑2022) → deterministic ≥ 95 % sizing.
• 5 000 bootstrap paths built from six 5‑year blocks (preserves serial
  structure).  *Bootstrap code kept but commented out by default.*
• Outputs a tidy summary table **StartCap | Ruin % | CVaR₅ | Efficiency** to
  console and CSV.
• Generates two PNGs by default:
    1. **frontier.png** – capital‑efficiency scatter (StartCap vs CVaR₅)
    2. **funded_heatmap.png** – funded‑ratio heat‑map for PlanFit (rows = 44
       windows, cols = years 1‑30).

Usage (inside venv)
-------------------
    python Tests/run_backtests.py data/returns_1950_2022.csv \
                                 data/withdrawals.csv       \
                                 --capital 3680000
"""
from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import default_rng

RNG = default_rng(42)

# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------

def load_returns(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    return df[["Stocks", "Bonds", "Cash"]].to_numpy(dtype=float)


def load_withdrawals(path: Path) -> np.ndarray:
    df = pd.read_csv(path).sort_values("Year")
    col = (
        df["RealWithdrawal"]
        .astype(str)
        .str.replace(r"[,$]", "", regex=True)
        .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)  # (1000) -> -1000
        .str.strip()
        .astype(float)
    )
    return col.to_numpy()

# ---------------------------------------------------------------------
# Helper – rolling windows & bootstrap paths
# ---------------------------------------------------------------------

def rolling_windows(mat: np.ndarray, h: int = 30):
    n = mat.shape[0]
    for i in range(0, n - h + 1):
        yield mat[i : i + h]


def bootstrap_paths(mat: np.ndarray, h: int = 30, block: int = 5, n: int = 5000):
    nrows = mat.shape[0]
    n_blocks = h // block
    for _ in range(n):
        idx = RNG.choice(range(nrows - block + 1), size=n_blocks)
        rows = np.concatenate([mat[i : i + block] for i in idx], axis=0)
        yield rows

# ---------------------------------------------------------------------
# Simulation engines
# ---------------------------------------------------------------------

def simulate_planfit(ret: np.ndarray, w: np.ndarray, cap: float, cf: np.ndarray):
    """Dynamic sleeve spend – cash first, then equity inside each horizon."""
    # initialise sleeves
    stk_bal = w[0] * cap
    bnd_bal = w[1] * cap
    csh_bal = w[2] * cap

    funded_traj = []  # for heat‑map

    for t, (r_vec, wd) in enumerate(zip(ret, cf)):
        # grow sleeves
        stk_bal *= 1 + r_vec[0]
        bnd_bal *= 1 + r_vec[1]
        csh_bal *= 1 + r_vec[2]

        # withdraw from cash then equity (stocks+bonds proportionally)
        if wd <= csh_bal:
            csh_bal -= wd
        else:
            short = wd - csh_bal
            csh_bal = 0.0
            risk_bal = stk_bal + bnd_bal
            if risk_bal < short:
                return False, 0.0, []  # ruin
            # proportional depletion
            if risk_bal > 0:
                stk_bal -= short * (stk_bal / risk_bal)
                bnd_bal -= short * (bnd_bal / risk_bal)

        # funded ratio for heat‑map
        rem_liab = cf[t:].sum()
        funded = (stk_bal + bnd_bal + csh_bal) / rem_liab if rem_liab else 0
        funded_traj.append(funded)

    return True, stk_bal + bnd_bal + csh_bal, funded_traj


def simulate_rebal(ret: np.ndarray, w: np.ndarray, cap: float, cf: np.ndarray):
    """Comparison mix with annual perfect rebalance."""
    stk_bal, bnd_bal, csh_bal = w * cap
    for r_vec, wd in zip(ret, cf):
        # grow
        stk_bal *= 1 + r_vec[0]
        bnd_bal *= 1 + r_vec[1]
        csh_bal *= 1 + r_vec[2]
        # withdraw proportional to current weights (after growth)
        tot = stk_bal + bnd_bal + csh_bal
        if tot < wd:
            return False, 0.0
        stk_bal -= wd * (stk_bal / tot)
        bnd_bal -= wd * (bnd_bal / tot)
        csh_bal -= wd * (csh_bal / tot)
        # rebalance to target weights at year‑end
        tot = stk_bal + bnd_bal + csh_bal
        stk_bal, bnd_bal, csh_bal = w * tot
    return True, stk_bal + bnd_bal + csh_bal

# ---------------------------------------------------------------------
# Capital solver (bisection)
# ---------------------------------------------------------------------

def solve_capital(mat: np.ndarray, cf: np.ndarray, sim_fn, w: np.ndarray,
                 target=0.95, lo=0.1e6, hi=10e6):
    windows = list(rolling_windows(mat, len(cf)))

    def pass_rate(cap):
        ok = sum(sim_fn(win, w, cap, cf)[0] for win in windows)
        return ok / len(windows)

    # Expand hi until pass_rate(lo) < target < pass_rate(hi)
    while pass_rate(hi) < target:
        hi *= 1.5
        if hi > 50e6:
            raise RuntimeError("Capital solver failed to bracket target")

    for _ in range(40):  # ≈ 1e-3 precision on 10‑M range
        mid = 0.5 * (lo + hi)
        if pass_rate(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi

# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def plot_frontier(df: pd.DataFrame, out="frontier.png"):
    plt.figure(figsize=(6, 4))
    plt.scatter(df["StartCap"], df["CVaR5"], s=60)
    for _, row in df.iterrows():
        plt.annotate(row["Strategy"], (row["StartCap"], row["CVaR5"]),
                     textcoords="offset points", xytext=(5, -5))
    plt.xlabel("Start Capital ($)")
    plt.ylabel("CVaR₅ ($)")
    plt.title("Capital‑Efficiency Frontier")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def plot_heatmap(traj: list[list[float]], out="funded_heatmap.png"):
    arr = np.array(traj)
    plt.figure(figsize=(6, 6))
    plt.imshow(arr, aspect="auto", cmap="RdYlGn", origin="lower")
    plt.colorbar(label="Funded Ratio")
    plt.xlabel("Year of Retirement Path")
    plt.ylabel("44 Rolling Windows (1950‑79 … 1993‑2022)")
    plt.title("PlanFit Funded‑Ratio Heat‑Map")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

@dataclass
class Strategy:
    name: str
    w: np.ndarray
    start_cap: float
    sim_fn: callable


def main():
    parser = argparse.ArgumentParser(description="PlanFit vs RTQ proxies")
    parser.add_argument("returns", type=Path)
    parser.add_argument("withdrawals", type=Path)
    parser.add_argument("--capital", type=float, default=3.68e6,
                        help="PlanFit starting capital")
    args = parser.parse_args()

    mat = load_returns(args.returns)
    cf  = load_withdrawals(args.withdrawals)
    H   = len(cf)

    # PlanFit weights from Table 1 for H=30 ⇒ 37 % equity / 63 % cash
    pf_w = np.array([0.37, 0.0, 0.63])

    # Strategy definitions (start_cap for comparators solved below)
    planfit = Strategy("PlanFit", pf_w, args.capital, simulate_planfit)
    bond60  = Strategy("60/40 bonds", np.array([0.60, 0.40, 0.0]), 0.0, simulate_rebal)
    cash60  = Strategy("60/40 cash",  np.array([0.60, 0.00, 0.40]), 0.0, simulate_rebal)
    strategies = [planfit, bond60, cash60]

    # Solve start‑capital for comparators to hit ≥ 95 % pass
    for strat in strategies[1:]:
        strat.start_cap = solve_capital(mat, cf, strat.sim_fn, strat.w)

    # Run deterministic windows and gather funded trajectories for PlanFit
    windows = list(rolling_windows(mat, H))
    rows, heat_rows = [], []
    for strat in strategies:
        ok, endings = 0, []
        traj_matrix = []
        for win in windows:
            if strat is planfit:
                success, end_w, traj = strat.sim_fn(win, strat.w, strat.start_cap, cf)
                traj_matrix.append(traj)
            else:
                success, end_w = strat.sim_fn(win, strat.w, strat.start_cap, cf)
            ok += success
            endings.append(end_w)
        ruin = 1 - ok / len(windows)
        endings = np.array(endings)
        cvar5 = np.mean(np.sort(endings)[: max(1, int(0.05 * len(endings)))])
        efficiency = cvar5 / strat.start_cap if strat.start
