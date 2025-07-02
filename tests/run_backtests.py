#!/usr/bin/env python3
"""
run_backtests.py  –  PlanFit Allocation Method · PoC
====================================================
• Dynamic sleeve-spend engine for PlanFit (cash first, then equity in each
  horizon bucket; surplus/deficit rolls to the next sleeve).
• Annual rebalancing for comparison mixes.
• 44 rolling 30-year windows (1950-79 … 1993-2022) → deterministic ≥ 95 % sizing.
• 5 000 bootstrap paths (commented out) for future Monte Carlo robustness.
• Outputs a tidy summary table

      StartCap | Ruin % | CVaR₅ | MedianMult | Efficiency

  to console and CSV.

• Generates two PNGs:
      1. frontier.png        – capital-efficiency scatter (StartCap vs CVaR₅)
      2. funded_heatmap.png  – funded-ratio heat-map for PlanFit

Usage (inside venv)
-------------------
    python Tests/run_backtests.py data/returns_1950_2022.csv \
                                 data/withdrawals.csv       \
                                 --capital 3680000
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import default_rng
from matplotlib.colors import Normalize

RNG = default_rng(42)

# ---------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------
def load_withdrawals(path: Path) -> np.ndarray:
    """Return a 1-D numpy array of real withdrawals (floats)."""
    df = pd.read_csv(path).sort_values("Year")
    col = (
        df["RealWithdrawal"]
        .astype(str)
        .str.replace(r"[,$]", "", regex=True)
        .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)  # (1000) → -1000
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
    """Yield n bootstrap paths preserving 5-year serial structure."""
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
    stk_bal, bnd_bal, csh_bal = w * cap
    funded_traj = []  # for heat-map

    for t, (r_vec, wd) in enumerate(zip(ret, cf)):
        # grow sleeves
        stk_bal *= 1 + r_vec[0]
        bnd_bal *= 1 + r_vec[1]
        csh_bal *= 1 + r_vec[2]

        # withdraw from cash then proportional from risk assets
        if wd <= csh_bal:
            csh_bal -= wd
        else:
            short = wd - csh_bal
            csh_bal = 0.0
            risk_bal = stk_bal + bnd_bal
            if risk_bal < short:
                return False, 0.0, []  # ruin
            stk_bal -= short * (stk_bal / risk_bal)
            bnd_bal -= short * (bnd_bal / risk_bal)

        # funded-ratio trajectory
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
        # withdraw proportional to current weights
        tot = stk_bal + bnd_bal + csh_bal
        if tot < wd:
            return False, 0.0
        stk_bal -= wd * (stk_bal / tot)
        bnd_bal -= wd * (bnd_bal / tot)
        csh_bal -= wd * (csh_bal / tot)
        # rebalance at year-end
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

    # expand hi until target is bracketed
    while pass_rate(hi) < target:
        hi *= 1.5
        if hi > 50e6:
            raise RuntimeError("Capital solver failed to bracket target")

    for _ in range(40):  # ≈1e-3 precision on 10 M range
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
    plt.title("Capital-Efficiency Frontier")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()


def plot_heatmap(traj: list[list[float]], start_years: list[int],
                 out="funded_heatmap.png"):
    """Heat-map clipped to 0-200 % funded; y-axis shows window start year."""
    arr = np.array(traj)
    plt.figure(figsize=(6, 6))
    im = plt.imshow(
        arr,
        aspect="auto",
        cmap="RdYlGn",
        origin="lower",
        norm=Normalize(vmin=0, vmax=2),  # 0–200 %
    )
    plt.colorbar(im, label="Funded Ratio")
    plt.xlabel("Year of Retirement Path")
    plt.ylabel("Start Year of 30-Year Window")
    # y-ticks every 5 years for readability
    tick_idx = np.arange(0, len(start_years), 5)
    plt.yticks(tick_idx, [start_years[i] for i in tick_idx])
    plt.title("PlanFit Funded-Ratio Heat-Map")
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
    parser = argparse.ArgumentParser(description="PlanFit vs static mixes")
    parser.add_argument("returns", type=Path)
    parser.add_argument("withdrawals", type=Path)
    parser.add_argument("--capital", type=float, default=3.68e6,
                        help="PlanFit starting capital")
    args = parser.parse_args()

    # Need year column for heat-map labels
    df_ret     = pd.read_csv(args.returns)
    start_year = int(df_ret["Year"].iloc[0])
    years_col  = df_ret["Year"].astype(int).to_list()
    mat        = df_ret[["Stocks", "Bonds", "Cash"]].to_numpy(dtype=float)

    cf = load_withdrawals(args.withdrawals)
    H  = len(cf)

    # PlanFit weights (Table 1): 37 % equity / 63 % cash for a 30-year horizon
    pf_w = np.array([0.37, 0.00, 0.63])

    # Strategy definitions
    planfit = Strategy("PlanFit", pf_w, args.capital, simulate_planfit)
    bond60  = Strategy("60/40 bonds", np.array([0.60, 0.40, 0.00]), 0.0, simulate_rebal)
    cash60  = Strategy("60/40 cash",  np.array([0.60, 0.00, 0.40]), 0.0, simulate_rebal)
    strategies = [planfit, bond60, cash60]

    # Solve start-capital for comparators to hit ≥95 % pass
    for strat in strategies[1:]:
        strat.start_cap = solve_capital(mat, cf, strat.sim_fn, strat.w)

    # Deterministic windows
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
            ok     += success
            endings.append(end_w)

        ruin        = 1 - ok / len(windows)
        endings     = np.array(endings)

        # ---- CVaR₅: allow negative by penalizing ruin with −1 yr withdrawal ----
        shortfall   = -cf[0]
        endings_adj = np.where(endings == 0, shortfall, endings)
        n_tail      = max(1, int(0.05 * len(endings_adj)))
        cvar5       = np.mean(np.sort(endings_adj)[: n_tail])
        efficiency  = cvar5 / strat.start_cap

        # ---- Median terminal multiplier (survivors only) ----
        survivors   = endings[endings > 0]
        median_mult = (np.median(survivors) / strat.start_cap
                       if survivors.size else 0.0)

        rows.append({
            "Strategy":    strat.name,
            "StartCap":    round(strat.start_cap, 0),
            "RuinPct":     round(ruin * 100, 2),
            "CVaR5":       round(cvar5, 0),
            "MedianMult":  round(median_mult, 3),
            "Efficiency":  round(efficiency, 3),
        })

        if strat is planfit:
            heat_rows = traj_matrix

    # === table & CSV ===
    df = pd.DataFrame(rows)
    print("\n=== Deterministic 44-window Test ===")
    print(df.to_string(index=False))
    df.to_csv("results_summary.csv", index=False)

    # === plots ===
    plot_frontier(df)
    if heat_rows:
        start_years = years_col[: len(years_col) - H + 1]
        plot_heatmap(heat_rows, start_years)

    # --- bootstrap block retained for future use (commented) ---
    # boot_infl = {s.name: [] for s in strategies}
    # for path in bootstrap_paths(mat, H, 5, 5000):
    #     for strat in strategies:
    #         extra = 0.0
    #         while not strat.sim_fn(path, strat.w,
    #                                strat.start_cap + extra, cf)[0]:
    #             extra += 0.05 * strat.start_cap
    #         boot_infl[strat.name].append(extra / strat.start_cap)
    # for name, lst in boot_infl.items():
    #     print(f"{name}: {100*np.median(lst):.2f}% bootstrap add-on")


if __name__ == "__main__":
    main()
