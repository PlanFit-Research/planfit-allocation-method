#!/usr/bin/env python3
"""
run_backtests.py  –  PlanFit Allocation Method · Proof of Concept
=================================================================
• Dynamic sleeve-spend engine for PlanFit (cash first, then equity per bucket).
• Static 60/40 comparators (bonds & cash sleeves) with pro-rata withdrawals
  and annual rebalancing.
• 44 rolling 30-year windows (1950-79 … 1993-2022).

  TWO SIZING MODES
  ----------------
  1. Equal-Probability (default) –– Each strategy is sized to clear a 95 %
     historical success bar.  Invoke with **no --fixed flag**.
  2. Capital-Constraint            –– All strategies start with the same
     user-supplied --capital.      Invoke with **--fixed --capital X**.

• Ruin severity = remaining retirement liability at failure, so CVaR₅ and
  MedianMult account for early- vs late-ruin.
• Outputs table:  StartCap | Ruin % | CVaR₅ | MedianTW | MedianMult | Efficiency
• Generates frontier.png  +  funded_heatmap.png (0-200 % funded scale).
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from numpy.random import default_rng

RNG = default_rng(42)

# ---------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------
def load_withdrawals(path: Path) -> np.ndarray:
    """Return a 1-D numpy array of real withdrawals (floats)."""
    df = pd.read_csv(path).sort_values("Year")
    col = (
        df["RealWithdrawal"]
        .astype(str)
        .str.replace(r"[,$]", "", regex=True)
        .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)   # (1000) → -1000
        .str.strip()
        .astype(float)
    )
    return col.to_numpy()

# ---------------------------------------------------------------------
# Rolling-window helper
# ---------------------------------------------------------------------
def rolling_windows(mat: np.ndarray, h: int = 30):
    n = mat.shape[0]
    for i in range(0, n - h + 1):
        yield mat[i : i + h]

# ---------------------------------------------------------------------
# Simulation engines
# ---------------------------------------------------------------------
def simulate_planfit(ret: np.ndarray, w: np.ndarray, cap: float, cf: np.ndarray):
    """
    PlanFit dynamic sleeve spend — cash first, then proportional from risk sleeves.
    Returns (success_bool, terminal_wealth, funded_ratio_trajectory).
    """
    stk_bal, bnd_bal, csh_bal = w * cap
    funded_traj = []

    for t, (r_vec, wd) in enumerate(zip(ret, cf)):
        # grow sleeves
        stk_bal *= 1 + r_vec[0]
        bnd_bal *= 1 + r_vec[1]
        csh_bal *= 1 + r_vec[2]

        # spend from cash then risk sleeves
        if wd <= csh_bal:
            csh_bal -= wd
        else:
            short = wd - csh_bal
            csh_bal = 0.0
            risk_bal = stk_bal + bnd_bal
            if risk_bal < short:          # ---- ruin ----
                deficit   = wd - (csh_bal + risk_bal)      # same formula as comparators
                rem_liab = short + cf[t + 1 :].sum()
                # pad the funded-ratio path so total length == H
                funded_traj.extend([0] * (len(cf) - t))   # current + all future years
                return False, -rem_liab, funded_traj
            stk_bal -= short * (stk_bal / risk_bal)
            bnd_bal -= short * (bnd_bal / risk_bal)

        # funded-ratio trajectory
        rem_liab = cf[t + 1 :].sum()
        funded = (stk_bal + bnd_bal + csh_bal) / rem_liab if rem_liab else 0
        funded_traj.append(funded)

    return True, stk_bal + bnd_bal + csh_bal, funded_traj


def simulate_rebal(ret: np.ndarray, w: np.ndarray, cap: float, cf: np.ndarray):
    """
    Static mix: pro-rata withdrawal, then perfect annual rebalance.
    Returns (success_bool, terminal_wealth).
    """
    stk_bal, bnd_bal, csh_bal = w * cap
    for t, (r_vec, wd) in enumerate(zip(ret, cf)):
        # grow
        stk_bal *= 1 + r_vec[0]
        bnd_bal *= 1 + r_vec[1]
        csh_bal *= 1 + r_vec[2]

        tot = stk_bal + bnd_bal + csh_bal
        if tot < wd:  # ruin
            rem_liab = (wd - tot) + cf[t + 1 :].sum()
            return False, -rem_liab
        # pro-rata withdrawal
        stk_bal -= wd * (stk_bal / tot)
        bnd_bal -= wd * (bnd_bal / tot)
        csh_bal -= wd * (csh_bal / tot)

        # rebalance
        tot = stk_bal + bnd_bal + csh_bal
        stk_bal, bnd_bal, csh_bal = w * tot
    return True, stk_bal + bnd_bal + csh_bal

# ---------------------------------------------------------------------
# Capital solver (bisection for ≥ 95 % success)
# ---------------------------------------------------------------------
def solve_capital(mat: np.ndarray, cf: np.ndarray, sim_fn, w: np.ndarray,
                  target=0.95, lo=0.1e6, hi=10e6):
    windows = list(rolling_windows(mat, len(cf)))

    def pass_rate(cap):
        ok = sum(sim_fn(win, w, cap, cf)[0] for win in windows)
        return ok / len(windows)

    while pass_rate(hi) < target:
        hi *= 1.5
        if hi > 50e6:
            raise RuntimeError("Capital solver failed to bracket target")

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        hi, lo = (mid, lo) if pass_rate(mid) >= target else (hi, mid)
    return hi

# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def plot_frontier(df, out="frontier.png"):
    plt.figure(figsize=(6, 4))
    plt.scatter(df["RTC"], df["MWM"], s=60)
    for _, row in df.iterrows():
        plt.annotate(row["Strategy"], (row["RTC"], row["MWM"]),
                     textcoords="offset points", xytext=(5, -5))
    plt.axvline(0, color="grey", lw=0.6)   # break-even tail cushion
    plt.xlabel("Relative Tail Cushion  (CVaR₅ ÷ Start Capital)")
    plt.ylabel("Median Wealth Multiple  (Median Terminal Wealth ÷ Start Capital)")
    plt.title("Capital-Efficiency Frontier – Capital Constraint")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

def plot_heatmap(traj: list[list[float]], start_years: list[int],
                 out="funded_heatmap.png"):
    arr = np.array(traj)
    plt.figure(figsize=(6, 6))
    im = plt.imshow(arr, aspect="auto", origin="lower",
                    cmap="RdYlGn", norm=Normalize(vmin=0, vmax=2))
    plt.colorbar(im, label="Funded Ratio")
    plt.xlabel("Year of Retirement Path")
    plt.ylabel("Start Year of 30-Year Window")
    yt = np.arange(0, len(start_years), 5)
    plt.yticks(yt, [start_years[i] for i in yt])
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
    ap = argparse.ArgumentParser(description="PlanFit vs static mixes")
    ap.add_argument("returns", type=Path)
    ap.add_argument("withdrawals", type=Path)
    ap.add_argument("--capital", type=float, default=None,
                          help="If --fixed is set, this amount is used for ALL strategies."
                             " Otherwise it is PlanFit’s start-cap (omit to auto-size).")
    ap.add_argument("--fixed", action="store_true",
                        help="Run a capital-constraint test: every portfolio starts with"
                             " the same --capital and no solver runs.")
    args = ap.parse_args()

    # returns CSV must include a 'Year' column for heat-map labels
    df_ret    = pd.read_csv(args.returns)
    years_col = df_ret["Year"].astype(int).to_list()
    mat       = df_ret[["Stocks", "Bonds", "Cash"]].to_numpy(float)

    cf = load_withdrawals(args.withdrawals)
    H  = len(cf)

    pf_w = np.array([0.37, 0.0, 0.63])  # PlanFit 30-yr weights

    # ─── portfolio definitions ─────────────────────────────────────────────
    planfit = Strategy("PlanFit", pf_w, 0.0, simulate_planfit)
    bond60  = Strategy("60/40 bonds", np.array([0.6, 0.4, 0.0]), 0.0, simulate_rebal)
    cash60  = Strategy("60/40 cash",  np.array([0.6, 0.0, 0.4]), 0.0, simulate_rebal)
    strategies = [planfit, bond60, cash60]

    # ─── sizing logic (three modes) ────────────────────────────────────────
    if args.fixed:                              # ➊ capital-constraint test
        if args.capital is None:
            ap.error("--fixed requires --capital")
        planfit.start_cap = args.capital
        for strat in strategies[1:]:
            strat.start_cap = args.capital

    else:                                       # ➋ equal-probability modes
        # PlanFit
        if args.capital is None:                # auto-size to 95 %
            planfit.start_cap = solve_capital(mat, cf,
                                              planfit.sim_fn, planfit.w)
            print(f"Solved PlanFit StartCap (95 %): "
                  f"${planfit.start_cap:,.0f}")
        else:                                   # user-supplied PlanFit capital
            planfit.start_cap = args.capital

        # Comparators solved to 95 % pass
        for strat in strategies[1:]:
            strat.start_cap = solve_capital(mat, cf,
                                            strat.sim_fn, strat.w)

    windows = list(rolling_windows(mat, H))
    rows, heat_rows = [], []
    for strat in strategies:
        ok, endings, traj_matrix = 0, [], []
        for win in windows:
            if strat is planfit:
                success, end_w, traj = strat.sim_fn(
                    win, strat.w, strat.start_cap, cf
                )
                traj_matrix.append(traj)
            else:
                success, end_w = strat.sim_fn(
                    win, strat.w, strat.start_cap, cf
                )
            ok += success
            endings.append(end_w)

        ruin        = 1 - ok / len(windows)
        endings     = np.array(endings)
        n_tail         = max(1, int(0.05 * len(endings)))
        cvar5          = np.mean(np.sort(endings)[: n_tail])
        efficiency     = cvar5 / strat.start_cap

        median_tw      = np.median(endings)                  # $ median
        median_mult    = median_tw / strat.start_cap         # multiple

        rows.append({
            "Strategy":    strat.name,
            "StartCap":    round(strat.start_cap, 0),
            "RuinPct":     round(ruin * 100, 2),
            "CVaR5":       round(cvar5, 0),
            "MedianTW":    round(median_tw, 0),
            "MWM":  round(median_mult, 3),
            "RTC":  round(efficiency, 3),
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


if __name__ == "__main__":
    main()
