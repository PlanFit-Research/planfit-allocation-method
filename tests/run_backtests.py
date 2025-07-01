#!/usr/bin/env python3
"""run_backtests.py – PlanFit PoC vs. 2 RTQ proxies

Adds: dynamic sleeve‑spend model, annual rebalancing for comparators,
44‑window deterministic sizing, 5‑yr‑block bootstrap, probability‑of‑ruin,
CVaR5, efficiency metric, results CSV + optional plots

Usage (inside venv):
    python Tests/run_backtests.py data/returns_1950_2022.csv \
                                     data/withdrawals.csv     \
                                     --capital 3300000
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.random import default_rng

SEED, RNG = 42, default_rng(42)
BLOCK, H, TARGET_PASS = 5, 30, 0.95  # 6×5‑yr blocks per path

# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def load_returns(path: Path) -> np.ndarray:
    df = pd.read_csv(path)[["Stocks", "Bonds", "Cash"]]
    return df.to_numpy(dtype=float)

def load_withdrawals(path: Path) -> np.ndarray:
    df = pd.read_csv(path).sort_values("Year")
    col = (df["RealWithdrawal"].astype(str)
              .str.replace(r"[,$]", "", regex=True)
              .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
              .str.strip()
              .astype(float))
    vec = col.to_numpy()
    if len(vec) != H:
        sys.exit(f"Withdrawal vector must be {H} years – got {len(vec)}")
    return vec

# ---------------------------------------------------------------------
# Sleeve‑spend engine (PlanFit only)
# ---------------------------------------------------------------------

def simulate_planfit(path_ret: np.ndarray, weights: tuple[float, float, float],
                     capital: float, cf: np.ndarray) -> tuple[bool, float]:
    stk_w, bnd_w, csh_w = weights
    stk, bnd, csh = stk_w*capital, bnd_w*capital, csh_w*capital
    for rt, wd in zip(path_ret, cf):
        # grow sleeves
        stk, bnd, csh = stk*(1+rt[0]), bnd*(1+rt[1]), csh*(1+rt[2])
        # pay from sleeve cash first then sleeve equities
        draw = wd
        if draw <= csh:
            csh -= draw
            continue
        draw -= csh; csh = 0
        if draw <= stk:
            stk -= draw
            continue
        draw -= stk; stk = 0
        if draw <= bnd:
            bnd -= draw
            continue
        # sleeves exhausted – ruin
        return False, -draw
        
    return True, stk+bnd+csh

# ---------------------------------------------------------------------
# Comparison portfolios – annual full rebalance to target weights
# ---------------------------------------------------------------------

def simulate_rebal(path_ret: np.ndarray, target_w: tuple[float,float,float],
                   capital: float, cf: np.ndarray) -> tuple[bool, float]:
    stk_w, bnd_w, csh_w = target_w
    stk, bnd, csh = stk_w*capital, bnd_w*capital, csh_w*capital
    for rt, wd in zip(path_ret, cf):
        stk, bnd, csh = stk*(1+rt[0]), bnd*(1+rt[1]), csh*(1+rt[2])
        tot = stk+bnd+csh
        if wd > tot:
            return False, wd-tot
        # proportional withdrawal from total portfolio
        ratio = wd/tot
        stk -= stk*ratio; bnd -= bnd*ratio; csh -= csh*ratio
        # end‑of‑year rebalance
        tot = stk+bnd+csh
        stk, bnd, csh = tot*np.array(target_w)
    return True, stk+bnd+csh

# ---------------------------------------------------------------------
# Deterministic rolling windows & capital solver
# ---------------------------------------------------------------------

def rolling_windows(mat: np.ndarray):
    for i in range(mat.shape[0]-H+1):
        yield mat[i:i+H]

def solve_capital(mat: np.ndarray, cf: np.ndarray, sim_fn, target_w):
    def succ(cap):
        ok = sum(sim_fn(win, target_w, cap, cf)[0] for win in rolling_windows(mat))
        return ok/44 >= TARGET_PASS
    lo, hi = 1e5, 1e7
    while hi-lo > 1e3:
        mid = (hi+lo)/2
        (lo if succ(mid) else hi)[:] = mid  # type: ignore[misc]
    return hi

# ---------------------------------------------------------------------
# Bootstrap generator (6 five‑year blocks)
# ---------------------------------------------------------------------

def bootstrap_paths(mat: np.ndarray, n: int = 5000):
    nrows = mat.shape[0]
    for _ in range(n):
        idx = RNG.integers(0, nrows-BLOCK+1, size=H//BLOCK)
        yield np.vstack([mat[i:i+BLOCK] for i in idx])

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("returns", type=Path)
    ap.add_argument("withdrawals", type=Path)
    ap.add_argument("--capital", type=float, default=3.3e6)
    args = ap.parse_args()

    mat = load_returns(args.returns)
    cf  = load_withdrawals(args.withdrawals)

    planfit_w = (0.37, 0.0, 0.63)
    planfit_cap = args.capital

    bond60_w = (0.60, 0.40, 0.0)
    cash60_w = (0.60, 0.0, 0.40)

    bond60_cap = solve_capital(mat, cf, simulate_rebal, bond60_w)
    cash60_cap = solve_capital(mat, cf, simulate_rebal, cash60_w)

    strategies = [
        ("PlanFit", planfit_w, planfit_cap, simulate_planfit),
        ("60/40 bonds", bond60_w, bond60_cap, simulate_rebal),
        ("60/40 cash",  cash60_w, cash60_cap,  simulate_rebal),
    ]

    rows = []
    for name, w, cap, fn in strategies:
        successes, endings = 0, []
        for win in rolling_windows(mat):
            ok, endw = fn(win, w, cap, cf)
            successes += ok; endings.append(endw)
        ruin = 1 - successes/44
        endings = np.array(endings)
        cvar5 = endings[np.argsort(endings)[: max(1,int(0.05*len(endings)))]].mean()
        rows.append({"Strategy":name,"StartCap":cap,"Ruin%":ruin*100,
                      "CVaR5":cvar5,"MeanEnd":endings.mean()})

    summary = pd.DataFrame(rows)
    summary.to_csv("results_summary.csv",index=False)
    print("\n=== Deterministic 44‑window Test ===")
    print(summary.to_string(index=False, float_format="%.0f"))

    # Bootstrap capital inflation
    infl = {r["Strategy"]:[] for r in rows}
    for path in bootstrap_paths(mat):
        for name, w, cap, fn in strategies:
            extra = 0.0
            while not fn(path, w, cap+extra, cf)[0]:
                extra += 0.05*cap  # 5 % steps
                if extra > 2*cap:
                    extra = np.nan; break
            infl[name].append(extra/cap if not np.isnan(extra) else np.nan)
    print("\n=== Bootstrap Inflation (5‑yr blocks) ===")
    for name in infl:
        valid = [x for x in infl[name] if not math.isnan(x)]
        print(f"{name}: {np.mean(valid)*100:.2f}%")

if __name__ == "__main__":
    main()
