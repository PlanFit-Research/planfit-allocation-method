#!/usr/bin/env python3
"""
run_backtests.py – PlanFit PoC vs. two RTQ proxies
=================================================
* Dynamic sleeve‑spend engine for PlanFit
* Annual rebalancing for comparison mixes
* 44 rolling 30‑yr windows for deterministic sizing (≥ 95 % pass)
* 5 000 bootstrap paths (six 5‑yr blocks) for robustness
* Outputs: ruin %, CVaR₅, mean surplus, bootstrap inflation

Usage (inside venv)
-------------------
    python Tests/run_backtests.py data/returns_1950_2022.csv \
                                 data/withdrawals.csv       \
                                 --capital 3300000
"""
from __future__ import annotations
import argparse, math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from numpy.random import default_rng

# ---------------------------------------------------------------------
# Globals & constants
# ---------------------------------------------------------------------
SEED, RNG = 42, default_rng(42)
H              = 30     # horizon length (yrs)
BLOCK_YEARS    = 5      # bootstrap block length
TARGET_PASS_RT = 0.95   # ≥ 95 % windows must succeed

# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def load_returns(path: Path) -> np.ndarray:
    """Return N×3 float matrix [stocks, bonds, cash] real returns."""
    df = pd.read_csv(path)[["Stocks", "Bonds", "Cash"]]
    return df.to_numpy(float)


def load_withdrawals(path: Path) -> np.ndarray:
    df = pd.read_csv(path).sort_values("Year")
    col = (df["RealWithdrawal"].astype(str)
             .str.replace(r"[,$]", "", regex=True)
             .str.replace(r"\(([^)]+)\)", r"-\1", regex=True)  # (1000) → -1000
             .str.strip()
             .astype(float))
    vec = col.to_numpy()
    if len(vec) != H:
        sys.exit(f"Withdrawal vector must be {H} years; got {len(vec)}")
    return vec

# ---------------------------------------------------------------------
# PlanFit sleeve‑spend engine
# ---------------------------------------------------------------------

def simulate_planfit(ret: np.ndarray, w: tuple[float,float,float],
                     capital: float, cf: np.ndarray) -> tuple[bool,float]:
    stk_bal, bnd_bal, csh_bal = (capital*np.array(w))
    for r_vec, wd in zip(ret, cf):
        # grow sleeves
        stk_bal *= 1+r_vec[0]; bnd_bal *= 1+r_vec[1]; csh_bal *= 1+r_vec[2]
        # withdraw from cash then equities (stocks→bonds)
        need = wd
        take = min(need, csh_bal); csh_bal -= take; need -= take
        if need > 0:
            take = min(need, stk_bal); stk_bal -= take; need -= take
        if need > 0:
            take = min(need, bnd_bal); bnd_bal -= take; need -= take
        if need > 0:                       # ruin
            return False, -need
    return True, stk_bal+bnd_bal+csh_bal

# ---------------------------------------------------------------------
# Comparison mix – annual full rebalance
# ---------------------------------------------------------------------

def simulate_rebal(ret: np.ndarray, target_w: tuple[float,float,float],
                   capital: float, cf: np.ndarray) -> tuple[bool,float]:
    stk_bal, bnd_bal, csh_bal = (capital*np.array(target_w))
    for r_vec, wd in zip(ret, cf):
        stk_bal *= 1+r_vec[0]; bnd_bal *= 1+r_vec[1]; csh_bal *= 1+r_vec[2]
        tot = stk_bal+bnd_bal+csh_bal
        if wd > tot:
            return False, wd-tot
        ratio = wd/tot
        stk_bal -= stk_bal*ratio; bnd_bal -= bnd_bal*ratio; csh_bal -= csh_bal*ratio
        # rebalance to target
        tot = stk_bal+bnd_bal+csh_bal
        stk_bal, bnd_bal, csh_bal = tot*np.array(target_w)
    return True, stk_bal+bnd_bal+csh_bal

# ---------------------------------------------------------------------
# Window utilities & capital solver
# ---------------------------------------------------------------------

def rolling_windows(mat: np.ndarray, h: int = H):
    for i in range(mat.shape[0]-h+1):
        yield mat[i:i+h]


def solve_capital(mat: np.ndarray, cf: np.ndarray, sim_fn, w: tuple[float,...],
                  target: float = TARGET_PASS_RT,
                  lo: float = 1e5, hi: float = 1e7) -> float:
    windows = list(rolling_windows(mat, len(cf)))

    def pass_rate(cap: float) -> float:
        return sum(sim_fn(win, w, cap, cf)[0] for win in windows) / len(windows)

    # Expand hi until it passes
    while pass_rate(hi) < target:
        hi *= 1.5
        if hi > 1e8:
            raise RuntimeError("Capital solve diverged > $100M")

    # Binary search to ~$1 precision
    lo_cap, hi_cap = lo, hi
    for _ in range(40):
        mid = 0.5*(lo_cap+hi_cap)
        if pass_rate(mid) >= target:
            hi_cap = mid
        else:
            lo_cap = mid
    return hi_cap

# ---------------------------------------------------------------------
# Bootstrap generator – six 5‑year blocks
# ---------------------------------------------------------------------

def bootstrap_paths(mat: np.ndarray, n: int = 5000):
    nrows = mat.shape[0]
    blocks = H//BLOCK_YEARS
    for _ in range(n):
        idx = RNG.integers(0, nrows-BLOCK_YEARS+1, size=blocks)
        yield np.vstack([mat[i:i+BLOCK_YEARS] for i in idx])

# ---------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("returns", type=Path)
    ap.add_argument("withdrawals", type=Path)
    ap.add_argument("--capital", type=float, default=3.3e6)
    args = ap.parse_args()

    mat = load_returns(args.returns)
    cf  = load_withdrawals(args.withdrawals)

    strat_cfg = [
        ("PlanFit",    (0.37,0.00,0.63), args.capital, simulate_planfit),
        ("60/40 bonds",(0.60,0.40,0.00), None,          simulate_rebal),
        ("60/40 cash", (0.60,0.00,0.40), None,          simulate_rebal),
    ]

    # Solve capital for comparison mixes
    for i,(name,w,cap,fn) in enumerate(strat_cfg):
        if cap is None:
            solved = solve_capital(mat, cf, fn, w)
            strat_cfg[i] = (name,w,solved,fn)

    rows = []
    for name,w,cap,fn in strat_cfg:
        endings, success = [], 0
        for win in rolling_windows(mat):
            ok,endw = fn(win,w,cap,cf); success+=ok; endings.append(endw)
        endings = np.array(endings)
        ruin = 1-success/len(endings)
        cvar5 = np.mean(np.sort(endings)[:max(1,int(0.05*len(endings)))])
        rows.append({"Strategy":name,"StartCap":cap,"Ruin%":ruin*100,
                      "CVaR5":cvar5,"MeanEnd":endings.mean()})

    summary = pd.DataFrame(rows)
    summary.to_csv("results_summary.csv", index=False)
    print("\n=== Deterministic 44‑window Test ===")
    print(summary.to_string(index=False, float_format="%.0f"))

    # Bootstrap inflation
    infl = {r["Strategy"]: [] for r in rows}
    for path in bootstrap_paths(mat):
        for name,w,cap,fn in strat_cfg:
            extra = 0.0
            while not fn(path,w,cap+extra,cf)[0]:
                extra += 0.05*cap
                if extra > 2*cap:
                    extra = np.nan; break
            infl[name].append(extra/cap if not math.isnan(extra) else np.nan)

    print("\n=== Bootstrap Inflation (5‑yr blocks) ===")
    for name in infl:
        valid = np.array([v for v in infl[name] if not math.isnan(v)])
        print(f"{name}: {valid.mean()*100:.2f}%")

if __name__ == "__main__":
    main()
