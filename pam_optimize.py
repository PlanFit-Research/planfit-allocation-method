#!/usr/bin/env python3
"""
pam_optimize.py
---------------
Generates horizon-specific stock/bond/cash weights (1-30 years) that minimize
the **95 % CVaR** (conditional value at risk) of the present-value capital
needed for a $1 real liability.

Input : data/returns_1950_2022.csv   (built by returns_fetch.py – real returns)
Output: horizon_table.csv            (paste into Excel supplement)

Key result: under strict real 95 % CVaR, bonds receive 0 % weight 
because every worst-tail block shows bonds losing more purchasing power
than cash.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ------------------------------------------------------------------ #
# Config constants
# ------------------------------------------------------------------ #
RET_CSV = "data/returns_1950_2022.csv"
OUT_CSV = "horizon_table.csv"
H_MAX   = 30
CVAR_Q  = 95        # tail percentile (e.g., 95 for 95 % CVaR)
INIT    = np.array([0.34, 0.33, 0.33])
TOL     = 1e-9
EPS     = 1e-9      # growth floor to avoid divide-by-zero in PV

# ------------------------------------------------------------------ #
# Helper functions
# ------------------------------------------------------------------ #
def load_returns(path):
    df = pd.read_csv(path)
    if not {"Stocks", "Bonds", "Cash"}.issubset(df.columns):
        raise ValueError("CSV must have columns Stocks, Bonds, Cash")
    return df[["Stocks", "Bonds", "Cash"]].to_numpy()

def pv_factor(w, block):
    """Present-value factor for one rolling block."""
    growth = np.prod(1 + block @ w)      # compound $1 through block
    return 1.0 / max(growth, EPS)        # smaller PV = better

def objective(w, blocks, q=CVAR_Q):
    """95 % (default) CVaR of PV factors across all blocks."""
    pv_vals  = [pv_factor(w, b) for b in blocks]
    cutoff   = np.percentile(pv_vals, q)
    tail     = [v for v in pv_vals if v >= cutoff]     # worst tail
    return np.mean(tail)

def rolling_blocks(mat, h):
    return [mat[i : i + h] for i in range(len(mat) - h + 1)]

def optimize_for_horizon(mat, h):
    blocks = rolling_blocks(mat, h)
    cons   = (
        {"type": "eq",  "fun": lambda w: w.sum() - 1},   # sum to 1
        {"type": "ineq","fun": lambda w: w},            # no negatives
    )
    res = minimize(
        objective, INIT, args=(blocks,),
        method="SLSQP", constraints=cons,
        options={"ftol": TOL, "disp": False},
    )
    if not res.success:
        raise RuntimeError(f"SLSQP failed for horizon {h}: {res.message}")
    w = res.x.clip(min=0)
    w /= w.sum()                         # re-normalize after clipping
    disc = objective(w, blocks)
    return w, disc

# ------------------------------------------------------------------ #
# Main routine
# ------------------------------------------------------------------ #
mat  = load_returns(RET_CSV)
rows = []
for h in range(1, H_MAX + 1):
    w, d = optimize_for_horizon(mat, h)
    rows.append([h, *w, d])
    print(f"h={h:2d}  stocks={w[0]:.4f}  bonds={w[1]:.4f}  "
          f"cash={w[2]:.4f}  disc={d:.4f}")

cols = ["Time_Horizon_Yrs", "Pct_Stocks", "Pct_Bonds",
        "Pct_Cash", "Safest_Discount"]
pd.DataFrame(rows, columns=cols).to_csv(
    OUT_CSV, index=False, float_format="%.6f"
)
print(f"\nSaved {OUT_CSV}")
