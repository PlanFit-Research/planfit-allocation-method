#!/usr/bin/env python3
"""
pam_optimize.py
---------------
Generates horizon-specific stock/bond/cash weights (1-30 years) that minimise
the median present-value capital needed for a $1 liability.

Input : data/returns_1950_2022.csv  (created by returns_fetch.py)
Output: horizon_table.csv           (paste into Excel Optimizer tab)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

RET_CSV = "data/returns_1950_2019.csv"
OUT_CSV = "horizon_table.csv"
H_MAX = 30
INIT = np.array([0.34, 0.33, 0.33])
TOL = 1e-9
EPS = 1e-9    # tiny floor to avoid divide-by-zero

# ------------------------------------------------------------------ #
# Helper functions
# ------------------------------------------------------------------ #
def load_returns(path):
    df = pd.read_csv(path)
    if not {"Stocks", "Bonds", "Cash"}.issubset(df.columns):
        raise ValueError("CSV must have columns Stocks, Bonds, Cash")
    return df[["Stocks", "Bonds", "Cash"]].to_numpy()

def pv_factor(weights, block):
    """Present-value factor for one rolling block."""
    growth = np.prod(1 + block @ weights)
    growth = max(growth, EPS)          # prevent <=0
    return 1 / growth                  # smaller is better

def objective(w, blocks, q=85):
    pv_vals = [pv_factor(w, b) for b in blocks]
    tail = np.percentile(pv_vals, q)
    shortfalls = [v for v in pv_vals if v >= tail]
    return sum(shortfalls) / len(shortfalls)   # CVaR

def rolling_blocks(mat, h):
    return [mat[i : i + h] for i in range(len(mat) - h + 1)]

def optimise_for_horizon(mat, h):
    blocks = rolling_blocks(mat, h)
    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: w},
    )
    res = minimize(
        objective,
        INIT,
        args=(blocks,),
        method="SLSQP",
        constraints=cons,
        options={"ftol": TOL, "disp": False},
    )
    if not res.success:
        raise RuntimeError(f"SLSQP failed for horizon {h}: {res.message}")
    w = res.x.clip(min=0)
    w /= w.sum()
    disc = objective(w, blocks)
    return w, disc

# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
mat = load_returns(RET_CSV)
rows = []
for h in range(1, H_MAX + 1):
    w, d = optimise_for_horizon(mat, h)
    rows.append([h, *w, d])
    print(f"h={h:2d}  stocks={w[0]:.4f}  bonds={w[1]:.4f}  cash={w[2]:.4f}  disc={d:.4f}")

cols = ["Time_Horizon_Yrs", "Pct_Stocks", "Pct_Bonds", "Pct_Cash", "Safest_Discount"]
pd.DataFrame(rows, columns=cols).to_csv(
    OUT_CSV,
    index=False,
    float_format="%.6f"      # six-decimal numeric strings
)
print(f"\nSaved {OUT_CSV}")
