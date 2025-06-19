import pandas as pd

h = pd.read_csv("horizon_table.csv")

assert h.shape == (30, 5), "Expected 30 rows × 5 columns."
assert all(abs(h[["Pct_Stocks", "Pct_Bonds", "Pct_Cash"]].sum(axis=1) - 1) < 1e-8), \
    "Each weight row must sum to 1."
print("All sanity checks passed ✓")
