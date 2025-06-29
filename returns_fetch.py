"""
returns_fetch.py
----------------
Builds an annual (1950-2022) *real* return CSV for U.S. stocks,
10-year Treasuries, and 3-month T-Bills.

Data sources
• Stocks & RF : Kenneth French Research Factors (monthly, public domain)
• Bonds      : Damodaran “Return on bond” (annual, © CC-BY 4.0, NYU Stern)
• CPI        : FRED CPIAUCSL (public domain)

Output: data/returns_1950_2022.csv

This file is the sole input to pam_optimize.py, which minimizes the 95 %
CVaR present-value factor across rolling blocks.
"""

import os, io, pathlib, sys, requests
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- helper -------------------------------------------------
def geom(series):
    """Compound monthly returns into one annual return."""
    return (1 + series).prod() - 1

# ------------------------------------------------------------------ #
# 1) French monthly factors (stocks excess & RF)
# ------------------------------------------------------------------ #
ff_url = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors.CSV"
)
ff = pd.read_csv(ff_url, skiprows=3, encoding="ISO-8859-1")
ff = ff[ff.iloc[:, 0].str.match(r"^\d{6}$", na=False)]
ff = ff.rename(columns={"Unnamed: 0": "Date"})
ff["Date"] = pd.to_datetime(ff["Date"], format="%Y%m")
ff[["Mkt-RF", "RF"]] = ff[["Mkt-RF", "RF"]].astype(float) / 100

# ------------------------------------------------------------------ #
# 2) CPI series (monthly, real deflator)
# ------------------------------------------------------------------ #
cpi_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
cpi = pd.read_csv(io.StringIO(requests.get(cpi_url, timeout=30).text))
cpi = cpi.rename(columns={cpi.columns[0]: "Date", cpi.columns[1]: "CPI"})
cpi["Date"] = pd.to_datetime(cpi["Date"], errors="coerce")
cpi = cpi.dropna().replace(".", pd.NA).dropna()
cpi["CPI"] = cpi["CPI"].astype(float)
cpi["Infl"] = cpi["CPI"].pct_change()

# ------------------------------------------------------------------ #
# 3) Merge → monthly master frame
# ------------------------------------------------------------------ #
df = ff.merge(cpi[["Date", "Infl"]], on="Date", how="inner")
df["Year"] = df["Date"].dt.year

# ------------------------------------------------------------------ #
# 4) Aggregate to annual (stocks-excess, cash, inflation)
# ------------------------------------------------------------------ #
annual = (
    df.groupby("Year")
      .agg({"Mkt-RF": geom, "RF": geom, "Infl": geom})
      .loc[1950:2022]
      .rename(columns={"Mkt-RF": "Stocks_nom", "RF": "Cash_nom"})
)

# Add risk-free to excess to get TOTAL stock return
annual["Stocks_nom"] += annual["Cash_nom"]

# ------------------------------------------------------------------ #
# 5) Damodaran 10-year Treasury total return (clean CSV in data/)
# ------------------------------------------------------------------ #
csv_path = pathlib.Path(DATA_DIR) / "histretSP_clean.csv"
if not csv_path.exists():
    sys.exit(
        "Damodaran CSV not found. See README for instructions to place "
        "histretSP_clean.csv in the data/ folder."
    )

damo = pd.read_csv(csv_path)

# Standardize headers
damo.columns = (
    damo.columns.str.strip()
                 .str.replace(r"\s+", " ", regex=True)
                 .str.lower()
)

# Pick numeric column whose name contains 'bond' OR ends with 'return'
numeric_cols = (
    damo.drop(columns="year")
        .apply(lambda s: pd.to_numeric(s.str.rstrip("%"), errors="coerce"))
)
bond_candidates = [
    c for c in numeric_cols.columns
    if ("bond" in c) or c.endswith("return")
]
if not bond_candidates:
    raise ValueError(f"No bond return column found: {list(damo.columns)}")
bond_col = bond_candidates[0]

damo = (
    damo[["year", bond_col]]
      .rename(columns={"year": "Year", bond_col: "Bonds_nom_pct"})
      .dropna()
)

# Restrict to 1950-2022 and convert % → decimal
damo = damo[damo["Year"].between(1950, 2022)]
damo["Bonds_nom"] = (
    damo["Bonds_nom_pct"].astype(str).str.rstrip("%").astype(float) / 100
)

# Merge and fill any missing early rows (1950-52) by interpolation
annual = annual.merge(damo[["Year", "Bonds_nom"]], on="Year", how="left")
annual["Bonds_nom"] = annual["Bonds_nom"].interpolate(limit_direction="both")

# ------------------------------------------------------------------ #
# 6) Convert nominal series to *real* returns
# ------------------------------------------------------------------ #
for col in ["Stocks_nom", "Bonds_nom", "Cash_nom"]:
    annual[col.replace("_nom", "_real")] = (
        (1 + annual[col]) / (1 + annual["Infl"]) - 1
    )

# ------------------------------------------------------------------ #
# 7) Write output CSV
# ------------------------------------------------------------------ #
out = (
    annual[["Stocks_real", "Bonds_real", "Cash_real"]]
      .reset_index()
      .rename(columns={
          "index": "Year",                
          "Stocks_real": "Stocks",
          "Bonds_real":  "Bonds",
          "Cash_real":   "Cash"})
)

out_path = pathlib.Path(DATA_DIR) / "returns_1950_2022.csv"
out.to_csv(out_path, index=False, float_format="%.6f")
print(f"Saved {out_path}")
