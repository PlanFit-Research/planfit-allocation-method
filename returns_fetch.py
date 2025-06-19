"""
returns_fetch.py
----------------
Downloads public-domain data and builds an annual (1950-2022) *real*-return
CSV for U.S. Stocks, 10-year Treasuries, and 3-month T-Bills.

• Stocks & RF  : Kenneth French Research Factors
• Bonds (1973-) : FRED total-return index DGS10TBITTL
  Bonds (1950-72): yield proxy (GS10 average)
• CPI          : FRED CPIAUCSL

Output: data/returns_1950_2022.csv
"""

import os, io, requests
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- helper: geometric annual return ----------
def geom(series):
    """Compound monthly returns into one annual return."""
    return (1 + series).prod() - 1

# ------------------------------------------------------------------ #
# 1) French monthly factors  (Stocks excess + RF)
# ------------------------------------------------------------------ #
ff_url = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors.CSV"
)
ff = pd.read_csv(
    ff_url,
    skiprows=3,
    encoding="ISO-8859-1"
)
ff = ff[ff.iloc[:, 0].str.match(r"^\d{6}$", na=False)]
ff = ff.rename(columns={"Unnamed: 0": "Date"})
ff["Date"] = pd.to_datetime(ff["Date"], format="%Y%m")
ff[["Mkt-RF", "RF"]] = ff[["Mkt-RF", "RF"]].astype(float) / 100

# ------------------------------------------------------------------ #
# 2) 10-year Treasury *yield* series  (proxy + for splice)
# ------------------------------------------------------------------ #
fred_yield = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GS10"
y_resp = requests.get(fred_yield, timeout=30)
y_resp.raise_for_status()

t10 = pd.read_csv(io.StringIO(y_resp.text))
t10 = t10.rename(columns={t10.columns[0]: "Date", t10.columns[1]: "GS10"})
t10["Date"] = pd.to_datetime(t10["Date"], errors="coerce")
t10 = t10.dropna().replace(".", pd.NA).dropna()
t10["GS10"] = t10["GS10"].astype(float) / 100

# ------------------------------------------------------------------ #
# 3) 10-year Treasury total-return index  (1973-present)
# ------------------------------------------------------------------ #
fred_tri = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10TBITTL"
tri_resp = requests.get(fred_tri, timeout=30)
tri_resp.raise_for_status()

tri = pd.read_csv(io.StringIO(tri_resp.text))
tri = tri.rename(columns={tri.columns[0]: "Date", tri.columns[1]: "TRI"})
tri["Date"] = pd.to_datetime(tri["Date"], errors="coerce")
tri = tri.dropna().replace(".", pd.NA).dropna()
tri["TRI"] = tri["TRI"].astype(float) / 100
tri["Year"] = tri["Date"].dt.year
tri_annual = tri.groupby("Year")["TRI"].apply(geom)   # geometric annual return

# ------------------------------------------------------------------ #
# 4) CPI series
# ------------------------------------------------------------------ #
fred_cpi = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
cpi_resp = requests.get(fred_cpi, timeout=30)
cpi_resp.raise_for_status()

cpi = pd.read_csv(io.StringIO(cpi_resp.text))
cpi = cpi.rename(columns={cpi.columns[0]: "Date", cpi.columns[1]: "CPI"})
cpi["Date"] = pd.to_datetime(cpi["Date"], errors="coerce")
cpi = cpi.dropna().replace(".", pd.NA).dropna()
cpi["CPI"] = cpi["CPI"].astype(float)
cpi["Infl"] = cpi["CPI"].pct_change()

# ------------------------------------------------------------------ #
# 5) Merge monthly frames
# ------------------------------------------------------------------ #
df = (
    ff.merge(t10, on="Date", how="inner")
      .merge(cpi[["Date", "Infl"]], on="Date", how="inner")
)

df["Year"] = df["Date"].dt.year

# ------------------------------------------------------------------ #
# 6) Annual aggregation (1950-2022)
# ------------------------------------------------------------------ #
annual = (
    df.groupby("Year")
      .agg({
          "Mkt-RF": geom,
          "RF":      geom,
          "Infl":    geom})
      .loc[1950:2022]
)

# Bond splice: TRI where available, otherwise yield proxy
yield_proxy = df.groupby("Year")["GS10"].mean()
annual["Bonds_nom"] = yield_proxy.combine_first(tri_annual)

# Nominal stock & cash totals
annual["Stocks_nom"] = annual["Mkt-RF"] + annual["RF"]
annual["Cash_nom"]   = annual["RF"]

# Convert to *real* returns
for col in ["Stocks_nom", "Bonds_nom", "Cash_nom"]:
    annual[col.replace("_nom", "_real")] = (1 + annual[col]) / (1 + annual["Infl"]) - 1

# Build output
out = (
    annual[["Stocks_real", "Bonds_real", "Cash_real"]]
      .reset_index()
      .rename(columns={
          "Stocks_real": "Stocks",
          "Bonds_real":  "Bonds",
          "Cash_real":   "Cash"})
)

csv_path = f"{DATA_DIR}/returns_1950_2022.csv"
out.to_csv(csv_path, index=False, float_format="%.6f")
print(f"Saved {csv_path}")
