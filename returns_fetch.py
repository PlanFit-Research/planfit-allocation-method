"""
returns_fetch.py
----------------
Builds an annual (1950-2022) *real* return CSV for U.S. Stocks,
10-year Treasuries, and 3-month T-Bills.

• Stocks & RF : Kenneth French Research Factors (monthly)
• Bonds       : Damodaran “10-year Treasury Total Return” (annual, CC-BY 4.0)
• CPI         : FRED CPIAUCSL

Output: data/returns_1950_2022.csv
"""

import os, io, requests
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


# ---------- helpers ------------------------------------------------
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
# 3) Merge French factors + CPI  → monthly master frame
# ------------------------------------------------------------------ #
df = ff.merge(cpi[["Date", "Infl"]], on="Date", how="inner")
df["Year"] = df["Date"].dt.year

# ------------------------------------------------------------------ #
# 4) Aggregate to annual (Stocks, Cash, Inflation)
# ------------------------------------------------------------------ #
annual = (
    df.groupby("Year")
      .agg({"Mkt-RF": geom, "RF": geom, "Infl": geom})
      .loc[1950:2022]
      .rename(columns={"Mkt-RF": "Stocks_nom", "RF": "Cash_nom"})
)

# ------------------------------------------------------------------ #
# 5) Damodaran 10-year Treasury Total Return (annual, nominal)
# ------------------------------------------------------------------ #
damo_url = (
    "http://pages.stern.nyu.edu/~adamodar/pc/datasets/histretSP_21.csv"
)
damo = pd.read_csv(damo_url)
damo = damo.rename(
    columns={
        damo.columns[0]: "Year",
        "10 year T.Bond": "Bonds_nom_pct"})
damo = damo[["Year", "Bonds_nom_pct"]].dropna()
damo["Year"] = damo["Year"].astype(int)
damo["Bonds_nom"] = damo["Bonds_nom_pct"].astype(float) / 100  # to decimal

# Merge into annual table
annual = annual.merge(
    damo[["Year", "Bonds_nom"]],
    on="Year", how="left"
)

# ------------------------------------------------------------------ #
# 6) Convert all series to *real* returns
# ------------------------------------------------------------------ #
for col in ["Stocks_nom", "Bonds_nom", "Cash_nom"]:
    annual[col.replace("_nom", "_real")] = (
        (1 + annual[col]) / (1 + annual["Infl"]) - 1
    )

# ------------------------------------------------------------------ #
# 7) Build output CSV
# ------------------------------------------------------------------ #
out = (
    annual[["Stocks_real", "Bonds_real", "Cash_real"]]
      .reset_index()
      .rename(
          columns={
              "Stocks_real": "Stocks",
              "Bonds_real": "Bonds",
              "Cash_real": "Cash"})
)

csv_path = f"{DATA_DIR}/returns_1950_2022.csv"
out.to_csv(csv_path, index=False, float_format="%.6f")
print(f"Saved {csv_path}")
