"""
returns_fetch.py
----------------
Downloads public-domain data and builds an annual (1950-2022) real-return
CSV for U.S. Stocks, 10-year Treasuries, and 3-month T-Bills.

• Stocks and RF: Kenneth French Research Factors (public domain)
• CPI and 10-yr yields: FRED (U.S. Government work, public domain)

Output: data/returns_1950_2022.csv
"""

import os, io, zipfile, requests
import pandas as pd

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
# 1) Fetch French Monthly Factors (direct CSV)
# ------------------------------------------------------------------ #
ff_csv_url = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors.CSV"
)

# Ken French CSVs use Windows-1252 encoding; pandas can load from URL
ff_csv = pd.read_csv(
    ff_csv_url,
    skiprows=3,               # skip header lines
    encoding="ISO-8859-1"     # avoid UnicodeDecodeError
)

# Trim footer rows that start after the last valid date (marked by 'NaN')
ff_csv = ff_csv[ff_csv.iloc[:, 0].str.match(r"^\d{6}$", na=False)]

ff_csv = (
    ff_csv.rename(columns={"Unnamed: 0": "Date"})
    .astype({"Date": str, "Mkt-RF": float, "RF": float})
)
ff_csv["Date"] = pd.to_datetime(ff_csv["Date"], format="%Y%m")

# Convert to decimal returns
ff_csv[["Mkt-RF", "RF"]] /= 100

# ------------------------------------------------------------------ #
# 2) Fetch 10-year Treasury constant-maturity yields (monthly, robust)
# ------------------------------------------------------------------ #
import io
fred_t10_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GS10"

resp = requests.get(fred_t10_url, timeout=30)
resp.raise_for_status()                            # network errors → exception

t10 = pd.read_csv(io.StringIO(resp.text))

# First column may not be named 'DATE' if FRED adds notes; take the first col
date_col = t10.columns[0]
t10 = t10.rename(columns={date_col: "Date", t10.columns[1]: "GS10"})

t10["Date"] = pd.to_datetime(t10["Date"], errors="coerce")
t10 = t10.dropna(subset=["Date", "GS10"]).replace(".", pd.NA).dropna()
t10["GS10"] = t10["GS10"].astype(float) / 100  # to decimal

# ------------------------------------------------------------------ #
# 3) Fetch CPI for inflation adjustment
# ------------------------------------------------------------------ #
fred_cpi = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
cpi = pd.read_csv(fred_cpi, parse_dates=["DATE"]).rename(
    columns={"DATE": "Date", "CPIAUCSL": "CPI"}
)
cpi["Infl"] = cpi["CPI"].pct_change()

# ------------------------------------------------------------------ #
# 4) Merge, aggregate to annual real returns
# ------------------------------------------------------------------ #
df = (
    ff_csv.merge(t10, on="Date", how="inner")
    .merge(cpi[["Date", "Infl"]], on="Date", how="inner")
)

df["Year"] = df["Date"].dt.year

annual = (
    df.groupby("Year")
    .agg(
        {
            "Mkt-RF": "mean",  # arithmetic mean of monthly excess
            "RF": "mean",
            "GS10": "mean",
            "Infl": lambda x: (1 + x).prod() - 1,
        }
    )
    .loc[1950:2022]
)

# Nominal total returns (simple approximation for bonds)
annual["Stocks_nom"] = annual["Mkt-RF"] + annual["RF"]
annual["Cash_nom"] = annual["RF"]
annual["Bonds_nom"] = annual["GS10"]

# Deflate to real
for col in ["Stocks_nom", "Cash_nom", "Bonds_nom"]:
    annual[col.replace("_nom", "_real")] = (1 + annual[col]) / (1 + annual["Infl"]) - 1

out = (
    annual[["Stocks_real", "Bonds_real", "Cash_real"]]
    .reset_index()
    .rename(columns={"Stocks_real": "Stocks", "Bonds_real": "Bonds", "Cash_real": "Cash"})
)

csv_path = f"{DATA_DIR}/returns_1950_2022.csv"
out.to_csv(csv_path, index=False)
print(f"Saved {csv_path}")
