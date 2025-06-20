# PlanFit Allocation Method (PAM)

Open-source framework for matching household liabilities to capital-efficient
portfolios.

This repository contains all code, data fetchers, and reproducible outputs
that support the paper  
*“PlanFit Allocation Method: A Liability-First Framework for Strategic Asset
Allocation.”*

---

## Methodology snapshot

* **Data set (1950-2022, real)**  
  * Stocks: French “Mkt-RF + RF” (monthly)  
  * Cash: French “RF” (monthly)  
  * Bonds: Damodaran “Return on bond” (annual)  
  * CPI: FRED CPIAUCSL  
  * Returns deflated with CPI to real terms.

* **Risk metric** — minimize **95% CVaR** of the present-value factor across
  every rolling block of length *h* years.

* **Outcome** — bonds receive 0% weight for horizons. Appendix tables
  show alternative setting (a 15%→0% bond glidepath) and the
  capital impact.

---

## Repository structure

| Path / file | Purpose |
|-------------|---------|
| `returns_fetch.py` | Builds `data/returns_1950_2022.csv` (real annual returns). |
| `pam_optimize.py` | Optimizes horizon-specific weights (1–30 yr) at 95 % CVaR; writes `horizon_table.csv`. |
| `horizon_table.csv` | Generated output to paste into the Excel supplement’s **Optimizer** tab. |
| `clip_returns.py` | Utility to create `returns_1950_2019.csv` for sensitivity Table C. |
| `supplement/PlanFit_Allocation_Method_Supplement.xlsx` | Companion workbook (README, Inputs, Calcs, Results). |
| `tests/sanity_check.py` | Smoke test: table is 30×5 and each row sums to 1.0. |
| `requirements.txt` | `numpy`, `pandas`, `scipy`, `requests`. |
| `LICENSE` | MIT + CC-BY notice + KeyBank disclaimer. |
| `data/` | Return files; includes `histretSP_clean.csv` (CC-BY). |

---

## Quick-start

# 1  Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt

# 2  Fetch returns & build optimizer table
python returns_fetch.py
python pam_optimize.py

# 3  (Optional) sanity check
python tests/sanity_check.py

## Why are bond weights zero?

Under a 95 % CVaR of real returns, every worst-tail block (e.g., 1979–1981
or 1992–1994) shows bonds losing more purchasing power than cash.
Any positive bond weight therefore worsens the tail present-value factor, so
the optimizer sets bonds to the lower bound.
Appendix B of the paper demonstrates how a 15%→0% policy floor re-introduces
bonds with only a modest increase in capital.

## Damodaran bond data (CC-BY 4.0)

# 1  Download histretSP.xls from
https://www.stern.nyu.edu/~adamodar/pc/datasets/histretSP.xls

# 2  Delete the prose rows so the first row is the header
Year, … , 10 year T.Bond, …

# 3  Save the sheet as data/histretSP_clean.csv
(already present in this repo under CC-BY 4.0).

No other manual downloads are required.
