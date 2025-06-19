# PlanFit Allocation Method (PAM)

Open-source framework for matching household liabilities to capital-efficient
portfolios.  
This repository contains **all code, data fetchers, and reproducible outputs**
that support the paper  
*“PlanFit Allocation Method: A Liability-First Framework for Strategic Asset
Allocation.”*

---

## Repository structure

| Path / file | Purpose |
|-------------|---------|
| `returns_fetch.py` | Auto-downloads public-domain annual real-return series (1950-2022) for U.S. Stocks, 10-yr Treasuries, and T-Bills. Saves to `data/`. |
| `pam_optimize.py` | Optimises horizon-specific weights (1–30 years) that minimise the present-value capital required for a \$1 liability. Writes `horizon_table.csv`. |
| `horizon_table.csv` | **Generated output** – paste these five columns into the Excel supplement’s **Optimizer** tab. |
| `supplement/PlanFit_Allocation_Method_Supplement.xlsx` | Companion workbook cited in the paper (README tab, inputs, calc engine, results). |
| `tests/sanity_check.py` | Quick assertions that the optimiser table is 30 × 5 and each row sums to 1.0. |
| `requirements.txt` | Minimal Python deps: `numpy`, `pandas`, `scipy`, `requests`. |
| `LICENSE` | MIT for code. |
| `data/` | Download cache for raw returns (ignored by Git history). |

---

## Quick-start

```bash
# 1  Set up environment
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2  Fetch returns & build optimizer table
python returns_fetch.py
python pam_optimize.py

> **Note on real returns**  
> All series are expressed in *real* terms after CPI adjustment.  
> Cash therefore shows negative values whenever T-bill yield < inflation.

> **Bond approximation**  
> Bond returns are approximated via a constant-duration (≈8) model applied
> to the FRED GS10 yield series. This replicates SBBI 10-year Treasury total
> returns within ~50 bp and retains public-domain licensing.

# 3  (Optional) sanity check
python tests/sanity_check.py
