import pandas as pd
from pathlib import Path

SRC   = Path("data/returns_1950_2022.csv")   # input produced by returns_fetch
DEST  = Path("data/returns_1950_2019.csv")   # output for Table C
ENDYR = 2019                                 # last year to keep

df = pd.read_csv(SRC)
if "Year" not in df.columns:
    raise ValueError("Column 'Year' not found in " + SRC.as_posix())

df_clip = df[df["Year"] <= ENDYR]
df_clip.to_csv(DEST, index=False, float_format="%.6f")
print(f"✓ wrote {DEST}  ({len(df_clip)} rows)")
