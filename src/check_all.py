import pandas as pd

for f in ["brent", "usdinr", "chokepoints"]:
    d = pd.read_csv(f"data/raw/{f}.csv")
    print(f, d.shape, d.iloc[0, 0], d.iloc[-1, 0])

brent = pd.read_csv("data/raw/brent.csv")
print("Missing Brent values:", brent["DCOILBRENTEU"].isna().sum())