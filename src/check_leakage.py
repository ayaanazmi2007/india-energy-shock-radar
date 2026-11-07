import numpy as np
import pandas as pd

date = pd.Timestamp("2022-06-15")

ds = pd.read_csv("data/processed/dataset.csv", parse_dates=["date"], index_col="date")
b = pd.read_csv("data/raw/brent.csv", parse_dates=["date"], index_col="date")["DCOILBRENTEU"].dropna()
b = b.loc["2019-01-01":]

i = b.index.get_loc(date)
expected = np.log(b.iloc[i - 1]) - np.log(b.iloc[i - 6])

print("Value in dataset.csv   :", round(ds.loc[date, "brent_ret_5d"], 6))
print("Computed by hand       :", round(expected, 6))
print("Brent dates used       :", b.index[i - 6].date(), "to", b.index[i - 1].date())
print("Row date               :", date.date())
