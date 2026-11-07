import pandas as pd

res = pd.read_csv("data/processed/oos_7d.csv", parse_dates=["date"], index_col="date")
ds = pd.read_csv("data/processed/dataset.csv", parse_dates=["date"], index_col="date")
res["brent"] = ds["brent"]
res["brent_move_pct"] = ds["brent"].pct_change() * 100

view = res.loc["2026-02-01":"2026-04-30",
               ["brent", "brent_move_pct", "y", "p_persist", "p_logit", "p_xgb"]]
print(view.round(2).to_string())