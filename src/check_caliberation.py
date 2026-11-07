import pandas as pd

for h in ["7d", "30d"]:
    res = pd.read_csv(f"data/processed/oos_{h}.csv", parse_dates=["date"], index_col="date")
    res["bucket"] = pd.cut(res["p_xgb"], [0, .1, .2, .3, .5, .7, 1.0])
    table = res.groupby("bucket", observed=True).agg(
        predicted=("p_xgb", "mean"), actual=("y", "mean"), days=("y", "size"))
    print("\nHorizon", h)
    print(table.round(2))