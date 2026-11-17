import numpy as np
import pandas as pd

df = pd.read_csv("data/processed/dataset.csv", parse_dates=["date"], index_col="date")
reg = pd.read_csv("data/processed/regimes.csv", parse_dates=["date"], index_col="date")


def pct_rank(s):
    """Where today sits against everything seen so far, 0 to 100."""
    s = s.dropna()
    return s.expanding(min_periods=250).apply(
        lambda x: (x[-1] >= x).mean(), raw=True) * 100


parts = pd.DataFrame(index=df.index)
parts["market_vol"] = pct_rank(df["brent_vol_10d"])
parts["geopolitics"] = pct_rank(df["gpr_level"])
parts["chokepoints"] = 100 * (1 - df[["hormuz_ratio", "babel_ratio"]].min(axis=1).clip(0, 1))
parts["regime"] = 100 * (reg["p_crisis"] + 0.5 * reg["p_tense"]).reindex(df.index)
parts["rupee"] = pct_rank(df["inr_vol_20d"])

WEIGHTS = {"market_vol": .25, "geopolitics": .20, "chokepoints": .25,
           "regime": .15, "rupee": .15}
parts["score"] = sum(parts[k] * w for k, w in WEIGHTS.items())
parts["band"] = pd.cut(parts["score"], [0, 33, 66, 100],
                       labels=["low", "elevated", "high"])

parts.dropna().to_csv("data/processed/risk_score.csv")
print(parts.dropna().tail())
print("\nAverage score by year:")
print(parts["score"].groupby(parts.index.year).mean().round(1))