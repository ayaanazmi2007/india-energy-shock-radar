import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

FEATURES = ["brent_ret_5d", "brent_ret_20d", "brent_vol_10d", "brent_vol_30d",
            "brent_vol_ratio", "inr_ret_20d", "inr_vol_20d", "gpr_level",
            "gpr_z90", "hormuz_ratio", "babel_ratio"]
HORIZONS = {"7d": 5, "30d": 21}

df = pd.read_csv("data/processed/dataset.csv", parse_dates=["date"], index_col="date")
raw = np.log(df["brent"]).diff()

# baseline 2 input: share of stress days over the last 20 trading days
stress_flag = (raw.abs() >= 0.04).astype(float)
df["past_stress_20"] = stress_flag.rolling(20).mean().shift(1)
df = df.dropna(subset=FEATURES + ["past_stress_20"])


def make_models():
    return {
        "logit": make_pipeline(StandardScaler(),
                               LogisticRegression(C=0.5, max_iter=1000)),
        "xgb": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss"),
    }


def walk_forward(target, gap, first_test="2022-01-01", step=63):
    data = df.dropna(subset=[target])
    start = data.index.searchsorted(pd.Timestamp(first_test))
    blocks = []
    for i in range(start, len(data), step):
        test = data.iloc[i:i + step]
        train = data.iloc[:max(i - gap, 0)]          # purge gap
        res = pd.DataFrame({"y": test[target],
                            "p_base": train[target].mean(),      # baseline 1
                            "p_persist": test["past_stress_20"]})  # baseline 2
        for name, m in make_models().items():
            m.fit(train[FEATURES], train[target])
            res[f"p_{name}"] = m.predict_proba(test[FEATURES])[:, 1]
        blocks.append(res)
    return pd.concat(blocks)


rows = []
for name, n in HORIZONS.items():
    res = walk_forward(f"y_{name}", gap=n)
    res.to_csv(f"data/processed/oos_{name}.csv")
    for c in [c for c in res.columns if c.startswith("p_")]:
        rows.append({"horizon": name, "model": c[2:],
                     "brier": brier_score_loss(res["y"], res[c]),
                     "auc": roc_auc_score(res["y"], res[c])})

scores = pd.DataFrame(rows)
base = scores[scores.model == "base"].set_index("horizon")["brier"]
scores["skill_vs_base"] = 1 - scores["brier"] / scores["horizon"].map(base)
scores.round(3).to_csv("data/processed/scores.csv", index=False)
print(scores.round(3))

# Which features matter most? Fit on everything that has a label.
for name in HORIZONS:
    data = df.dropna(subset=[f"y_{name}"])
    m = make_models()["xgb"].fit(data[FEATURES], data[f"y_{name}"])
    imp = pd.Series(m.feature_importances_, index=FEATURES).sort_values(ascending=False)
    imp.to_csv(f"data/processed/importance_{name}.csv")
    print(name, imp.round(3).to_dict())