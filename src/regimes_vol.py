import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from arch import arch_model

df = pd.read_csv("data/processed/dataset.csv", parse_dates=["date"], index_col="date")

TRAIN_END = "2023-12-31"   # the model learns regimes only from data up to here
NAMES = ["calm", "tense", "crisis"]

# ---------- Part A: regimes ----------
obs = pd.DataFrame({
    "log_vol": np.log(df["brent_vol_10d"]),
    "gpr_z90": df["gpr_z90"],
    "hormuz_ratio": df["hormuz_ratio"].clip(upper=2),
}).replace([np.inf, -np.inf], np.nan).dropna()

mu, sd = obs.loc[:TRAIN_END].mean(), obs.loc[:TRAIN_END].std()
Z = (obs - mu) / sd                 # put every clue on the same scale
Ztrain = Z.loc[:TRAIN_END].values

# HMM training is sensitive to its starting point: try 10 and keep the best
best = None
for seed in range(10):
    m = GaussianHMM(n_components=3, covariance_type="full", n_iter=300,
                    random_state=seed)
    m.fit(Ztrain)
    s = m.score(Ztrain)
    if best is None or s > best[0]:
        best = (s, m)
model = best[1]

order = np.argsort(model.means_[:, 0])       # lowest volatility first = calm
A = model.transmat_[np.ix_(order, order)]    # daily chance of moving between regimes

# Filtered probabilities: for each day, use ONLY data up to that day
rows = []
for i in range(60, len(Z)):
    rows.append(model.predict_proba(Z.values[: i + 1])[-1][order])
P = pd.DataFrame(rows, index=Z.index[60:], columns=[f"p_{n}" for n in NAMES])
P["regime"] = P.idxmax(axis=1).str[2:]

A5 = np.linalg.matrix_power(A, 5)
P["p_crisis_in_5d"] = P[[f"p_{n}" for n in NAMES]].values @ A5[:, 2]
P.to_csv("data/processed/regimes.csv")

means = model.means_[order] * sd.values + mu.values
print("\nWhat each regime looks like (real units):")
print(pd.DataFrame(means, index=NAMES, columns=obs.columns).round(2))
print("\nChance of moving between regimes each day:")
print(pd.DataFrame(A, index=NAMES, columns=NAMES).round(3))
print("\nAverage days a regime lasts:",
      dict(zip(NAMES, (1 / (1 - np.diag(A))).round(1))))
print(P.tail())

# ---------- Part B: GARCH volatility ----------
brent = pd.read_csv("data/raw/brent.csv", parse_dates=["date"],
                    index_col="date")["DCOILBRENTEU"].dropna()
brent = brent[brent > 0]
r = 100 * np.log(brent).diff().dropna()    # daily returns in percent

fit = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
print(fit.summary())
(fit.conditional_volatility * np.sqrt(252)).to_csv(
    "data/processed/garch_vol.csv", header=["garch_vol_ann_pct"])

# ---------- Part C: figure for the paper ----------
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
df["brent"].loc[P.index].plot(ax=ax[0], color="black")
ax[0].set_ylabel("Brent, USD per barrel")
P[["p_calm", "p_tense", "p_crisis"]].plot.area(ax=ax[1], alpha=0.7)
ax[1].set_ylabel("Regime probability")
plt.tight_layout()
plt.savefig("paper/fig_regimes.png", dpi=200)
print("Saved paper/fig_regimes.png")