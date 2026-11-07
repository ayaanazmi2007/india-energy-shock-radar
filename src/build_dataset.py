import numpy as np
import pandas as pd

START = "2019-01-01"      # PortWatch history starts around here
LAGS = {"brent": 1, "usdinr": 7, "choke": 7, "gpr": 7}  # days until data is really known
STRESS_MOVE = 0.04        # a stress day = Brent moves 4%+ in a day
HORIZONS = {"7d": 5, "30d": 21}   # trading days ahead


def load_series(path, column):
    d = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return d[column].dropna()


def load_gpr():
    g = pd.read_excel("data/raw/gpr_daily.xls")
    if "date" in g.columns:
        g["date"] = pd.to_datetime(g["date"])
    else:
        g["date"] = pd.to_datetime(g["DAY"].astype(str), format="%Y%m%d")
    return g.set_index("date")["GPRD"].dropna()   # change GPRD if yours is named differently


def load_choke():
    c = pd.read_csv("data/raw/chokepoints.csv", parse_dates=["date"])
    return c.pivot_table(index="date", columns="portname", values="n_total")


def col(df, name):   # pick a column by part of its name
    return df[[c for c in df.columns if name.lower() in c.lower()][0]]


def align(s, cal, lag):
    """Put a series on the trading calendar, but only after its real publication lag."""
    s = s[~s.index.duplicated(keep="last")].sort_index().copy()
    s.index = s.index + pd.Timedelta(days=lag)
    return s.reindex(s.index.union(cal)).ffill().reindex(cal)


brent = load_series("data/raw/brent.csv", "DCOILBRENTEU").loc[START:]
cal = brent.index    # Brent trading days are the master calendar

brent_k = align(brent, cal, LAGS["brent"])
inr_k = align(load_series("data/raw/usdinr.csv", "DEXINUS"), cal, LAGS["usdinr"])
gpr_k = align(load_gpr(), cal, LAGS["gpr"])
choke_k = align(load_choke(), cal, LAGS["choke"])

lb = np.log(brent_k)
ret = lb.diff()

f = pd.DataFrame(index=cal)
f["brent_ret_5d"] = lb.diff(5)
f["brent_ret_20d"] = lb.diff(20)
f["brent_vol_10d"] = ret.rolling(10).std()
f["brent_vol_30d"] = ret.rolling(30).std()
f["brent_vol_ratio"] = f["brent_vol_10d"] / f["brent_vol_30d"]
f["inr_ret_20d"] = np.log(inr_k).diff(20)
f["inr_vol_20d"] = np.log(inr_k).diff().rolling(20).std()
f["gpr_level"] = gpr_k
f["gpr_z90"] = (gpr_k - gpr_k.rolling(90).mean()) / gpr_k.rolling(90).std()

h, b = col(choke_k, "Hormuz"), col(choke_k, "Bab el-Mandeb")
f["hormuz_ratio"] = h.rolling(7).mean() / h.rolling(180).median()
f["babel_ratio"] = b.rolling(7).mean() / b.rolling(180).median()

# Labels use the TRUE Brent prices of the days that come after each date
raw_ret = np.log(brent).diff()
stress = (raw_ret.abs() >= STRESS_MOVE).astype(float)
stress[raw_ret.isna()] = np.nan

for name, n in HORIZONS.items():
    ahead = pd.concat([stress.shift(-k) for k in range(1, n + 1)], axis=1)
    f[f"y_{name}"] = ahead.max(axis=1, skipna=False)   # NaN if the window is unfinished

f["brent"] = brent
f.to_csv("data/processed/dataset.csv")

print(f.shape)
print(f[["y_7d", "y_30d"]].mean())   # share of days followed by a shock
print(f.tail(3).T)