import numpy as np
import pandas as pd
from arch import arch_model

IMPORT_MBPD = 5.45419   # PLACEHOLDER: replace with the latest PPAC figure (million barrels/day)
DAYS = 21           # trading days in the forecast window
N_SIMS = 5000
rng = np.random.default_rng(42)

brent = pd.read_csv("data/raw/brent.csv", parse_dates=["date"],
                    index_col="date")["DCOILBRENTEU"].dropna()
brent = brent[brent > 0]
inr = pd.read_csv("data/raw/usdinr.csv", parse_dates=["date"],
                  index_col="date")["DEXINUS"].dropna()

r = 100 * np.log(brent).diff().dropna()
fit = arch_model(r, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")

fc = fit.forecast(horizon=DAYS, method="simulation", simulations=N_SIMS, reindex=False)
sims = fc.simulations.values[-1]    # shape (N_SIMS, DAYS), in percent
paths = brent.iloc[-1] * np.exp(np.cumsum(sims / 100, axis=1))   # simulated Brent paths

# Rupee: a simple independent shock sized by the last 60 days of rupee moves
inr_vol = np.log(inr).diff().tail(60).std()
inr_end = inr.iloc[-1] * np.exp(rng.normal(0, inr_vol * np.sqrt(DAYS), N_SIMS))

# Monthly bill = barrels per day x 30 days x average price
bill_usd_bn = IMPORT_MBPD * 1e6 * 30 * paths.mean(axis=1) / 1e9
bill_inr_lakh_cr = bill_usd_bn * inr_end / 1000    # 1 lakh crore rupees = 1,000 bn rupees
current_usd_bn = IMPORT_MBPD * 1e6 * 30 * brent.iloc[-1] / 1e9

q = [5, 25, 50, 75, 95]
summary = pd.DataFrame({"usd_bn": np.percentile(bill_usd_bn, q),
                        "inr_lakh_crore": np.percentile(bill_inr_lakh_cr, q)},
                       index=[f"p{x}" for x in q])

print("Latest Brent used:", round(brent.iloc[-1], 2), "on", brent.index[-1].date())
print("Bill if Brent stays flat (USD bn):", round(current_usd_bn, 1))
print(summary.round(2))
summary.to_csv("data/processed/bill_summary.csv")

fan = pd.DataFrame(np.percentile(paths, q, axis=0).T, columns=[f"p{x}" for x in q])
fan.index.name = "day_ahead"
fan.index += 1
fan.to_csv("data/processed/forecast_fan.csv")
print("Saved bill_summary.csv and forecast_fan.csv")