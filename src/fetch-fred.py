import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("FRED_API_KEY")
URL = "https://api.stlouisfed.org/fred/series/observations"

def get_series(series_id, start="2015-01-01"):
    params = {"series_id": series_id, "api_key": KEY,
              "file_type": "json", "observation_start": start}
    r = requests.get(URL, params=params, timeout=30)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["observations"])[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.rename(columns={"value": series_id}).set_index("date")

if __name__ == "__main__":
    brent = get_series("DCOILBRENTEU")
    inr = get_series("DEXINUS")
    brent.to_csv("data/raw/brent.csv")
    inr.to_csv("data/raw/usdinr.csv")
    print(brent.tail())
    print(inr.tail())