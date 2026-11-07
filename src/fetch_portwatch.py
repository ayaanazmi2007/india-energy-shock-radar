import requests
import pandas as pd

ROOT = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
DAILY = f"{ROOT}/Daily_Chokepoints_Data/FeatureServer/0/query"
INFO = f"{ROOT}/PortWatch_chokepoints_database/FeatureServer/0/query"

def list_chokepoints():
    params = {"where": "1=1", "outFields": "*", "f": "json"}
    data = requests.get(INFO, params=params, timeout=60).json()
    return pd.DataFrame([f["attributes"] for f in data["features"]])

def fetch_chokepoint(portid):
    rows, offset = [], 0
    while True:
        params = {"where": f"portid='{portid}'", "outFields": "*", "f": "json",
                  "orderByFields": "date", "resultOffset": offset,
                  "resultRecordCount": 1000}
        data = requests.get(DAILY, params=params, timeout=60).json()
        feats = data.get("features", [])
        rows += [f["attributes"] for f in feats]
        if not feats or not data.get("exceededTransferLimit"):
            break
        offset += len(feats)
    df = pd.DataFrame(rows)
    if pd.api.types.is_numeric_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], unit="ms")
    else:
        df["date"] = pd.to_datetime(df["date"])
    return df

if __name__ == "__main__":
    table = list_chokepoints()
    print(table[["portid", "portname"]])
    names = ["Hormuz", "Bab el-Mandeb", "Suez", "Malacca", "Cape of Good Hope"]
    pattern = "|".join(names)
    match = table[table["portname"].str.contains(pattern, case=False)]
    ids = match["portid"].tolist()
    frames = [fetch_chokepoint(i) for i in ids]
    out = pd.concat(frames)
    out.to_csv("data/raw/chokepoints.csv", index=False)
    print(out.groupby("portname")["date"].agg(["min", "max", "count"]))