import pandas as pd

g = pd.read_excel("data/raw/gpr_daily.xls")
print(g.columns.tolist())
print(g.tail())