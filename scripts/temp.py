import pandas as pd

df = pd.read_csv("data/snapshots/snapshot_2018.csv")

dup = (
    df.groupby(["user_id", "item_id"])
      .size()
      .sort_values(ascending=False)
)

print(dup.head(20))