import pandas as pd


years = [2010, 2015]

dfs = {
    y: pd.read_csv(f"data/snapshots/snapshot_{y}.csv")
    for y in years
}

df_old = dfs[2010]
df_new = dfs[2015]

users_old = set(df_old["user_id"])
users_new = set(df_new["user_id"])

print("Users subset:", users_old.issubset(users_new))

items_old = set(df_old["item_id"])
items_new = set(df_new["item_id"])

print("Items subset:", items_old.issubset(items_new))


edges_old = set(zip(df_old["user_id"], df_old["item_id"]))
edges_new = set(zip(df_new["user_id"], df_new["item_id"]))

print("Edges subset:", edges_old.issubset(edges_new))