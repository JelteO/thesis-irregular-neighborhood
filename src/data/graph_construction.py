import pandas as pd
import numpy as np
import os
import torch
import torch_geometric
from torch_geometric.data import HeteroData

print(torch.__version__)
print(torch_geometric.__version__)

root_dir = os.getcwd()

df = pd.read_csv(f"{root_dir}/data/processed/fraud_dataset_processed.csv")
# df = df.sort_values("entry_id").reset_index(drop=True)


# index mapping, categorical to node ids. For PROFIT_CENTER & GL_ACCOUNT nodes
# GNN only works with integer node indices
# PyTorch Geometric expected format is edge_index = [[source_nodes], [target_nodes]]

gl_account_unique = df["gl_account"].unique()
gl_account_mapping = {account: idx for idx, account in enumerate(gl_account_unique)}
df["gl_account_idx"] = df["gl_account"].map(gl_account_mapping)
print(df[["entry_id", "gl_account", "gl_account_idx"]].head(5))

profit_center_unique = df["profit_center"].unique()
profit_center_mapping = {center: idx for idx, center in enumerate(profit_center_unique)}
df["profit_center_idx"] = df["profit_center"].map(profit_center_mapping)
print(df[["entry_id", "profit_center", "profit_center_idx"]].head(5))

# feature engineering (node features)
feature_cols = [
    "feature_log_amount_local",
    "feature_log_amount_doc",
    "feature_iszero_amount_doc",
    "feature_ratio_amount_local_doc",
    "feature_israre_postingkey",
    "feature_israre_accountkey",
    "feature_israre_currency",
]

entry_features = torch.tensor(df[feature_cols].values, dtype=torch.float)
print(entry_features.shape)

# create degree feature for profit_center and gl_account
gl_degree = df["gl_account_idx"].value_counts().sort_index()
pc_degree = df["profit_center_idx"].value_counts().sort_index()

gl_feature = torch.tensor(gl_degree.to_numpy().reshape(-1, 1), dtype=torch.float)
pc_feature = torch.tensor(pc_degree.to_numpy().reshape(-1, 1), dtype=torch.float)

# edge construction (foreign keys to edges)
entry_idx = torch.arange(len(df))
gl_idx = torch.tensor(df["gl_account_idx"].values)
pc_idx = torch.tensor(df["profit_center_idx"].values)

label_mapping = {"regular": 0, "local": 1, "global": 2}
label = torch.tensor(df["label"].map(label_mapping).values)

# https://pytorch.org/blog/how-computational-graphs-are-executed-in-pytorch/
# https://medium.com/we-talk-data/pytorch-geometric-tutorial-94af3ae2b8cb
# https://docs.pytorch.org/docs/stable/generated/torch.stack.html
# https://pytorch-geometric.readthedocs.io/en/2.5.3/_modules/torch_geometric/data/data.html?
# HeteroData https://pytorch-geometric.readthedocs.io/en/2.6.0/notes/heterogeneous.html?
data = HeteroData()
data["entry"].x = entry_features
data["entry"].y = label
data["profit_center"].x = pc_feature
data["gl_account"].x = gl_feature

# edge 1
data["entry", "posts_to", "gl_account"].edge_index = torch.stack([entry_idx, gl_idx])
data["gl_account", "posted_by", "entry"].edge_index = torch.stack([gl_idx, entry_idx])

# edge 2
data["entry", "assigned_to", "profit_center"].edge_index = torch.stack([entry_idx, pc_idx])
data["profit_center", "contains", "entry"].edge_index = torch.stack([pc_idx, entry_idx])

# print(data)
data.validate(raise_on_error=True)

torch.save(data, f"{root_dir}/data/processed/graph_hetero.pt")

homo_convert = data.to_homogeneous()



num_entries = len(df)
if homo_convert.num_nodes is None:
    raise ValueError
homo_convert.entry_mask = torch.zeros(homo_convert.num_nodes, dtype=torch.bool)
homo_convert.entry_mask[:num_entries] = True
homo_convert.validate(raise_on_error=True)
torch.save(homo_convert, f"{root_dir}/data/processed/graph_homo.pt")
