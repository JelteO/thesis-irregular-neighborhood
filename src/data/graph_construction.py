import pandas as pd
import numpy as np
import os
import torch
import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from pathlib import Path

ROOT_DIR = Path(os.getcwd())
OUTPUT_DIR = ROOT_DIR / "outputs"

def create_graphs():
    df = pd.read_csv(f"{ROOT_DIR}/data/processed/fraud_dataset_processed.csv")

    # categorical data for one-hot features on entry node
    # data is known during transaction, no prior knowledge leakage in model
    ENTRY_CATEGORICAL = ["posting_key", "account_key", "company_code", "currency"]

    # index mapping, categorical to node ids. For PROFIT_CENTER & GL_ACCOUNT nodes
    # GNN only works with integer node indices
    # PyTorch Geometric expected format is edge_index = [[source_nodes], [target_nodes]]
    gl_unique = df["gl_account"].unique()
    pc_unique = df["profit_center"].unique()

    gl_account_mapping = {account: idx for idx, account in enumerate(gl_unique)}
    df["gl_idx"] = df["gl_account"].map(gl_account_mapping)

    pc_mapping = {center: idx for idx, center in enumerate(pc_unique)}
    df["pc_idx"] = df["profit_center"].map(pc_mapping)

    numb_gl = len(gl_unique)
    numb_pc = len(pc_unique)

    # numerical features, log normalised amounts (from preprocessing.py)
    NUMERICAL_COLUMNS = ["feature_amount_local", "feature_amount_doc"]
    num_features = torch.tensor(df[NUMERICAL_COLUMNS].values, dtype=torch.float)

    # categorical features as one-hot encoding
    ohe_df = pd.get_dummies(
        df[ENTRY_CATEGORICAL], columns=ENTRY_CATEGORICAL, dtype=np.float32
    )
    cat_features = torch.tensor(ohe_df.values, dtype=torch.float)

    # concat to a single entry feature vector
    entry_features = torch.cat([num_features, cat_features], dim=1)

    # metadata: keep track of which column present what feature
    # gnn_model.py requires this metadata for slicinig for cross-entropy loss at reconstruction
    cat_dims = {col: df[col].nunique() for col in ENTRY_CATEGORICAL}
    entry_feature_info = {
        "num_numerical": len(NUMERICAL_COLUMNS),
        "categorical_dims": cat_dims,
        "categorical_order": ENTRY_CATEGORICAL,
    }

    gl_x = torch.arange(numb_gl, dtype=torch.long).unsqueeze(1)  # shape [73, 1]
    pc_x = torch.arange(numb_pc, dtype=torch.long).unsqueeze(1)  # shape [157, 1]

    entry_idx = torch.arange(len(df))
    gl_idx = torch.tensor(df["gl_idx"].values)
    pc_idx = torch.tensor(df["pc_idx"].values)

    label_mapping = {"regular": 0, "local": 1, "global": 2}
    label = torch.tensor(df["label"].map(label_mapping).values)

    # https://pytorch.org/blog/how-computational-graphs-are-executed-in-pytorch/
    # https://medium.com/we-talk-data/pytorch-geometric-tutorial-94af3ae2b8cb
    # https://docs.pytorch.org/docs/stable/generated/torch.stack.html
    # https://pytorch-geometric.readthedocs.io/en/2.5.3/_modules/torch_geometric/data/data.html
    # HeteroData https://pytorch-geometric.readthedocs.io/en/2.6.0/notes/heterogeneous.html

    data = HeteroData()
    data["entry"].x = entry_features
    data["profit_center"].x = pc_x
    data["gl_account"].x = gl_x

    data.entry_feature_info = entry_feature_info
    data.num_gl_accounts = numb_gl
    data.num_profit_centers = numb_pc

    data["entry"].y = label
    data["gl_account"].y = torch.zeros(gl_x.shape[0], dtype=torch.long)  # regular
    data["profit_center"].y = torch.zeros(pc_x.shape[0], dtype=torch.long)  # regular

    # edge 1
    data["entry", "posts_to", "gl_account"].edge_index = torch.stack(
        [entry_idx, gl_idx]
    )
    data["gl_account", "posted_by", "entry"].edge_index = torch.stack(
        [gl_idx, entry_idx]
    )

    # edge 2
    data["entry", "assigned_to", "profit_center"].edge_index = torch.stack(
        [entry_idx, pc_idx]
    )
    data["profit_center", "contains", "entry"].edge_index = torch.stack(
        [pc_idx, entry_idx]
    )

    data.validate(raise_on_error=True)

    # sanity check before save
    expected_dim = len(NUMERICAL_COLUMNS) + sum(cat_dims.values())
    assert (
        entry_features.shape[1] == expected_dim
    ), f"feature dim mismatch: {entry_features.shape[1]} vs expected {expected_dim}"
    print(f"check passed\n entry feat:{expected_dim}dims")
    torch.save(data, f"{ROOT_DIR}/data/processed/graph_hetero.pt")

    data_homo = data.to_homogeneous()
    num_nodes = data_homo.num_nodes
    assert num_nodes is not None and data_homo.edge_index is not None
    
    data_homo.entry_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data_homo.entry_mask[: (len(df))] = True
    data_homo.validate(raise_on_error=True)
    torch.save(data_homo, f"{ROOT_DIR}/data/processed/graph_homo.pt")
    print(f"homo: {data_homo.num_nodes} nodes, {data_homo.edge_index.shape[1]} edges")

    # subset homo graph
    anomaly_idx = (label != 0).nonzero(as_tuple=True)[0]
    regular_idx = (label == 0).nonzero(as_tuple=True)[0]
    perm = torch.randperm(len(regular_idx), generator=torch.Generator().manual_seed(42))
    keep_entries = torch.cat([anomaly_idx, regular_idx[perm[:10_000]]])

    n_e = len(keep_entries)  # 10.100
    gl_offset = n_e  # 10.100
    pc_offset = n_e + numb_gl  # 10.173

    new_idx = torch.arange(n_e)
    gl_src = gl_idx[keep_entries]
    pc_src = pc_idx[keep_entries]

    edge_index = torch.stack(
        [
            torch.cat([new_idx, gl_src + gl_offset, new_idx, pc_src + pc_offset]),
            torch.cat([gl_src + gl_offset, new_idx, pc_src + pc_offset, new_idx]),
        ]
    )
    
    feat_dim = entry_features.shape[1]# 388
    gl_x_pad = torch.zeros(numb_gl, feat_dim)
    pc_x_pad = torch.zeros(numb_pc, feat_dim)
    x = torch.cat([entry_features[keep_entries], gl_x_pad, pc_x_pad], dim=0)

    y = torch.cat(
        [label[keep_entries], torch.zeros(numb_gl + numb_pc, dtype=torch.long)]
    )

    homo_sample = Data(x=x, edge_index=edge_index, y=y)
    homo_sample.n_entries = n_e
    homo_sample.validate(raise_on_error=True)
    torch.save(homo_sample, f"{ROOT_DIR}/data/processed/graph_homo_sample.pt")
    print(f"homo_sample: {homo_sample.num_nodes} nodes, {edge_index.shape[1]} edges")

    return True

