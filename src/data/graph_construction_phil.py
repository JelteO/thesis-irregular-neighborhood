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

def create_graphs_phil():
    df = pd.read_csv(f"{ROOT_DIR}/data/processed/city_payments_processed.csv")

    # categorical data for one-hot features on entry node
    # data is known during transaction, no prior knowledge leakage in model
    ENTRY_CATEGORICAL = [
        "month",
        "day",
        "character_title",
        "sub_obj_title",
        "doc_ref_no_prefix_definition",
    ]

    # index mapping, categorical to node ids. For PROFIT_CENTER & GL_ACCOUNT nodes
    # GNN only works with integer node indices
    # PyTorch Geometric expected format is edge_index = [[source_nodes], [target_nodes]]
    gl_unique = df["vendor_name"].unique()
    pc_unique = df["department_title"].unique()

    gl_account_mapping = {account: idx for idx, account in enumerate(gl_unique)}
    df["gl_idx"] = df["vendor_name"].map(gl_account_mapping)

    pc_mapping = {center: idx for idx, center in enumerate(pc_unique)}
    df["pc_idx"] = df["department_title"].map(pc_mapping)

    numb_gl = len(gl_unique)
    numb_pc = len(pc_unique)

    # numerical features, log normalised amounts (from preprocessing.py)
    NUMERICAL_COLUMNS = ["amount_log"]
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

    label = torch.zeros(len(df), dtype=torch.long)

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
    torch.save(data, f"{ROOT_DIR}/data/processed/graph_hetero_phil.pt")

    return True
