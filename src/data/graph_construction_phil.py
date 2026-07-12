import pandas as pd
import numpy as np
import os
import torch
from torch_geometric.data import HeteroData
from pathlib import Path

ROOT_DIR = Path(os.getcwd())
OUTPUT_DIR = ROOT_DIR / "outputs"


def create_graphs_phil():
    df = pd.read_csv(f"{ROOT_DIR}/data/processed/city_payments_processed.csv")

    # entry fields that become one-hot node features
    # vendor_name and department_title are left out on purpose, those become nodes
    ENTRY_CATEGORICAL = [
        "month",
        "day",
        "character_title",
        "sub_obj_title",
        "doc_ref_no_prefix_definition",
    ]

    # node types keep the schreyer names (gl_account, profit_center)
    # so gnn_model.py runs unchanged on both datasets
    gl_unique = df["vendor_name"].unique()  # gl_account = 'vendors'
    pc_unique = df["department_title"].unique()  # profit_center = 'departments'

    gl_account_mapping = {account: idx for idx, account in enumerate(gl_unique)}
    df["gl_idx"] = df["vendor_name"].map(gl_account_mapping)

    pc_mapping = {center: idx for idx, center in enumerate(pc_unique)}
    df["pc_idx"] = df["department_title"].map(pc_mapping)

    numb_gl = len(gl_unique)
    numb_pc = len(pc_unique)

    # numerical features, log normalised amounts (from preprocessing_phil.py)
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

    # nodes own index nr, the gnn turns these into learned embedding
    gl_x = torch.arange(numb_gl, dtype=torch.long).unsqueeze(1)
    pc_x = torch.arange(numb_pc, dtype=torch.long).unsqueeze(1)

    entry_idx = torch.arange(len(df))
    gl_idx = torch.tensor(df["gl_idx"].values)
    pc_idx = torch.tensor(df["pc_idx"].values)

    # assemble the graph with one feature matrix per node type
    data = HeteroData()
    data["entry"].x = entry_features
    data["profit_center"].x = pc_x
    data["gl_account"].x = gl_x

    data.entry_feature_info = entry_feature_info  # metadata
    data.num_gl_accounts = numb_gl
    data.num_profit_centers = numb_pc

    # philadelphia has no labels, all zeros is a placeholder so the shared
    # gnn code runs, nothing selects or evaluates on it
    data["entry"].y = torch.zeros(len(df), dtype=torch.long)
    data["gl_account"].y = torch.zeros(gl_x.shape[0], dtype=torch.long)
    data["profit_center"].y = torch.zeros(pc_x.shape[0], dtype=torch.long)

    # edge 1 (both directions)
    data["entry", "posts_to", "gl_account"].edge_index = torch.stack(
        [entry_idx, gl_idx]
    )
    data["gl_account", "posted_by", "entry"].edge_index = torch.stack(
        [gl_idx, entry_idx]
    )

    # edge 2 (both directions)
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
