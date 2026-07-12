import pandas as pd
import numpy as np
import os
import torch
from torch_geometric.data import HeteroData
from torch_geometric.data import Data
from pathlib import Path
from src.data.split import make_entry_split, sample_common_test

ROOT_DIR = Path(os.getcwd())
OUTPUT_DIR = ROOT_DIR / "outputs"


def create_graphs():
    df = pd.read_csv(f"{ROOT_DIR}/data/processed/fraud_dataset_processed.csv")

    # four entry fields that become one-hot node features (386 columns)
    # gl_account and profit_center are left out since those columsn become nodes
    ENTRY_CATEGORICAL = ["posting_key", "account_key", "company_code", "currency"]

    # MAPPING
    # map every gl_account & profit_center value to integer node id
    # Edge format is edge_index = [[source_nodes], [target_nodes]]
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
    gl_x = torch.arange(numb_gl, dtype=torch.long).unsqueeze(1)  # [73, 1] tensor int64
    pc_x = torch.arange(numb_pc, dtype=torch.long).unsqueeze(1)  # [157, 1]

    entry_idx = torch.arange(len(df))  # tensor int64 (533009)[0,1,2,...]
    gl_idx = torch.tensor(df["gl_idx"].values)
    pc_idx = torch.tensor(df["pc_idx"].values)

    label_mapping = {"regular": 0, "local": 1, "global": 2}
    label = torch.tensor(df["label"].map(label_mapping).values)  # tensor int64

    # ASSEMBLE the graph with one feature matrix per node type
    data = HeteroData()
    data["entry"].x = entry_features  # float32 (533009, 388)
    data["profit_center"].x = pc_x
    data["gl_account"].x = gl_x  # int64 (73,1)

    data.entry_feature_info = entry_feature_info  # metadata
    data.num_gl_accounts = numb_gl
    data.num_profit_centers = numb_pc

    # entry nodes get their label, backbone nodes are always 0
    data["entry"].y = label  # int64 (533009,)
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
    torch.save(data, f"{ROOT_DIR}/data/processed/graph_hetero.pt")

    # why: DOMINANT is homogeneous, so entries and backbone share one node space here
    # Unlike the hetero model, it has no per-node embedding table, so fit and eval graph
    # do not need matching node indices
    #
    # data:
    # The 'fit graph' has 10230 nodes: 10000 sampled train regulars + 73 gl + 157 pc
    # The 'eval graph' adds the 10050 common test entries on top: 20280 nodes
    # homo graphs for DOMINANT, same three phase protocol as the gnn fit graph:
    # only train entry edges
    def build_homo(keep_entries):
        numb_entries = len(keep_entries)
        gl_offset = numb_entries
        pc_offset = numb_entries + numb_gl

        new_idx = torch.arange(numb_entries)
        gl_src = gl_idx[keep_entries]
        pc_src = pc_idx[keep_entries]

        edge_index = torch.stack(
            [
                torch.cat([new_idx, gl_src + gl_offset, new_idx, pc_src + pc_offset]),
                torch.cat([gl_src + gl_offset, new_idx, pc_src + pc_offset, new_idx]),
            ]
        )

        # self loops on every node
        # Train regulars only include a few gl/pc categories,
        # so without these self loops, the highest backbone nodes have no edges
        n_nodes = numb_entries + numb_gl + numb_pc
        self_loops = torch.arange(n_nodes)
        edge_index = torch.cat(
            [edge_index, torch.stack([self_loops, self_loops])], dim=1
        )

        feat_dim = entry_features.shape[1]  # 388

        # PADDING
        # torch.full builds a matrix of the given shape with every cell set to
        # the same value: (73, 388) and (157, 388), filled with 0.1.
        #
        # These rows are stacked under the entry rows below, so gl/pc nodes get a flat
        # dummy feature vector instead of real features.
        #
        # The value is 0.1 and not 0.0 because an isolated all-zero node reconstructs
        # to exactly zero, and sqrt(0) in DOMINANT's loss gives NaN gradients
        gl_x_padding = torch.full((numb_gl, feat_dim), 0.1)
        pc_x_padding = torch.full((numb_pc, feat_dim), 0.1)
        x = torch.cat([entry_features[keep_entries], gl_x_padding, pc_x_padding], dim=0)

        y = torch.cat(
            [label[keep_entries], torch.zeros(numb_gl + numb_pc, dtype=torch.long)]
        )

        homo = Data(x=x, edge_index=edge_index, y=y)
        homo.n_entries = numb_entries
        homo.entry_ids = keep_entries  # global row ids, for explanations
        homo.validate(raise_on_error=True)
        return homo

    train_ids, val_ids, test_ids = make_entry_split(label, seed=42)
    common_ids = sample_common_test(
        label, test_ids, n_regular=10_000, seed=42
    )  # 10000 + 50 anomalies

    # full batch DOMINANT cannot handle all 426k train entries on cpu
    # so fit on a 10k sample of train regulars
    perm = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(42))
    fit_entries = train_ids[perm[:10_000]]

    homo_fit = build_homo(fit_entries)
    torch.save(homo_fit, f"{ROOT_DIR}/data/processed/graph_homo_fit.pt")
    print(f"homo_fit: {homo_fit.num_nodes} nodes")

    eval_entries = torch.cat([fit_entries, common_ids])  # 10000 + 10050
    homo_eval = build_homo(eval_entries)
    homo_eval.n_fit = len(fit_entries)  # common test entries (:common_ids) start here
    homo_eval.entry_feature_info = entry_feature_info
    torch.save(homo_eval, f"{ROOT_DIR}/data/processed/graph_homo_eval.pt")
    print(f"homo_eval: {homo_eval.num_nodes} nodes")

    return True


# https://pytorch.org/blog/how-computational-graphs-are-executed-in-pytorch/
# https://medium.com/we-talk-data/pytorch-geometric-tutorial-94af3ae2b8cb
# https://docs.pytorch.org/docs/stable/generated/torch.stack.html
# https://pytorch-geometric.readthedocs.io/en/2.5.3/_modules/torch_geometric/data/data.html
# HeteroData https://pytorch-geometric.readthedocs.io/en/2.6.0/notes/heterogeneous.html
