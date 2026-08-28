# preprocessing.py - raw csv to scaled features for graph + baselines
# in:data/raw/fraud_dataset_v2.csv (533009, 10)
# out: processed csv with feature columns, and two feature frames
# (scaled and unscaled) (533009, 616+2)
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import os
from pathlib import Path
import torch
from src.data.split import make_entry_split

ROOT_DIR = Path(os.getcwd())
OUTPUT_DIR = ROOT_DIR / "outputs"


def preprocessing():
    df = pd.read_csv(f"{ROOT_DIR}/data/raw/fraud_dataset_v2.csv")

    # <class 'pandas.DataFrame'>
    # RangeIndex: 533009 entries, 0 to 533008
    # Data columns (total 10 columns):
    #  #   Column  Non-Null Count   Dtype
    # ---  ------  --------------   -----
    #  0   BELNR   533009 non-null  int64
    #  1   WAERS   533009 non-null  str
    #  2   BUKRS   533009 non-null  str
    #  3   KTOSL   533009 non-null  str
    #  4   PRCTR   533009 non-null  str
    #  5   BSCHL   533009 non-null  str
    #  6   HKONT   533009 non-null  str
    #  7   DMBTR   533009 non-null  float64
    #  8   WRBTR   533009 non-null  float64
    #  9   label   533009 non-null  str
    # dtypes: float64(2), int64(1), str(7)

    df = df.rename(
        columns={
            "BELNR": "entry_id",  # int64
            "WAERS": "currency",  # str
            "BUKRS": "company_code",  # str
            "KTOSL": "account_key",  # str
            "PRCTR": "profit_center",  # str
            "BSCHL": "posting_key",  # str
            "HKONT": "gl_account",  # str
            "DMBTR": "amount_local",  # float64
            "WRBTR": "amount_doc",  # float64
            "label": "label",  # str
        }
    )

    ########################## GRAPH FEATURES ######################
    # only the two scaled amounts, the one-hot happens in graph_construction.py
    df = df.sort_values("entry_id").reset_index(drop=True)
    os.makedirs(f"{ROOT_DIR}/data/processed", exist_ok=True)

    df_graph = df.copy()

    amount_local_log = (df_graph["amount_local"] + 1e-4).apply(np.log)
    amount_doc_log = (df_graph["amount_doc"] + 1e-4).apply(np.log)

    # why: min-max statistics come from the 426327 train rows only, so the scaler
    # never sees held-out amount values. The split is a function of (labels, seed=42),
    # so these rows are byte-identical to the ones the models train on later
    #
    # data: regular amount_local median ~4.9e5, global anomalies all ~9.2e7,
    # so scaled test values can fall slightly outside [0,1] and that is oke
    label_int = torch.tensor(
        df_graph["label"].map({"regular": 0, "local": 1, "global": 2}).values
    )
    train_ids, _, _ = make_entry_split(label_int, seed=42)
    train_rows = train_ids.numpy()

    local_min = amount_local_log.iloc[train_rows].min()
    local_max = amount_local_log.iloc[train_rows].max()
    doc_min = amount_doc_log.iloc[train_rows].min()
    doc_max = amount_doc_log.iloc[train_rows].max()

    df_graph["feature_amount_local"] = (amount_local_log - local_min) / (
        local_max - local_min
    )
    df_graph["feature_amount_doc"] = (amount_doc_log - doc_min) / (doc_max - doc_min)

    df_graph.to_csv(
        f"{ROOT_DIR}/data/processed/fraud_dataset_processed.csv", index=False
    )
    #######################################################################

    ########################## BASELINE FEATURES ##########################
    # for baseline models, the gl_account and profit_center become one-hot
    # columns instead of neighbour nodes. This means, same information as the gnn,
    # flattened categorical data, one-hot encoding
    CATEGORICAL_COLUMNS = [
        "account_key",
        "profit_center",
        "posting_key",
        "gl_account",
        "company_code",
        "currency",
    ]

    NUMERICAL_COLUMNS = [
        "amount_local",
        "amount_doc",
    ]

    df_all = df.copy()

    # numeric scaling on train statistics only, one hot vocabulary stays global
    num_all = df_all[NUMERICAL_COLUMNS].copy()
    num_all = (num_all + 1e-4).apply(np.log)
    num_train_min = num_all.iloc[train_rows].min()
    num_train_max = num_all.iloc[train_rows].max()
    # scale on entire set but with min-max of the train set
    num_all_scaled = (num_all - num_train_min) / (num_train_max - num_train_min)

    # why: the one-hot vocabulary is built on the full dataset.
    # It only defines the feature space (which columns exist), it contains no
    # statistics, so this leaks nothing
    #
    # data: 616 one-hot columns:
    # posting_key 73
    # + account_key 79
    # + company_code 158
    # + currency 76
    # + gl_account 73
    # + profit_center 157
    #
    # gl_account and profit_center are one-hot here because the baselines have no graph
    # The gnn gets them as neighbours instead (388 dims there)
    ohe_all = pd.get_dummies(df_all[CATEGORICAL_COLUMNS], dtype=np.float32)

    print(f"Categorical feature count: {ohe_all.shape[1]}")
    print(f"Numerical feature count: {num_all.shape[1]}")

    features_all = pd.concat([ohe_all, num_all_scaled], axis=1)
    features_all_unscaled = pd.concat([ohe_all, num_all], axis=1)

    features_all["entry_id"] = df_all["entry_id"].values
    features_all["label"] = df_all["label"].values

    features_all_unscaled["entry_id"] = df_all["entry_id"].values
    features_all_unscaled["label"] = df_all["label"].values

    #######################################################################

    return (
        features_all,
        features_all_unscaled,
    )


""" links were used:
https://scikit-learn.org/1.5/modules/decomposition.html#incrementalpca
https://sklearn.org/1.8/auto_examples/decomposition/plot_incremental_pca.html
https://visualstudiomagazine.com/articles/2021/10/20/anomaly-detection-pca.aspx
https://ieeexplore.ieee.org/document/6200273
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
"""
