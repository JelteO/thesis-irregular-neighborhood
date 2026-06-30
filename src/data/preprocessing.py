import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path

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

    #### SPECIFIC FOR GRAPH ONLY ###########
    df = df.sort_values("entry_id").reset_index(drop=True)
    os.makedirs(f"{ROOT_DIR}/data/processed", exist_ok=True)

    df_graph = df.copy()

    amount_local_log = np.log(df_graph["amount_local"] + 1e-4)
    amount_doc_log = np.log(df_graph["amount_doc"] + 1e-4)

    train_mask = df_graph["label"] == "regular"
    local_min, local_max = (
        amount_local_log[train_mask].min(),
        amount_local_log[train_mask].max(),
    )
    doc_min, doc_max = (
        amount_doc_log[train_mask].min(),
        amount_doc_log[train_mask].max(),
    )

    df_graph["feature_amount_local"] = (amount_local_log - local_min) / (
        local_max - local_min
    )
    df_graph["feature_amount_doc"] = (amount_doc_log - doc_min) / (doc_max - doc_min)

    df_graph.to_csv(
        f"{ROOT_DIR}/data/processed/fraud_dataset_processed.csv", index=False
    )
    ########################################

    # categorical data, one-hot encoding
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

    df_train = df[df["label"] == "regular"].copy()
    df_eval = df.copy()

    num_train = df_train[NUMERICAL_COLUMNS].copy()
    num_eval = df_eval[NUMERICAL_COLUMNS].copy()

    num_train = (num_train + 1e-4).apply(np.log)
    num_eval = (num_eval + 1e-4).apply(np.log)

    num_train_scaled = (num_train - num_train.min()) / (
        num_train.max() - num_train.min()
    )
    num_eval_scaled = (num_eval - num_train.min()) / (num_train.max() - num_train.min())

    ohe_train = pd.get_dummies(df_train[CATEGORICAL_COLUMNS], dtype=np.float32)
    ohe_eval = pd.get_dummies(df_eval[CATEGORICAL_COLUMNS], dtype=np.float32)
    ohe_eval = ohe_eval.reindex(columns=ohe_train.columns, fill_value=0)

    features_train = pd.concat([ohe_train, num_train_scaled], axis=1)
    features_eval = pd.concat([ohe_eval, num_eval_scaled], axis=1)

    features_train_unscaled = pd.concat([ohe_train, num_train], axis=1)
    features_eval_unscaled = pd.concat([ohe_eval, num_eval], axis=1)

    features_train["entry_id"] = df_train["entry_id"].values
    features_train["label"] = df_train["label"].values

    features_eval["entry_id"] = df_eval["entry_id"].values
    features_eval["label"] = df_eval["label"].values

    features_train_unscaled["entry_id"] = df_train["entry_id"].values
    features_train_unscaled["label"] = df_train["label"].values

    features_eval_unscaled["entry_id"] = df_eval["entry_id"].values
    features_eval_unscaled["label"] = df_eval["label"].values

    return (
        features_train,
        features_eval,
        features_train_unscaled,
        features_eval_unscaled,
    )


""" links were used:
https://scikit-learn.org/1.5/modules/decomposition.html#incrementalpca
https://sklearn.org/1.8/auto_examples/decomposition/plot_incremental_pca.html
https://visualstudiomagazine.com/articles/2021/10/20/anomaly-detection-pca.aspx
https://ieeexplore.ieee.org/document/6200273
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
"""
