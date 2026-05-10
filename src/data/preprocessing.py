import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

root_dir = os.getcwd()

df = pd.read_csv(f"{root_dir}/data/raw/fraud_dataset_v2.csv")

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

df = df.sort_values("entry_id").reset_index(drop=True)

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

ohe = pd.get_dummies(df[CATEGORICAL_COLUMNS], dtype=np.float32)

num = df[NUMERICAL_COLUMNS] + 1e-4
num = num.apply(np.log)
num = (num - num.min()) / (num.max() - num.min())
num = num.astype(np.float32)

features = pd.concat([ohe, num], axis=1)
features.insert(0, "entry_id", df["entry_id"].to_numpy())
features["label"] = df["label"].values


features.to_csv(f"{root_dir}/data/processed/fraud_dataset_processed.csv", index=False)

print("script 'preprocessing.py' finished.")

""" links were used:
https://scikit-learn.org/1.5/modules/decomposition.html#incrementalpca
https://sklearn.org/1.8/auto_examples/decomposition/plot_incremental_pca.html
https://visualstudiomagazine.com/articles/2021/10/20/anomaly-detection-pca.aspx?
https://ieeexplore.ieee.org/document/6200273
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
