import pandas as pd
import numpy as np
import os
from pathlib import Path

ROOT_DIR = Path(os.getcwd())
OUTPUT_DIR = ROOT_DIR / "outputs"

def preprocessing_phil():

    df = pd.read_csv(f"{ROOT_DIR}/data/raw/city_payments_fy2017.csv")
    df["check_date"] = pd.to_datetime(df["check_date"])
    df = df.sort_values("check_date").reset_index(drop=True)

    """RangeIndex: 238894 entries, 0 to 238893
    Data columns (total 16 columns):
    #   Column                        Non-Null Count   Dtype
    ---  ------                        --------------   -----
    0   fy                            238894 non-null  int64
    1   fm                            238894 non-null  int64
    2   check_date                    238894 non-null  str
    3   document_no                   238894 non-null  str
    4   dept                          238894 non-null  int64  [prefix for node 2]
    5   department_title              238894 non-null  str    [node 2]
    6   char_                         238894 non-null  int64
    7   character_title               238894 non-null  str
    8   sub_obj                       238894 non-null  str
    9   sub_obj_title                 238894 non-null  str
    10  vendor_name                   238894 non-null  str    [node 1]
    11  doc_ref_no_prefix             238843 non-null  str    [drop?, 0.02% missing values]
    12  doc_ref_no_prefix_definition  238843 non-null  str    [drop?, 0.02% missing values]
    13  contract_number               115646 non-null  str    [drop, 52% missing values]
    14  contract_description          115646 non-null  str    [drop, 52% missing values]
    15  transaction_amount            238894 non-null  float64
    dtypes: float64(1), int64(4), str(11)
    memory usage: 29.2 MB
    print(df.info())
    """

    df["month"] = df["check_date"].dt.month
    df["day"] = df["check_date"].dt.day

    amount = df["transaction_amount"]
    amount_log = np.sign(amount) * np.log1p(np.abs(amount))
    df["amount_log"] = (amount_log - amount_log.min()) / (
        amount_log.max() - amount_log.min()
    )

    # 51 missing values (0.02%) fillna
    df["doc_ref_no_prefix_definition"] = df["doc_ref_no_prefix_definition"].fillna(
        "unknown"
    )

    df = df.drop(
        columns=[
            "fy",
            "fm",
            "check_date",
            "document_no",
            "dept",
            "char_",
            "sub_obj",
            "doc_ref_no_prefix",
            "contract_number",
            "contract_description",
            "transaction_amount",
        ]
    )

    os.makedirs(f"{ROOT_DIR}/data/processed", exist_ok=True)
    df.to_csv(f"{ROOT_DIR}/data/processed/city_payments_processed.csv", index=False)
    return df


if __name__ == "__main__":
    preprocessing_phil()
    print("preprocessing done")

""" links I have used:
https://scikit-learn.org/1.5/modules/decomposition.html#incrementalpca
https://sklearn.org/1.8/auto_examples/decomposition/plot_incremental_pca.html
https://visualstudiomagazine.com/articles/2021/10/20/anomaly-detection-pca.aspx
https://ieeexplore.ieee.org/document/6200273
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score
"""
