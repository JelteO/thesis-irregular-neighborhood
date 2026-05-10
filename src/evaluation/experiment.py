import pandas as pd
import numpy as np
import os

from src.models.baselines import PCABaseline
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from torchmetrics.functional.retrieval import retrieval_precision, retrieval_recall


def load_data():
    root_dir = os.getcwd()
    df = pd.read_csv(f"{root_dir}/data/processed/fraud_dataset_processed.csv")

    df_train = df[df["label"] == "regular"].copy()
    df_eval = df.copy()
    return df_train, df_eval, df


def experiment_pca(n_components=2):
    df_train, df_eval, df = load_data()

    FEATURE_COLUMS = [col for col in df.columns if col not in ["entry_id", "label"]]

    x_train = df_train[FEATURE_COLUMS].to_numpy()
    x_eval = df_eval[FEATURE_COLUMS].to_numpy()
    y_eval = (df_eval["label"] != "regular").astype(int).to_numpy()

    model = PCABaseline(n_components=n_components, random_state=42)
    model.fit(x_train=x_train)
    scores = model.score(x_eval=x_eval)
    df_scores = pd.DataFrame(
        {
            "entry_id": df_eval["entry_id"].values,
            "label": df_eval["label"].values,
            "label_bin": y_eval,
            "score": scores,
            "model": "pca",
        }
    )

    print("X train shape:", x_train.shape)
    print("PCA n_components:", model.pca.n_components)
    print("PCA explained variance ratio:", model.pca.explained_variance_ratio_)
    print("PCA total explained variance:", model.pca.explained_variance_ratio_.sum())

    # df_scores_sort = df_scores.sort_values("score", ascending=False)

    print(f"shape y_eval {y_eval.shape}, shape scores {scores.shape}")
    roc_auc_scores = roc_auc_score(y_true=y_eval, y_score=scores)
    avg_prec_scores = average_precision_score(y_true=y_eval, y_score=scores)

    print(f"roc_auc_score: {roc_auc_scores}")
    print(f"avg_precision_score: {avg_prec_scores}")

    pd.set_option("display.max.columns", 10)
    print(df_scores.head(20))

    target = torch.tensor(y_eval, dtype=torch.bool)
    predict = torch.tensor(scores, dtype=torch.float32)

    print("target shape:", target.shape)
    print("target dtype:", target.dtype)
    print("target first 20:", target[:20])
    print("target positives:", target.sum().item())
    print("target negatives:", (~target).sum().item())
    print("unique target values:", torch.unique(target, return_counts=True))

    print(torch.unique(target, return_counts=True))

    precision_at_100 = retrieval_precision(preds=predict, target=target, top_k=100)
    recall_at_100 = retrieval_recall(preds=predict, target=target, top_k=100)

    print("P@100:", precision_at_100.item())
    print("R@100:", recall_at_100.item())

    df_ranked = df_scores.sort_values("score", ascending=False).reset_index(drop=True)
    df_ranked["rank"] = np.arange(1, len(df_ranked) + 1)

    print(df_ranked[df_ranked["label"] != "regular"].head(20))


def experiment_if():
    df_train, df_eval, df = load_data()
    
    