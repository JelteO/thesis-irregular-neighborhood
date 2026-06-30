import pandas as pd
import numpy as np
from src.models.baselines import PCABaseline, IsolationForestBaseline
from src.models.schreyer_ae import SchreyerAEBaseline
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from torchmetrics.functional.retrieval import retrieval_precision, retrieval_recall
from src.data.preprocessing import preprocessing
from pygod.detector import DOMINANT
from src.data.graph_construction import create_graphs
import os
from pathlib import Path

ROOT_DIR = Path(os.getcwd())
OUTPUT_DIR = ROOT_DIR / "outputs"

def load_data():
    df_train, df_eval, df_train_unscaled, df_eval_unscaled = preprocessing()
    return df_train, df_eval, df_train_unscaled, df_eval_unscaled


def split_data(df_train, df_eval):
    feature_cols = [col for col in df_train.columns if col not in ["entry_id", "label"]]

    x_train = df_train[feature_cols].to_numpy()
    x_eval = df_eval[feature_cols].to_numpy()

    y_eval = (df_eval["label"] != "regular").astype(int).to_numpy()
    return x_train, x_eval, y_eval


def scoring_metric(y_eval, scores):
    if hasattr(scores, "numpy"):
        scores = scores.numpy()

    roc_auc = roc_auc_score(y_true=y_eval, y_score=scores)
    avg_prec = average_precision_score(y_true=y_eval, y_score=scores)

    target = torch.tensor(y_eval, dtype=torch.bool)
    predict = torch.tensor(scores.astype(np.float32), dtype=torch.float32)

    p_at_100 = retrieval_precision(preds=predict, target=target, top_k=100).item()
    r_at_100 = retrieval_recall(preds=predict, target=target, top_k=100).item()

    return {
        "roc_auc": roc_auc,
        "avg_precision": avg_prec,
        "p100": p_at_100,
        "r100": r_at_100,
    }


def experiment_pca():
    df_train, df_eval, _, _ = load_data()
    x_train, x_eval, y_eval = split_data(df_train, df_eval)

    model = PCABaseline(n_components=0.95, random_state=42)
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

    metrics = scoring_metric(y_eval=y_eval, scores=scores)
    return df_scores, metrics


def experiment_if():
    _, _, df_train_unscaled, df_eval_unscaled = load_data()
    x_train_unscaled_np, x_eval_unscaled_np, y_eval = split_data(
        df_train_unscaled, df_eval_unscaled
    )

    if_model = IsolationForestBaseline(
        max_samples="auto", random_state=42, contamination="auto"
    )
    if_model.fit(x_train_unscaled_np)
    scores = if_model.score(x_eval_unscaled_np)

    df_scores = pd.DataFrame(
        {
            "entry_id": df_eval_unscaled["entry_id"].values,
            "label": df_eval_unscaled["label"].values,
            "label_bin": y_eval,
            "score": scores,
            "model": "if",
        }
    )

    metrics = scoring_metric(y_eval=y_eval, scores=scores)
    return df_scores, metrics


def experiment_schreyer(epochs=10, device="cpu"):
    df_train, df_eval, _, _ = load_data()
    x_train, x_eval, y_eval = split_data(df_train, df_eval)

    input_dim = x_train.shape[1]
    num_categorical = input_dim - 2  # 2 numerical columns

    model = SchreyerAEBaseline(
        input_dim=input_dim, num_categorical=num_categorical, device=device
    )
    model.fit(x_train=x_train, epochs=epochs)
    scores = model.score(x_eval=x_eval)

    df_scores = pd.DataFrame(
        {
            "entry_id": df_eval["entry_id"].values,
            "label": df_eval["label"].values,
            "label_bin": y_eval,
            "score": scores,
            "model": "schreyer_ae",
        }
    )

    metrics = scoring_metric(y_eval=y_eval, scores=scores)

    df_ranked = df_scores.sort_values("score", ascending=False).reset_index(drop=True)
    df_ranked["rank"] = np.arange(1, len(df_ranked) + 1)
    df_ranked.to_csv(
        f"{OUTPUT_DIR}/anomaly_ranking_schreyerae_baseline.csv", index=False
    )

    return df_scores, metrics


def experiment_dominant():
    _, _, _, _ = load_data()
    create_graphs()
    graph = torch.load("data/processed/graph_homo_sample.pt", weights_only=False)

    model = DOMINANT(hid_dim=64, num_layers=4, epoch=100, batch_size=0)
    model.fit(graph)

    scores = model.decision_score_
    n_entries = graph.n_entries

    y_eval = (graph.y[:n_entries] != 0).numpy().astype(int)
    scores_eval = scores[:n_entries]

    df_scores = pd.DataFrame(
        {
            "entry_id": np.arange(n_entries),
            "label_bin": y_eval,
            "score": scores_eval,
            "model": "dominant",
        }
    )

    metrics = scoring_metric(y_eval=y_eval, scores=scores_eval)
    return df_scores, metrics
