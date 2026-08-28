# comparison.py - one shared metric function for every model
# ---------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torchmetrics.functional.retrieval import retrieval_precision, retrieval_recall


def metrics_from_scores(scores, labels_bin, k=100):
    """
    Compute ROC-AUC, average precision and precision/recall at k

    NOTE:
    values are p100 and r100 for historical reasons, but k
    is set to the number of anomalies present (50 on the test split). With a
    fixed k of 100 even a perfect model could only reach 0.5
    """
    scores = np.asarray(scores, dtype=np.float32)
    labels_bin = np.asarray(labels_bin).astype(int)

    roc_auc = roc_auc_score(labels_bin, scores)
    avg_precision = average_precision_score(labels_bin, scores)

    score_tensor = torch.tensor(scores)
    label_tensor = torch.tensor(labels_bin, dtype=torch.bool)

    # test set holds 50 anomalies, so a fixed P@100 could never exceed 0.5
    # set k at the anomaly count makes perfect model score 1.0 again
    top_k = min(k, int(label_tensor.sum()))
    precision_at_k = retrieval_precision(score_tensor, label_tensor, top_k=top_k).item()
    recall_at_k = retrieval_recall(score_tensor, label_tensor, top_k=top_k).item()

    return {
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "p100": precision_at_k,
        "r100": recall_at_k,
    }


def compare_on_common(model_dataframes, common_ids):
    n_common = len(common_ids)

    results = {}
    for model_name, df in model_dataframes.items():
        assert len(df) == n_common, f"{model_name}: {len(df)} rows, expected {n_common}"

        scores = df["score"].to_numpy()
        labels_bin = df["label_bin"].to_numpy()
        results[model_name] = metrics_from_scores(scores, labels_bin)

    return build_comparison_table(results)


def build_comparison_table(results):
    table = pd.DataFrame(results).T
    table = table[["roc_auc", "avg_precision", "p100", "r100"]]
    table = table.round(6)
    table_result = table.sort_values("roc_auc", ascending=False)
    return table_result
