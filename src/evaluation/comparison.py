import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torchmetrics.functional.retrieval import retrieval_precision, retrieval_recall


def common_entry_ids(entry_labels, n_regular=10_000, seed=42):
    labels = torch.as_tensor(entry_labels)
    anomaly_ids = (labels != 0).nonzero(as_tuple=True)[0]
    regular_ids = (labels == 0).nonzero(as_tuple=True)[0]

    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(len(regular_ids), generator=generator)
    sampled_regular_ids = regular_ids[shuffled[:n_regular]]

    return torch.cat([anomaly_ids, sampled_regular_ids])


def metrics_from_scores(scores, labels_bin, k=100):
    scores = np.asarray(scores, dtype=np.float32)
    labels_bin = np.asarray(labels_bin).astype(int)

    roc_auc = roc_auc_score(labels_bin, scores)
    avg_precision = average_precision_score(labels_bin, scores)

    score_tensor = torch.tensor(scores)
    label_tensor = torch.tensor(labels_bin, dtype=torch.bool)
    precision_at_k = retrieval_precision(score_tensor, label_tensor, top_k=k).item()
    recall_at_k = retrieval_recall(score_tensor, label_tensor, top_k=k).item()

    return {
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "p100": precision_at_k,
        "r100": recall_at_k,
    }


def compare_on_common(model_dataframes, common_ids):
    n_common = len(common_ids)
    positions = common_ids.numpy()

    results = {}
    for model_name, df in model_dataframes.items():
        if len(df) == n_common:
            common_rows = df
        else:
            common_rows = df.iloc[positions]

        scores = common_rows["score"].to_numpy()
        labels_bin = common_rows["label_bin"].to_numpy()
        results[model_name] = metrics_from_scores(scores, labels_bin)

    return build_comparison_table(results)


def build_comparison_table(results):
    table = pd.DataFrame(results).T
    table = table[["roc_auc", "avg_precision", "p100", "r100"]]
    table = table.round(4)
    table_result = table.sort_values("roc_auc", ascending=False)
    return table_result
