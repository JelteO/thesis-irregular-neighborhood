import pandas as pd
import numpy as np
from src.models.baselines import PCABaseline, IsolationForestBaseline
from src.models.schreyer_ae import SchreyerAEBaseline
import torch
from src.data.preprocessing import preprocessing
from pygod.detector import DOMINANT
from src.data.split import make_entry_split, sample_common_test
import os
from pathlib import Path
from src.evaluation.comparison import metrics_from_scores

ROOT_DIR = Path(os.getcwd())
OUTPUT_DIR = ROOT_DIR / "outputs"

SEED = 42


def load_data():
    df_eval, df_eval_unscaled = preprocessing()
    return df_eval, df_eval_unscaled


def get_split_and_common(labels):
    # same split as everywhere else, derived from the label column
    # entry_id / node index is just the row position (data sorted in preprocessing.py)
    label_mapping = {"regular": 0, "local": 1, "global": 2}
    label_int = torch.tensor(labels.map(label_mapping).values)

    train_ids, val_ids, test_ids = make_entry_split(label_int, seed=SEED)
    common_ids = sample_common_test(label_int, test_ids, n_regular=10_000, seed=SEED)

    return label_int, train_ids, test_ids, common_ids


def feature_matrix(df):
    feature_cols = [col for col in df.columns if col not in ["entry_id", "label"]]
    x = df[feature_cols].to_numpy()
    return x


def scores_to_df(common_ids, label_int, scores, model_name):
    # scores over the common test subset, in common_ids order
    # entry_id is the global node index (row position), same as the gnn uses
    labels = label_int[common_ids].numpy()
    labels_bin = (labels != 0).astype(int)

    df_scores = pd.DataFrame(
        {
            "entry_id": common_ids.numpy(),
            "label": labels,
            "label_bin": labels_bin,
            "score": scores,
            "model": model_name,
        }
    )
    return df_scores


def experiment_pca():
    df_eval, _ = load_data()
    label_int, train_ids, _, common_ids = get_split_and_common(df_eval["label"])

    x_all = feature_matrix(df_eval)
    x_train = x_all[train_ids.numpy()]
    x_common = x_all[common_ids.numpy()]

    model = PCABaseline(n_components=0.95, random_state=SEED)
    model.fit(x_train=x_train)
    scores = model.score(x_eval=x_common)

    df_scores = scores_to_df(common_ids, label_int, scores, "pca")
    y_eval = df_scores["label_bin"].to_numpy()

    metrics = metrics_from_scores(scores, y_eval)
    return df_scores, metrics


def experiment_if():
    _, df_eval_unscaled = load_data()
    label_int, train_ids, _, common_ids = get_split_and_common(
        df_eval_unscaled["label"]
    )

    x_all = feature_matrix(df_eval_unscaled)
    x_train = x_all[train_ids.numpy()]
    x_common = x_all[common_ids.numpy()]

    if_model = IsolationForestBaseline(
        max_samples="auto", random_state=SEED, contamination="auto"
    )
    if_model.fit(x_train)
    scores = if_model.score(x_common)

    df_scores = scores_to_df(common_ids, label_int, scores, "if")
    y_eval = df_scores["label_bin"].to_numpy()

    metrics = metrics_from_scores(scores, y_eval)
    return df_scores, metrics


def experiment_schreyer(epochs=50, device="cpu"):
    df_eval, _ = load_data()
    label_int, train_ids, test_ids, common_ids = get_split_and_common(df_eval["label"])

    x_all = feature_matrix(df_eval)
    x_train = x_all[train_ids.numpy()]

    input_dim = x_all.shape[1]
    num_categorical = input_dim - 2  # 2 numerical columns

    model = SchreyerAEBaseline(
        input_dim=input_dim, num_categorical=num_categorical, device=device, alpha=0.8
    )
    model.fit(x_train=x_train, epochs=epochs)

    # score the full test set for the ranking csv
    x_test = x_all[test_ids.numpy()]
    scores_test = model.score(x_eval=x_test)

    labels_test = label_int[test_ids].numpy()
    labels_test_bin = (labels_test != 0).astype(int)

    df_ranked = pd.DataFrame(
        {
            "entry_id": test_ids.numpy(),
            "label": labels_test,
            "label_bin": labels_test_bin,
            "score": scores_test,
            "model": "schreyer_ae",
        }
    )
    df_ranked = df_ranked.sort_values("score", ascending=False).reset_index(drop=True)
    df_ranked["rank"] = np.arange(1, len(df_ranked) + 1)
    df_ranked.to_csv(
        f"{OUTPUT_DIR}/anomaly_ranking_schreyerae_baseline.csv", index=False
    )

    # score the common subset for the comparison table
    x_common = x_all[common_ids.numpy()]
    scores_common = model.score(x_eval=x_common)

    df_scores = scores_to_df(common_ids, label_int, scores_common, "schreyer_ae")

    y_eval = df_scores["label_bin"].to_numpy()
    metrics = metrics_from_scores(scores_common, y_eval)
    return df_scores, metrics


def experiment_dominant():
    graph_fit = torch.load("data/processed/graph_homo_fit.pt", weights_only=False)
    graph_eval = torch.load("data/processed/graph_homo_eval.pt", weights_only=False)

    # fixed hyperparameters from the literature, so no validation pass needed
    model = DOMINANT(hid_dim=64, num_layers=4, epoch=100, batch_size=0)
    model.fit(graph_fit)

    scores_all = model.decision_function(graph_eval)
    scores_all = np.asarray(scores_all)

    # common test entries sit after the fit entries in the eval graph
    n_fit = graph_eval.n_fit  # 10000, where the test entries start
    n_eval = graph_eval.n_entries  # 20050, where the backbone starts
    scores_eval = scores_all[n_fit:n_eval]  # (10050,) only the test entries

    common_ids = graph_eval.entry_ids[n_fit:n_eval]  # global row ids
    labels_common = graph_eval.y[n_fit:n_eval].numpy()
    y_eval = (labels_common != 0).astype(int)

    df_scores = pd.DataFrame(
        {
            "entry_id": common_ids.numpy(),
            "label": labels_common,
            "label_bin": y_eval,
            "score": scores_eval,
            "model": "dominant",
        }
    )

    metrics = metrics_from_scores(scores_eval, y_eval)

    dominant_explanations(model, graph_eval, scores_eval, common_ids)

    return df_scores, metrics


def dominant_explanations(model, graph_eval, scores_eval, common_ids, top_n=20):
    df_raw = pd.read_csv("data/processed/fraud_dataset_processed.csv")

    # field to column slicing (numeric cols first)
    info = graph_eval.entry_feature_info  # metadata
    field_slices = {}
    offset = info["num_numerical"]
    for field in info["categorical_order"]:
        n_cats = info["categorical_dims"][field]
        field_slices[field] = (offset, offset + n_cats)
        offset += n_cats

    model.model.eval()
    with torch.no_grad():
        x_reconstructed, _ = model.model(graph_eval.x, graph_eval.edge_index)

    n_fit = graph_eval.n_fit
    n_eval = graph_eval.n_entries
    # how far off was the RE
    squared_error = (graph_eval.x[:n_eval] - x_reconstructed[:n_eval]) ** 2

    # top 20 highest positions, best first
    top_positions = torch.argsort(torch.tensor(scores_eval), descending=True)[
        :top_n
    ].tolist()

    explanation_rows = []
    for rank, pos in enumerate(top_positions, start=1):
        # position counts within the common subset, shift by n_fit for the graph
        graph_pos = n_fit + pos
        row_id = int(common_ids[pos])

        # sum per blok
        per_field_error = {}
        for field, (a, b) in field_slices.items():
            per_field_error[field] = squared_error[graph_pos, a:b].sum().item()
        top_field = max(per_field_error, key=lambda f: per_field_error[f])

        # how often does it occur?
        value = df_raw.loc[row_id, top_field]
        value_count = int((df_raw[top_field] == value).sum())

        explanation_rows.append(
            {
                "rank": rank,
                "entry_row": row_id,
                "label": df_raw.loc[row_id, "label"],
                "top_field": top_field,
                "value": value,
                "value_count": value_count,
                "ground_truth_hit": value_count == 1,
            }
        )

    df_explanations = pd.DataFrame(explanation_rows)
    df_explanations.to_csv(f"{OUTPUT_DIR}/dominant_explanations.csv", index=False)
    hit_count = int(df_explanations["ground_truth_hit"].sum())
    print(f"DOMINANT exp hitrate: {hit_count}/{top_n}")
