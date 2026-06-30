# main.py
# Runs the whole pipeline: preprocessing, graph building, GNN
# training and explainability for both datasets, and finally the baselines.
# Just run `python main.py` and everything happens in order.

import os
import torch
from src.data.preprocessing import preprocessing
from src.data.preprocessing_phil import preprocessing_phil
from src.data.graph_construction import create_graphs
from src.data.graph_construction_phil import create_graphs_phil
from src.models.gnn_model import run_gnn
from src.evaluation.experiment import (
    experiment_pca,
    experiment_if,
    experiment_schreyer,
    experiment_dominant,
)
from src.models.seed import set_seed
from src.evaluation.comparison import common_entry_ids, compare_on_common
import pandas as pd

from pathlib import Path

ROOT_DIR = Path.cwd()
OUTPUT_DIR = ROOT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SEED = 42
EPOCHS = 20


def run_schreyer():
    # only preprocess if we haven't already
    if not os.path.exists("data/processed/fraud_dataset_processed.csv"):
        print("preprocessing schreyer data...")
        preprocessing()
    else:
        print("processed schreyer data already exists, skipping")

    # only build the graph if it isn't there yet
    if not os.path.exists("data/processed/graph_hetero.pt"):
        print("building schreyer graph...")
        create_graphs()
    else:
        print("schreyer graph already exists, skipping")

    return run_gnn(dataset="schreyer", has_labels=True, seed=SEED, epochs=EPOCHS)


def run_philadelphia():
    if not os.path.exists("data/processed/city_payments_processed.csv"):
        print("preprocessing philadelphia data...")
        preprocessing_phil()
    else:
        print("processed philadelphia data already exists, skipping")

    if not os.path.exists("data/processed/graph_hetero_phil.pt"):
        print("building philadelphia graph...")
        create_graphs_phil()
    else:
        print("philadelphia graph already exists, skipping")

    return run_gnn(dataset="philadelphia", has_labels=False, seed=SEED, epochs=EPOCHS)


def run_baselines():
    # Run 4 baselines. These only apply ot the labelled dataset

    scores_by_model = {}

    for name, experiment in [
        ("PCA", experiment_pca),
        ("isolation forest", experiment_if),
        ("schreyer AE", experiment_schreyer),
        ("DOMINANT", experiment_dominant),
    ]:
        # one broken baseline shouldn't stop the others, wrap each one
        try:
            print(f"\nrunning baseline: {name}")
            set_seed(seed=SEED)
            df_scores, metrics = experiment()
            print(f"{name}: {metrics}")
            scores_by_model[name] = df_scores
        except Exception as e:
            print(f"baseline {name} failed: {e}")
    return scores_by_model


if __name__ == "__main__":
    _, _, df_gnn_common = run_schreyer()
    run_philadelphia()
    dfs = run_baselines()
    dfs["GNN (hetero)"] = df_gnn_common

    labels = pd.read_csv("data/processed/fraud_dataset_processed.csv")["label"]
    common_ids = common_entry_ids((labels != "regular").astype(int))
    table = compare_on_common(dfs, common_ids)

    print("\nComparison Table of common 10.100 entries")
    print(table.to_string())
    table.to_csv(f"{OUTPUT_DIR}/comparison_common_10100.csv")
    print("\nMain.py done")
