# gnn_model.py - hetero GNN autoencoder, training, scoring, audit
# in: graph_hetero.pt (533239 nodes) and the split from split.py
# out: anomaly ranking csv, ablation csvs, audit table, plots
# ---------------------------------------------------------------

from torch_geometric.nn import SAGEConv, HeteroConv
import torch
import os
from torch.nn import Embedding, Linear, MSELoss, CrossEntropyLoss, ModuleDict
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
from torch.optim.adam import Adam
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random
from torch_geometric.data import HeteroData
from src.data.split import make_entry_split, sample_common_test
from src.evaluation.comparison import metrics_from_scores

pd.options.display.max_columns = 9

device = torch.device("cpu")
from pathlib import Path

ROOT_DIR = Path(os.getcwd())
OUTPUT_DIR = ROOT_DIR / "outputs"

EMBEDDING_DIM = 16
HIDDEN_DIM = 64
GAMMA = 0.65
ALPHA = 1.0


def plot_score_distribution(scores, save_path, title="Score distribution"):
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=80)
    plt.yscale("log")
    plt.xlabel("anomaly score")
    plt.ylabel("count (log scale)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"score distribution plot saved: {save_path}")


def plot_latent_space(latent_vectors, labels, save_path, scores=None, title="plot"):
    labels = labels.astype(int)
    n_points = len(latent_vectors)

    rng = np.random.default_rng(42)

    if scores is None:
        # Labelled set
        # only 10 anomalies in the test set, so use all of these entries
        # fill the rest of the sample with regular entries
        anomaly_idx = np.where(labels != 0)[0]
        regular_pool = np.where(labels == 0)[0]

        n_regular = min(len(regular_pool), 3000 - len(anomaly_idx))
        regular_idx = rng.choice(regular_pool, size=n_regular, replace=False)

        sample_idx = np.concatenate([anomaly_idx, regular_idx])
    else:
        # Unlabelled set
        # plain random sample
        n_sample = min(3000, n_points)
        sample_idx = rng.choice(n_points, size=n_sample, replace=False)

    latent_subset = latent_vectors[sample_idx]

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init="pca")
    latent_2d = tsne.fit_transform(latent_subset)

    plt.figure(figsize=(8, 6))

    if scores is not None:
        scores_subset = scores[sample_idx]

        # normalise scores to 0-1
        # then a power < 1 to stretch high end
        score_min = scores_subset.min()
        score_max = scores_subset.max()
        score_norm = (scores_subset - score_min) / (score_max - score_min + 1e-12)
        color_val = score_norm**0.3

        # draw low scores first so the high ones end up on top
        order = np.argsort(color_val)
        sc = plt.scatter(
            latent_2d[order, 0],
            latent_2d[order, 1],
            c=color_val[order],
            cmap="inferno",
            s=18,
        )
        plt.colorbar(sc, label="anomaly score (power-scaled)")
    else:
        labels_subset = labels[sample_idx]

        # Schreyer is labelled
        # color each journal entry by anomaly type
        # draw first regular, then the anomalies on top so they stay visible
        class_names = {0: "regular", 1: "local", 2: "global"}
        class_sizes = {0: 12, 1: 60, 2: 60}
        for class_value in [0, 1, 2]:
            mask = labels_subset == class_value
            plt.scatter(
                latent_2d[mask, 0],
                latent_2d[mask, 1],
                s=class_sizes[class_value],
                label=class_names[class_value],
            )

        plt.legend()

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"latent space plot saved: {save_path}")


def split_entry_targets(x_entry, num_numerical, cat_dims, cat_order):
    num_target = x_entry[:, :num_numerical]
    cat_targets = {}

    offset_index = num_numerical
    for field in cat_order:
        n_cat = cat_dims[field]
        one_hot = x_entry[:, offset_index : offset_index + n_cat]
        cat_targets[field] = one_hot.argmax(dim=1)
        offset_index += n_cat
    return num_target, cat_targets


# why: with all edges present, a train entry can see test features in two hops,
# so the shared backbone leaks held-out features into the weights.
# Filtering edges per phase closes that path.
# All nodes stay in graph, so entry ids remain global and the audit code keeps working
# data: train graph keeps 426327 of 533009 posts_to edges, train+val 479667
def graph_with_entry_edges(graph, allowed_entries):
    keep = torch.zeros(graph["entry"].num_nodes, dtype=torch.bool)
    keep[allowed_entries] = True

    filtered = HeteroData()
    filtered["entry"].x = graph["entry"].x
    filtered["entry"].y = graph["entry"].y
    filtered["gl_account"].x = graph["gl_account"].x
    filtered["gl_account"].y = graph["gl_account"].y
    filtered["profit_center"].x = graph["profit_center"].x
    filtered["profit_center"].y = graph["profit_center"].y

    filtered.entry_feature_info = graph.entry_feature_info
    filtered.num_gl_accounts = graph.num_gl_accounts
    filtered.num_profit_centers = graph.num_profit_centers

    for edge_type in graph.edge_types:
        edge_index = graph[edge_type].edge_index
        source_type = edge_type[0]

        if source_type == "entry":
            mask = keep[edge_index[0]]
        else:
            mask = keep[edge_index[1]]

        filtered[edge_type].edge_index = edge_index[:, mask]

    return filtered


@torch.no_grad()
def compute_entry_re_score(model, x_dict, edge_index_dict, gamma):
    model.eval()

    # structural nodes are embedding indices
    # cast to long before lookup
    x_dict = dict(x_dict)
    x_dict["gl_account"] = x_dict["gl_account"].long()
    x_dict["profit_center"] = x_dict["profit_center"].long()

    z_dict = model.encoder(x_dict, edge_index_dict)
    num_pred, cat_logits = model.decoder(z_dict["entry"])

    num_target, cat_targets = split_entry_targets(
        x_dict["entry"], NUM_NUMERICAL, CAT_DIMS, CAT_ORDER
    )

    numerical_error = ((num_pred - num_target) ** 2).mean(dim=1)

    categorical_errors = []
    for field in CAT_ORDER:
        field_error = F.cross_entropy(
            cat_logits[field], cat_targets[field], reduction="none"
        )
        categorical_errors.append(field_error)
    categorical_error = torch.stack(categorical_errors, dim=1).mean(dim=1)

    re_score = gamma * categorical_error + (1 - gamma) * numerical_error

    # index 0 is the target entry
    return re_score[0].item()


@torch.no_grad()
def occlusion_explain(model, subgraph, gamma):
    model.eval()

    baseline_score = compute_entry_re_score(
        model, subgraph.x_dict, subgraph.edge_index_dict, gamma
    )

    # neighbor types to occlude, with the edges connecting them to the entry
    # (both directions, the graph is bidirectional)
    neighbours_to_occlude = {
        "gl_account (vendor)": [
            ("entry", "posts_to", "gl_account"),
            ("gl_account", "posted_by", "entry"),
        ],
        "profit_center (department)": [
            ("entry", "assigned_to", "profit_center"),
            ("profit_center", "contains", "entry"),
        ],
    }

    results = []

    for neighbour_name, edge_types in neighbours_to_occlude.items():
        occluded_edges = dict(subgraph.edge_index_dict)
        for edge_type in edge_types:
            empty_edge = torch.empty((2, 0), dtype=torch.long)
            occluded_edges[edge_type] = empty_edge

        occluded_score = compute_entry_re_score(
            model, subgraph.x_dict, occluded_edges, gamma
        )

        # positive drop means the neighbour was pushing the score up
        score_drop = baseline_score - occluded_score
        if baseline_score != 0:
            drop_percentage = (score_drop / baseline_score) * 100
        else:
            drop_percentage = 0.0

        results.append(
            {
                "neighbour": neighbour_name,
                "baseline_score": baseline_score,
                "occluded_score": occluded_score,
                "score_drop": score_drop,
                "drop_percentage": drop_percentage,
            }
        )

    return results


@torch.no_grad()
def feature_error_breakdown(model, subgraph, gamma):
    model.eval()

    x_dict = dict(subgraph.x_dict)
    x_dict["gl_account"] = x_dict["gl_account"].long()
    x_dict["profit_center"] = x_dict["profit_center"].long()

    z_dict = model.encoder(x_dict, subgraph.edge_index_dict)
    num_pred, cat_logits = model.decoder(z_dict["entry"])

    num_target, cat_targets = split_entry_targets(
        x_dict["entry"], NUM_NUMERICAL, CAT_DIMS, CAT_ORDER
    )

    breakdown = {}

    numerical_error = ((num_pred[0] - num_target[0]) ** 2).mean().item()
    weighted = {"numerical (amount)": (1 - gamma) * numerical_error}

    for field in CAT_ORDER:
        field_error = F.cross_entropy(
            cat_logits[field][0:1], cat_targets[field][0:1]
        ).item()
        weighted[field] = (gamma / len(CAT_ORDER)) * field_error

    total = sum(weighted.values())
    if total > 0:
        breakdown = {field: value / total for field, value in weighted.items()}
    else:
        breakdown = weighted

    return breakdown


@torch.no_grad()
def build_audit_table(model, graph, df_best_test, gamma, top_n=20):
    model.eval()

    top_entries = df_best_test.head(top_n)
    audit_rows = []

    for _, row in tqdm(top_entries.iterrows(), total=len(top_entries), desc="audit"):
        entry_id = int(row["entry_id"])

        # data: input_nodes points at one flagged entry with batch_size=1,
        # so the loader samples the 2-hop subgraph around it. PyG always puts
        # the input node at position 0 of the batch, which is why the
        # explainability functions below can simply read index [0]
        single_loader = NeighborLoader(
            data=graph,
            num_neighbors=[10] * 2,
            batch_size=1,
            input_nodes=("entry", torch.tensor([entry_id])),
            subgraph_type="bidirectional",
        )
        subgraph = next(iter(single_loader))

        # occlusion: how much does each neighbour matter?
        occlusion_results = occlusion_explain(model, subgraph, gamma)
        vendor_drop = occlusion_results[0]["drop_percentage"]
        department_drop = occlusion_results[1]["drop_percentage"]

        # feature breakdown: which attribute was hardest to reconstruct?
        feature_breakdown = feature_error_breakdown(model, subgraph, gamma)
        top_feature = max(feature_breakdown.items(), key=lambda item: item[1])

        audit_rows.append(
            {
                "rank": int(row["rank"]),
                "entry_id": entry_id,
                "score": round(float(row["score"]), 4),
                "vendor_drop_pct": round(vendor_drop, 1),
                "department_drop_pct": round(department_drop, 1),
                "top_feature": top_feature[0],
                "top_feature_error": round(top_feature[1], 4),
            }
        )

    return pd.DataFrame(audit_rows)


class Encoder(torch.nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_pc, num_gl):
        super().__init__()
        self.emb_pc = Embedding(num_embeddings=num_pc, embedding_dim=embedding_dim)
        self.emb_gl = Embedding(num_embeddings=num_gl, embedding_dim=embedding_dim)
        self.conv1 = HeteroConv(
            {
                ("entry", "posts_to", "gl_account"): SAGEConv((-1, -1), hidden_dim),
                ("gl_account", "posted_by", "entry"): SAGEConv((-1, -1), hidden_dim),
                ("entry", "assigned_to", "profit_center"): SAGEConv(
                    (-1, -1), hidden_dim
                ),
                ("profit_center", "contains", "entry"): SAGEConv((-1, -1), hidden_dim),
            },
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {
                ("entry", "posts_to", "gl_account"): SAGEConv((-1, -1), hidden_dim),
                ("gl_account", "posted_by", "entry"): SAGEConv((-1, -1), hidden_dim),
                ("entry", "assigned_to", "profit_center"): SAGEConv(
                    (-1, -1), hidden_dim
                ),
                ("profit_center", "contains", "entry"): SAGEConv((-1, -1), hidden_dim),
            },
            aggr="sum",
        )

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            "entry": x_dict["entry"],  # 388-dim float
            "gl_account": self.emb_gl(x_dict["gl_account"].squeeze(-1)),
            "profit_center": self.emb_pc(x_dict["profit_center"].squeeze(-1)),
        }
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict


class Decoder(torch.nn.Module):
    def __init__(self, hidden_dim, num_numerical, cat_dims):
        super().__init__()
        self.num_lin = Linear(hidden_dim, num_numerical)
        self.cat_lin = ModuleDict(
            {field: Linear(hidden_dim, n_cat) for field, n_cat in cat_dims.items()}
        )

    def forward(self, z):
        num_pred = self.num_lin(z)
        cat_logits = {field: head(z) for field, head in self.cat_lin.items()}
        return num_pred, cat_logits


class Model(torch.nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim, num_pc, num_gl, num_numerical, cat_dims
    ):
        super().__init__()
        self.encoder = Encoder(hidden_dim, embedding_dim, num_pc, num_gl)
        self.decoder = Decoder(hidden_dim, num_numerical, cat_dims)

    def forward(self, x_dict, edge_index_dict, batch_size):
        z_entry = self.encode(x_dict, edge_index_dict, batch_size)
        num_pred, cat_logits = self.decoder(z_entry)
        return num_pred, cat_logits

    def encode(self, x_dict, edge_index_dict, batch_size):
        """function to retrieve latent space"""
        z_dict = self.encoder(x_dict, edge_index_dict)
        return z_dict["entry"][:batch_size]


mse_mean = MSELoss(reduction="mean")
ce_mean = CrossEntropyLoss(reduction="mean")


def train():
    model.train()
    total_loss, total_mse, total_ce, total_examples = 0.0, 0.0, 0.0, 0.0

    for batch in tqdm(neighbor_train):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch["entry"].batch_size

        num_pred, cat_logits = model(batch.x_dict, batch.edge_index_dict, batch_size)

        x_entry = batch.x_dict["entry"][:batch_size]
        num_target, cat_targets = split_entry_targets(
            x_entry=x_entry,
            num_numerical=NUM_NUMERICAL,
            cat_dims=CAT_DIMS,
            cat_order=CAT_ORDER,
        )

        mse = mse_mean(num_pred, num_target)
        ce_per_field = [ce_mean(cat_logits[f], cat_targets[f]) for f in CAT_ORDER]
        ce_average = torch.stack(ce_per_field).mean()

        loss = GAMMA * ce_average + (1 - GAMMA) * mse

        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size
        total_mse += float(mse) * batch_size
        total_ce += float(ce_average) * batch_size

    return (
        total_loss / total_examples,
        total_mse / total_examples,
        total_ce / total_examples,
    )


@torch.no_grad()
def calc_regular_centroid(loader):
    model.eval()
    all_z = []
    for batch in tqdm(loader, desc="centroid"):
        batch = batch.to(device)
        batch_size = batch["entry"].batch_size
        z = model.encode(batch.x_dict, batch.edge_index_dict, batch_size)
        all_z.append(z)
    all_z = torch.cat(all_z, dim=0)
    centroid = all_z.mean(dim=0)
    return centroid


@torch.no_grad()
def evaluate(loader, centroid, epoch=None, split_str=None, alpha=None, gamma=None):
    model.eval()

    if alpha is None:
        alpha = ALPHA
    if gamma is None:
        gamma = GAMMA

    targets, entry_ids = [], []
    re_all, sd_all, z_all = [], [], []

    for batch in tqdm(loader):
        batch = batch.to(device)
        batch_size = batch["entry"].batch_size

        z_entry = model.encode(batch.x_dict, batch.edge_index_dict, batch_size)
        num_pred, cat_logits = model.decoder(z_entry)

        x_entry = batch.x_dict["entry"][:batch_size]
        num_target, cat_targets = split_entry_targets(
            x_entry, NUM_NUMERICAL, CAT_DIMS, CAT_ORDER
        )

        ############## RE Component #####################
        # per entry MSE on amounts
        num_err = ((num_pred - num_target) ** 2).mean(dim=1)

        # per entry CE per field
        cat_err_per_field = []
        for field in CAT_ORDER:
            ce = F.cross_entropy(
                cat_logits[field], cat_targets[field], reduction="none"
            )
            cat_err_per_field.append(ce)
        cat_err = torch.stack(cat_err_per_field, dim=1).mean(dim=1)

        # reconstruction score per entry
        re_score = gamma * cat_err + (1 - gamma) * num_err

        ############## SD Component #####################
        sd_score = ((z_entry - centroid) ** 2).sum(dim=1)

        re_all.append(re_score)
        sd_all.append(sd_score)
        z_all.append(z_entry)

        target = batch["entry"].y[:batch_size]
        entry_id = batch["entry"].n_id[:batch_size]
        targets.append(target)
        entry_ids.append(entry_id)

    re_all = torch.cat(re_all, dim=0)
    sd_all = torch.cat(sd_all, dim=0)
    z_all = torch.cat(z_all, dim=0)

    # RE and SD on different scales (cross-entropy vs squared distance),
    # min-max puts both on [0,1] so alpha mixes fairly data
    re_normal = (re_all - re_all.min()) / (re_all.max() - re_all.min() + 1e-12)
    sd_normal = (sd_all - sd_all.min()) / (sd_all.max() - sd_all.min() + 1e-12)

    score = alpha * re_normal + (1 - alpha) * sd_normal

    all_preds = score.numpy()
    all_targets = torch.cat(targets, dim=0).numpy()
    all_ids = torch.cat(entry_ids, dim=0).numpy()
    all_bin = (all_targets != 0).astype(int)

    if HAS_LABELS:
        shared_metrics = metrics_from_scores(all_preds, all_bin)
        roc = shared_metrics["roc_auc"]
        p100 = shared_metrics["p100"]
        r100 = shared_metrics["r100"]
    else:
        roc, p100, r100 = None, None, None

    score_series = pd.Series(all_preds)
    rank = score_series.rank(method="dense", ascending=False).astype(int).to_numpy()

    df_epoch = pd.DataFrame(
        {
            "entry_id": all_ids,
            "label": all_targets,
            "label_bin": all_bin,
            "score": all_preds,
            "re_norm": re_normal.numpy(),
            "sd_norm": sd_normal.numpy(),
            "model": "gnn_model",
            "epoch": epoch if epoch is not None else 0,
            "rank": rank,
            "split": split_str,
            "roc_auc": roc,
            "p100": p100,
            "r100": r100,
        }
    )
    metrics = {"roc_auc": roc, "p100": p100, "r100": r100}
    return metrics, df_epoch, z_all.numpy(), all_targets


# why: gamma sits inside the training loss, so every value needs its own
# retrained model (10 epochs each).
def gamma_ablation(gamma_sweep, epochs=10):
    global model, optimizer, GAMMA
    gamma_iter = []
    for g in gamma_sweep:
        GAMMA = g
        model = Model(
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_pc=NUM_PC,
            num_gl=NUM_GL,
            num_numerical=NUM_NUMERICAL,
            cat_dims=CAT_DIMS,
        )
        optimizer = Adam(model.parameters(), lr=0.0001)

        for e in range(epochs):
            train()
        centroid = calc_regular_centroid(neighbor_train)
        test_metrics, _, _, _ = evaluate(
            loader=neighbor_val,
            centroid=centroid,
            epoch=epochs,
            split_str="val",
            alpha=1.0,
            gamma=g,
        )
        gamma_iter.append({"gamma": g, **test_metrics})
    df_gamma = pd.DataFrame(gamma_iter)
    df_gamma.to_csv(f"{OUTPUT_DIR}/ablation_gamma_{DATASET}.csv", index=False)

    GAMMA = 0.65
    model = Model(
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_pc=NUM_PC,
        num_gl=NUM_GL,
        num_numerical=NUM_NUMERICAL,
        cat_dims=CAT_DIMS,
    )
    model.load_state_dict(
        torch.load(
            f"{ROOT_DIR}/data/processed/best_model_{DATASET}.pt", weights_only=True
        )
    )
    return df_gamma


def run_gnn(dataset, has_labels, seed=42, epochs=20) -> tuple:
    global DATASET, HAS_LABELS, graph
    global NUM_GL, NUM_PC, NUM_NUMERICAL, CAT_DIMS, CAT_ORDER, OUTPUT_DIM
    global model, optimizer
    global neighbor_train, neighbor_val, neighbor_test, neighbor_common
    global EPOCHS

    DATASET = dataset
    HAS_LABELS = has_labels
    EPOCHS = epochs

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if DATASET == "schreyer":
        graph = torch.load(
            f"{ROOT_DIR}/data/processed/graph_hetero.pt", weights_only=False
        )
    else:
        graph = torch.load(
            f"{ROOT_DIR}/data/processed/graph_hetero_phil.pt", weights_only=False
        )

    OUTPUT_DIM = graph["entry"].x.shape[1]
    NUM_GL = graph.num_gl_accounts
    NUM_PC = graph.num_profit_centers
    NUM_NUMERICAL = graph.entry_feature_info["num_numerical"]
    CAT_DIMS = graph.entry_feature_info["categorical_dims"]
    CAT_ORDER = graph.entry_feature_info["categorical_order"]

    # three phase protocol: fit on the train-only graph, select epoch and
    # alpha/gamma on the train+val graph, score test once on the full graph.
    #
    # why: in a graph,a plain loss mask is not enough. With all edges present
    # the features of held-out entries still reach the train entries through
    # the shared gl/pc nodes during message passing, so the edges themselves
    # have to be split per phase
    if HAS_LABELS:
        train_ids, val_ids, test_ids = make_entry_split(graph["entry"].y, seed)
        torch.save(
            {"train": train_ids, "val": val_ids, "test": test_ids},
            f"{ROOT_DIR}/data/processed/split_{DATASET}.pt",
        )
        graph_train = graph_with_entry_edges(graph, train_ids)
        train_and_val = torch.cat([train_ids, val_ids])
        graph_trainval = graph_with_entry_edges(graph, train_and_val)
    else:
        # unlabelled deployment: no model selection happens, so training
        # and scoring on everything leaks nothing
        all_entries = torch.arange(graph["entry"].num_nodes)
        train_ids = all_entries
        val_ids = None
        test_ids = all_entries
        graph_train = graph
        graph_trainval = graph

    neighbor_train = NeighborLoader(
        data=graph_train,
        input_nodes=("entry", train_ids),
        shuffle=True,
        num_neighbors=[10] * 2,
        batch_size=128,
        subgraph_type="bidirectional",
    )

    neighbor_val = None
    if HAS_LABELS:
        neighbor_val = NeighborLoader(
            data=graph_trainval,
            input_nodes=("entry", val_ids),
            shuffle=False,
            num_neighbors=[10] * 2,
            batch_size=128,
            subgraph_type="bidirectional",
        )

    # test pass runs on the FULL graph, all edges allowed at this point
    neighbor_test = NeighborLoader(
        data=graph,
        input_nodes=("entry", test_ids),
        shuffle=False,
        num_neighbors=[10] * 2,
        batch_size=128,
        subgraph_type="bidirectional",
    )

    neighbor_common = None
    if HAS_LABELS:
        common_ids = sample_common_test(
            graph["entry"].y, test_ids, n_regular=10_000, seed=seed
        )
        # scored on the FULL graph: this is a test-time pass, edges allowed
        neighbor_common = NeighborLoader(
            data=graph,
            input_nodes=("entry", common_ids),
            shuffle=False,
            num_neighbors=[10] * 2,
            batch_size=128,
            subgraph_type="bidirectional",
        )

    model = Model(
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_pc=NUM_PC,
        num_gl=NUM_GL,
        num_numerical=NUM_NUMERICAL,
        cat_dims=CAT_DIMS,
    )
    optimizer = Adam(model.parameters(), lr=0.0001)

    best_epoch = EPOCHS
    best_val_roc = -1.0
    all_rows = []

    # why: the score is alpha*RE + (1-alpha)*SD.
    # Epoch selection runs with alpha=1.0, so the SD term gets weight zero and the
    # centroid value does not influence anything.
    # A placeholder of zeros fills the argument and skips a full pass over 426k train
    # entries per epoch. The real centroid is computed once after training, for the
    # alpha ablation where alpha < 1
    # data: centroid shape (64,), the latent dim of the encoder
    zero_centroid = torch.zeros(HIDDEN_DIM)

    for epoch in range(1, EPOCHS + 1):
        loss, mse, ce = train()
        print(f"Epoch {epoch:02d} | Loss {loss:.4f} (MSE {mse:.4f}, CE {ce:.4f})")

        if HAS_LABELS:
            val_metrics, _, _, _ = evaluate(
                neighbor_val, zero_centroid, epoch, "val", alpha=1.0, gamma=GAMMA
            )
            print(f"         | val ROC {val_metrics['roc_auc']:.4f}")

            # roc is never None here, this branch only runs when labels exist
            val_roc = val_metrics["roc_auc"]
            if val_roc is not None and val_roc > best_val_roc:
                best_val_roc = val_roc
                best_epoch = epoch
                torch.save(
                    model.state_dict(),
                    f"{ROOT_DIR}/data/processed/best_model_{DATASET}.pt",
                )

    if not HAS_LABELS:
        torch.save(
            model.state_dict(), f"{ROOT_DIR}/data/processed/best_model_{DATASET}.pt"
        )

    model.load_state_dict(
        torch.load(
            f"{ROOT_DIR}/data/processed/best_model_{DATASET}.pt", weights_only=True
        )
    )
    centroid = calc_regular_centroid(neighbor_train)

    # alpha and gamma are selection choices, so both ablations run on validation.
    # alpha only reweights existing scores (no retraining needed)
    if HAS_LABELS:
        alpha_iter = []
        for a in [0.0, 0.25, 0.5, 0.65, 0.75, 0.9, 1.0]:
            test_metrics_a, df_test_a, z_test_a, y_test_a = evaluate(
                neighbor_val, centroid, best_epoch, "val", alpha=a, gamma=GAMMA
            )
            alpha_iter.append({"alpha": a, **test_metrics_a})
        df_alpha = pd.DataFrame(alpha_iter)
        df_alpha.to_csv(f"{OUTPUT_DIR}/ablation_alpha_{DATASET}.csv", index=False)

    test_metrics, df_test, z_test, y_test = evaluate(
        neighbor_test, centroid, best_epoch, "test", alpha=ALPHA, gamma=GAMMA
    )
    all_rows.append(df_test)

    if HAS_LABELS:
        print(
            f"\n best epoch {best_epoch}; test ROCAUC {test_metrics['roc_auc']:.4f} "
            f"; P@100 {test_metrics['p100']:.4f}; R@100 {test_metrics['r100']:.4f}"
        )
    else:
        print(f"\n {DATASET}: test set scored, {len(df_test)} entries")

    df_performance = pd.concat(all_rows, ignore_index=True)

    df_best_test = (
        df_performance[df_performance["split"] == "test"]
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
        .copy()
    )
    df_best_test["rank"] = df_best_test.index + 1

    df_best_test.to_csv(
        f"{OUTPUT_DIR}/anomaly_ranking_BEST_EPOCH_{DATASET}.csv", index=False
    )

    if HAS_LABELS:
        label_map = {0: "regular", 1: "local", 2: "global"}
        top100 = df_best_test.head(100).copy()
        top100["label_name"] = top100["label"].map(label_map)
        print("top100 labels:")
        print(top100["label_name"].value_counts())
        print("\nanomalies top100:")
        print(top100[top100["label"] != 0][["entry_id", "label_name", "score", "rank"]])
    else:
        print(f"\ntop-10 highest scoring journal entries ({DATASET}):")
        print(df_best_test.head(10)[["entry_id", "score", "re_norm", "rank"]])

    if HAS_LABELS:
        plot_latent_space(
            z_test,
            y_test,
            save_path=f"{OUTPUT_DIR}/latent_space_test_{DATASET}.png",
            title="GNN latent space - Schreyer (t-SNE)",
        )
    else:
        plot_latent_space(
            z_test,
            y_test,
            save_path=f"{OUTPUT_DIR}/latent_space_test_{DATASET}.png",
            scores=df_test["score"].to_numpy(),
            title="GNN latent space - Philadelphia (t-SNE)",
        )

    # gamma ablation is slow on CPU, run separately when needed
    if HAS_LABELS:
        gamma_ablation(gamma_sweep=[0.0, 0.25, 0.5, 0.65, 0.75, 0.9, 1.0], epochs=10)

    sep_scores = df_best_test["score"].to_numpy()
    sep_series = pd.Series(sep_scores)

    score_max = float(sep_series.max())
    score_p99 = float(sep_series.quantile(0.99))
    score_median = float(sep_series.quantile(0.50))

    separation = {
        "max": score_max,
        "p99": score_p99,
        "median": score_median,
        "max_over_median": score_max / (score_median + 1e-12),
        "p99_over_median": score_p99 / (score_median + 1e-12),
    }

    print("\nscore separation:")
    for key, value in separation.items():
        print(f"{key}:{value:.4f}")

    pd.DataFrame([separation]).to_csv(
        f"{OUTPUT_DIR}/score_separation_{DATASET}.csv", index=False
    )

    if not HAS_LABELS:
        raw = pd.read_csv(f"{ROOT_DIR}/data/processed/city_payments_processed.csv")
        top_ids = df_best_test.head(20)["entry_id"].astype(int).tolist()

        # entry_id is the row index in the sorted processed CSV
        top_raw = raw.iloc[top_ids].copy()
        top_raw["score"] = df_best_test.head(20)["score"].values
        top_raw["rank"] = range(1, len(top_raw) + 1)
        top_raw.to_csv(f"{OUTPUT_DIR}/top20_inspection_{DATASET}.csv", index=False)

    audit_table = build_audit_table(model, graph, df_best_test, GAMMA, top_n=20)

    if not HAS_LABELS:
        raw = pd.read_csv(f"{ROOT_DIR}/data/processed/city_payments_processed.csv")
        audit_table["vendor_name"] = raw.iloc[audit_table["entry_id"]][
            "vendor_name"
        ].values
        audit_table["department"] = raw.iloc[audit_table["entry_id"]][
            "department_title"
        ].values
        audit_table["sub_obj_title"] = raw.iloc[audit_table["entry_id"]][
            "sub_obj_title"
        ].values

    audit_table.to_csv(f"{OUTPUT_DIR}/audit_table_{DATASET}.csv", index=False)
    print(f"\naudit table saved: audit_table_{DATASET}.csv")

    plot_score_distribution(
        scores=df_best_test["score"].to_numpy(),
        save_path=f"{OUTPUT_DIR}/score_distribution_{DATASET}.png",
        title=f"Anomaly score distribution - {DATASET}",
    )

    scores_without_top = df_best_test["score"].to_numpy()[1:]

    plot_score_distribution(
        scores=scores_without_top,
        save_path=f"{OUTPUT_DIR}/score_distribution_{DATASET}_no_top1.png",
        title=f"Anomaly score distribution (excl. top 1 entry) - {DATASET}",
    )

    df_gnn_common = None
    if HAS_LABELS:
        _, df_gnn_common, _, _ = evaluate(
            neighbor_common, centroid, best_epoch, "common", alpha=ALPHA, gamma=GAMMA
        )
        df_gnn_common.to_csv(
            f"{OUTPUT_DIR}/gnn_common_10050_{DATASET}.csv", index=False
        )

    return df_best_test, test_metrics, df_gnn_common


if __name__ == "__main__":
    run_gnn(dataset="schreyer", has_labels=True, seed=42, epochs=20)


# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/bipartite_sage_unsup.py#L173
# https://www.kaggle.com/code/rayanaay/graph-neural-network-graphsage-sample-agregate
