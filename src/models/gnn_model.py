from torch_geometric.nn import SAGEConv, to_hetero, Sequential, HeteroConv
import torch
import os, sys
import torch_geometric.transforms as T
from torch.nn import Embedding, Linear, MSELoss, ReLU, CrossEntropyLoss, ModuleDict
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.loader import NeighborLoader
from torch.optim.adam import Adam
from sklearn.metrics import roc_auc_score
from torchmetrics.functional.retrieval import retrieval_precision, retrieval_recall
from torch_geometric.explain import Explainer, GNNExplainer
import pandas as pd
from src.data.graph_construction import create_graphs

pd.options.display.max_columns = 9

device = torch.device("cpu")
root_dir = os.getcwd()

create_graphs()
graph = torch.load(f"{root_dir}/data/processed/graph_hetero.pt", weights_only=False)

EMBEDDING_DIM = 16
HIDDEN_DIM = 64
OUTPUT_DIM = graph["entry"].x.shape[1]
NUM_GL = graph.num_gl_accounts
NUM_PC = graph.num_profit_centers
NUM_NUMERICAL = graph.entry_feature_info["num_numerical"]
CAT_DIMS = graph.entry_feature_info["categorical_dims"]
CAT_ORDER = graph.entry_feature_info["categorical_order"]
GAMMA = 0.65

regular_idx = (graph["entry"].y == 0).nonzero(as_tuple=True)[0]
local_idx = (graph["entry"].y == 1).nonzero(as_tuple=True)[0]  # indices
global_idx = (graph["entry"].y == 2).nonzero(as_tuple=True)[0]  # indices


def split(index, generator, frac_train=0.8, frac_val=0.1):
    index = index[torch.randperm(len(index), generator=generator)]
    n = len(index)
    n_train = int(frac_train * n)
    n_val = int(frac_val * n)
    return index[:n_train], index[n_train : n_train + n_val], index[n_train + n_val :]


gen = torch.Generator().manual_seed(42)
regular_train, regular_val, regular_test = split(regular_idx, gen)
_, local_val, local_test = split(local_idx, gen)
_, global_val, global_test = split(global_idx, gen)

print(f"val set anomaly split: {len(local_val)}(local)/{len(global_val)}(global)")
print(f"test set anomaly split: {len(local_test)}(local)/{len(global_test)}(global)")


train_mask = regular_train

val_mask = torch.cat([regular_val, local_val, global_val])
test_mask = torch.cat([regular_test, local_test, global_test])

val_mask = val_mask[torch.randperm(len(val_mask), generator=gen)]
test_mask = test_mask[torch.randperm(len(test_mask), generator=gen)]


LOADER_KWARGS = dict(
    data=graph,
    num_neighbors=[10] * 2,  # bepaalt het aantal hops (= aantal conv-lagen)
    batch_size=128,
    subgraph_type="bidirectional",
)

neighbor_train = NeighborLoader(
    input_nodes=("entry", train_mask), shuffle=True, **LOADER_KWARGS
)
neighbor_val = NeighborLoader(
    input_nodes=("entry", val_mask), shuffle=False, **LOADER_KWARGS
)
neighbor_test = NeighborLoader(
    input_nodes=("entry", test_mask), shuffle=False, **LOADER_KWARGS
)


def get_prec_recall(predict, target, top_k: int):
    target = torch.as_tensor(target, dtype=torch.bool)
    predict = torch.as_tensor(predict, dtype=torch.float32)
    p = retrieval_precision(preds=predict, target=target, top_k=top_k)
    r = retrieval_recall(preds=predict, target=target, top_k=top_k)
    return p, r


class Encoder(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, embedding_dim, num_pc, num_gl):
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
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        # doel van decoder is terugbrengen van bijv 64 dim naar 2 dim
        self.lin = Linear(hidden_dim, output_dim)

    def forward(self, z):
        return self.lin(z)


class Model(torch.nn.Module):
    def __init__(
        self, embedding_dim, hidden_dim, output_dim, graph_meta, num_pc, num_gl
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encoder = Encoder(
            self.hidden_dim, self.output_dim, self.embedding_dim, num_pc, num_gl
        )
        self.decoder = Decoder(self.hidden_dim, self.output_dim)

    def forward(self, x_dict, edge_index_dict, batch_size):
        z_dict = self.encoder(x_dict, edge_index_dict)
        x_con = self.decoder(z_dict["entry"][:batch_size])
        return x_con


model = Model(
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    graph_meta=graph.metadata(),
    num_pc=NUM_PC,
    num_gl=NUM_GL,
)
loss_func_mean = MSELoss(reduction="mean")
loss_func = MSELoss(reduction="none")


def train():
    model.train()

    total_loss = 0
    total_examples = 0

    for batch in tqdm(neighbor_train):
        optimizer.zero_grad()
        batch = batch.to(device)
        batch_size = batch["entry"].batch_size
        x_pred = model(batch.x_dict, batch.edge_index_dict, batch_size)
        loss = loss_func_mean(x_pred, batch.x_dict["entry"][:batch_size])
        loss.backward()
        optimizer.step()

        total_examples += batch_size
        total_loss += float(loss) * batch_size

    return total_loss / total_examples


@torch.no_grad()  # no gradient adj
def evaluate(loader, epoch=None, split_str=None):
    model.eval()
    pred, targets, entry_ids = [], [], []

    for batch in tqdm(loader):
        batch = batch.to(device)
        batch_size = batch["entry"].batch_size
        x_pred = model(batch.x_dict, batch.edge_index_dict, batch_size)

        loss = loss_func(x_pred, batch.x_dict["entry"][:batch_size])
        score = loss.mean(dim=1)

        pred.append(score)
        target = batch["entry"].y[:batch_size]
        entry_id = batch["entry"].n_id[:batch_size]
        targets.append(target)
        entry_ids.append(entry_id)

    all_preds = torch.cat(pred, dim=0).numpy()
    all_targets = torch.cat(targets, dim=0).numpy()
    all_ids = torch.cat(entry_ids, dim=0).numpy()
    all_bin = (all_targets != 0).astype(int)

    p100, r100 = get_prec_recall(predict=all_preds, target=all_bin, top_k=100)
    roc = roc_auc_score(all_bin, all_preds)

    rank = (
        pd.Series(all_preds)
        .rank(method="dense", ascending=False)
        .astype(int)
        .to_numpy()
    )
    df_epoch = pd.DataFrame(
        {
            "entry_id": all_ids,
            "label": all_targets,
            "label_bin": all_bin,
            "score": all_preds,
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
    return metrics, df_epoch


EPOCHS = 10
optimizer = Adam(model.parameters(), lr=0.0001)
df_performance = pd.DataFrame(
    columns=[
        "entry_id",
        "label",
        "label_bin",
        "score",
        "model",
        "epoch",
        "rank",
        "split",
        "roc_auc_scores",
    ]
)

BEST_VAL_AUC = 0
BEST_EPOCH = 0
PATIENCE = 3
EPOCH_NO_IMPROVE = 0
all_rows = []

create_graphs()

for epoch in range(1, EPOCHS + 1):
    print(f"Epoch: {epoch:02d}")
    loss = train()
    val_metrics, df_val = evaluate(neighbor_val, epoch, "val")
    all_rows.append(df_val)
    print(f"Epoch {epoch:02d} | Loss {loss:.4f} | Val ROC {val_metrics['roc_auc']:.4f}")

    if val_metrics["roc_auc"] > BEST_VAL_AUC:
        BEST_VAL_AUC = val_metrics["roc_auc"]
        BEST_EPOCH = epoch
        EPOCH_NO_IMPROVE = 0
        torch.save(model.state_dict(), f"{root_dir}/data/processed/best_model.pt")
    else:
        EPOCH_NO_IMPROVE += 1

    if EPOCH_NO_IMPROVE >= PATIENCE:
        print(f"early stopped at epoch: {epoch}")
        break


model.load_state_dict(
    torch.load(f"{root_dir}/data/processed/best_model.pt", weights_only=True)
)
test_metrics, df_test = evaluate(neighbor_test, BEST_EPOCH, "test")
all_rows.append(df_test)
print(
    f"\n best epoch {BEST_EPOCH}; test ROCAUC {test_metrics['roc_auc']:.4f} "
    f"; P@100 {test_metrics['p100']:.4f}; R@100 {test_metrics['r100']:.4f}"
)

df_performance = pd.concat(all_rows, ignore_index=True)


df_best_test = (
    df_performance[df_performance["split"] == "test"]
    .sort_values("score", ascending=False)
    .reset_index(drop=True)
    .copy()
)
df_best_test["rank"] = df_best_test.index + 1

label_map = {0: "regular", 1: "local", 2: "global"}
top100 = df_best_test.head(100).copy()
top100["label_name"] = top100["label"].map(label_map)


print(f"best epoch: {BEST_EPOCH} | test ROC-AUC: {test_metrics['roc_auc']:.4f}\n")
print("top100 labels:")
print(top100["label_name"].value_counts())
print("\nanomalieën top100:")
print(top100[top100["label"] != 0][["entry_id", "label_name", "score", "rank"]])

df_best_test.to_csv(f"{root_dir}/anomaly_ranking_BEST_EPOCH.csv", index=False)

# https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/bipartite_sage_unsup.py#L173
# https://www.kaggle.com/code/rayanaay/graph-neural-network-graphsage-sample-agregate
# https://pytorch-geometric.readthedocs.io/en/latest/tutorial/explain.html
