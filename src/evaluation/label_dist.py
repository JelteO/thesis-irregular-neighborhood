# label_distribution.py - label counts in top-k rankings + labelled score plot
# in: score dataframes per model (common set) and the full test ranking csv
# out: outputs/label_distribution_top_k.csv and a labelled score distribution plot
# ---------------------------------------------------------------

import pandas as pd

import matplotlib.pyplot as plt

LABEL_NAMES = {0: "regular", 1: "local", 2: "global"}
LABEL_COLORS = {"regular": "tab:blue", "local": "tab:orange", "global": "tab:green"}


def label_distribution(scores_by_model, ks=(50, 100)):
    # counts regular/local/global in the top-k of every model's ranking
    rows = []
    for model_name, df in scores_by_model.items():
        ranked = df.sort_values("score", ascending=False).reset_index(drop=True)
        for k in ks:
            top = ranked.head(k)
            counts = top["label"].map(LABEL_NAMES).value_counts()
            rows.append(
                {
                    "model": model_name,
                    "k": k,
                    "regular": int(counts.get("regular", 0)),
                    "local": int(counts.get("local", 0)),
                    "global": int(counts.get("global", 0)),
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv("outputs/label_distribution_top_k.csv", index=False)
    print("\nlabel distribution per model:")
    print(table.to_string(index=False))
    return table


def plot_labelled_score_distribution(
    ranking_csv="outputs/anomaly_ranking_BEST_EPOCH_schreyer.csv",
    save_path="outputs/score_distribution_labelled_schreyer.png",
):
    # scatter in the style of the Schreyer notebook: x = entry index
    # (no meaning, just spreads the points), y = anomaly score on a log scale
    df = pd.read_csv(ranking_csv)
    df["label_name"] = df["label"].map(LABEL_NAMES)

    # scores of exactly 0 cannot be shown on a log axis, clip to a small floor
    floor = 1e-8
    df["score_plot"] = df["score"].clip(lower=floor)

    # shuffle so the x position carries no information, same as the notebook
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    regular = df[df["label_name"] == "regular"]
    local = df[df["label_name"] == "local"]
    global_ = df[df["label_name"] == "global"]

    plt.figure(figsize=(10, 6))

    plt.scatter(
        regular.index,
        regular["score_plot"],
        c=LABEL_COLORS["regular"],
        marker="o",
        s=30,
        linewidth=0.3,
        edgecolors="w",
        label="regular",
    )
    plt.scatter(
        local.index,
        local["score_plot"],
        c=LABEL_COLORS["local"],
        marker="x",
        s=120,
        linewidth=3,
        label="local",
    )
    plt.scatter(
        global_.index,
        global_["score_plot"],
        c=LABEL_COLORS["global"],
        marker="x",
        s=120,
        linewidth=3,
        label="global",
    )

    # threshold at the rank-100 score, drawn as a horizontal line
    threshold = df.sort_values("score", ascending=False)["score"].iloc[99]
    n_anomalies_above = int((df[df["score"] >= threshold]["label"] != 0).sum())
    plt.axhline(threshold, color="black", linestyle="--", linewidth=1)
    plt.text(
        len(df) * 0.01,
        threshold * 1.5,
        f"top 100 ({n_anomalies_above}/50 anomalies)",
        fontsize=9,
        ha="left",
        va="bottom",
        zorder=10,
        bbox=dict(
            boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85
        ),
    )

    anomalies = len(local) + len(global_)
    plt.yscale("log")
    plt.xticks([])
    plt.xlabel("journal entry")
    plt.ylabel("anomaly score (log scale)")
    plt.title(
        f"Anomaly scores per journal entry full test set ({len(regular)} regular, {anomalies} anomalies)"
    )
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=3,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"labelled score distribution plot saved: {save_path}")
