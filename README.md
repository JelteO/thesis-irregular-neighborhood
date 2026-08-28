# The Irregular Neighborhood

Code for my master's thesis at LIACS, Leiden University: _An Explainable Graph Neural Network Pipeline for Anomaly Detection in ERP Journal Entry Data_.

Jelte Oldenhof, MSc Computer Science (Data Science), LIACS, Leiden University, 2026.
Supervised by Dr. Akrati Saxena and Prof. Frank Takes.

The pipeline turns tabular journal entries into a heterogeneous graph, trains a graph autoencoder on regular entries only, and uses the reconstruction error as an anomaly score. An explanation layer then reports per flagged entry how much of the score comes from its neighbors and which of its own fields carries the error.

The thesis itself is the documentation. This README only covers how to run the code.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12. Everything runs on CPU, no GPU needed.

## Running

```bash
python main.py
```

This runs the whole thing in order: preprocessing, graph construction, GNN training and explanation for both datasets, then the four baselines and the comparison table. Preprocessing and graph construction are skipped when their output already exists, so a second run is faster.

Everything is fixed on seed 42, so a rerun gives the same numbers.

## Data

Both datasets are included in `data/raw/` as unmodified copies of their public sources:

- `fraud_dataset_v2.csv` — the synthetic SAP benchmark from Schreyer et al., 533,009 entries with 100 injected anomalies. Source: https://github.com/GitiHubi/deepAD
- `city_payments_fy2017.csv` — City of Philadelphia vendor payments FY2017, 238,894 rows, no labels. Source: https://opendataphilly.org/datasets/city-payments/

Processed files and graphs land in `data/processed/` and are not tracked.

## Structure

```
main.py                              runs everything
src/data/
  preprocessing.py                   raw csv to features (primary)
  preprocessing_phil.py              raw csv to features (secondary)
  graph_construction.py              hetero graph + homo graphs for DOMINANT
  graph_construction_phil.py         hetero graph for the secondary dataset
  split.py                           train/val/test split, single source of truth
src/models/
  gnn_model.py                       encoder, decoder, training, explanation layer
  schreyer_ae.py                     adversarial autoencoder baseline
  baselines.py                       PCA and Isolation Forest
  seed.py                            fixes all randomness
src/evaluation/
  experiment.py                      runs each baseline on the same split
  comparison.py                      metrics and the comparison table
  fidelity.py                        aggregates the occlusion drops
  label_dist.py                      label counts in the top-k, labelled score plot
  rarity_check.py                    ground truth check on the feature attribution
```

## Outputs

Everything is written to `outputs/`. The plots are tracked in the repo, the csv files are not, so they are regenerated on a run:

- `comparison_common_10050.csv` — the model comparison table
- `anomaly_ranking_*.csv` — full ranking per model
- `audit_table_*.csv` — per flagged entry the two occlusion drops and the top field
- `fidelity_*.csv` — the occlusion drops aggregated over the top 20
- `rarity_check_schreyer.csv` — how often the blamed value occurs in the data
- `label_distribution_top_k.csv` — regular/local/global counts in the top-k
- the score distribution and t-SNE plots as png

## Note on runtime

The GNN trains in ten epochs and is fine on a laptop. DOMINANT is the slow part, it builds a dense adjacency and is therefore fitted on a 10,000 entry sample rather than the full training set. That is also why the model comparison runs on a common subset of 10,050 entries instead of the full test split.

## Contact

Questions or feedback: open an issue, or reach out on
[LinkedIn](https://www.linkedin.com/in/jelte-oldenhof/).

## Reference

The thesis is available in the Leiden University student repository:
https://theses.liacs.nl

The primary dataset and the adversarial autoencoder baseline come from the work
of Schreyer et al., "Detection of Accounting Anomalies in the Latent Space using
Adversarial Autoencoder Neural Networks": https://arxiv.org/abs/1908.00734
