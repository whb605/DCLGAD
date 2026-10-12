# DCLGAD

**DCLGAD: Dual Curriculum Learning Training Guided Graph Autoencoder for Anomaly Detection**.



## Installation

```bash
pip install -r requirements.txt
```

`pygod` is optional and only imported inside the branch that loads pygod-built datasets
(`weibo`, `reddit`, `disney`, `books`, `inj_cora`); the `.mat` path works without it.

## Usage

```bash
python main.py --dataset YelpRes --device cuda
```

| Argument | Default | Meaning |
|---|---|---|
| `--dataset` | `YelpRes` | dataset name, resolved to `./dataset/<name>.mat` |
| `--hidden_dim` | `128` | hidden size |
| `--epoch` | `100` | epochs per stage |
| `--lr` | `5e-3` | learning rate |
| `--dropout` | `0.3` | dropout |
| `--alpha` | `0.5` | attribute / structure loss balance (Eq. 18) |
| `--ratio` | `0.25` | curriculum ratio $\beta$ (Eq. 7) |
| `--lambda_param` | `1.0` | pacing exponent $\lambda$: 0.5 root, 1 linear, 2 quadratic |
| `--seed` | `10` | random seed |
| `--device` | `cuda` | `cuda` / `cpu` |


## Data

Each `.mat` file provides `Network` (adjacency), `Attributes` (node features) and `Label`
(1 = anomaly). Only `YelpRes.mat` is bundled here; the remaining datasets follow the same format.

## Notes

- `./similarity_matrix/` caches the 1-hop and 2-hop similarity matrices per dataset and is
  regenerated automatically; delete it after changing the graph.
- The reported numbers are averaged over multiple random seeds; a single run will vary.

