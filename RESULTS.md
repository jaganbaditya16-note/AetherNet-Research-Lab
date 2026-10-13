# AetherNet Research Lab 0.3.0 — preliminary benchmark record

## Run setup

- Version: 0.3.0. One repeat per dataset, random seed 121, maximum 300 rows per dataset, stratified 80/20 train/test split.
- Seven datasets: four synthetic tasks and three bundled scikit-learn datasets. Wine uses all 178 available rows because it is smaller than the requested cap.
- Ten model approaches. Results were generated in the development container used to prepare this update—not on the user's Windows laptop. Timings are not laptop performance claims.
- This is a smoke / exploratory run with one split, not statistically reliable evidence of superiority.

## Single-split accuracy snapshot

Percent accuracy on the held-out portion of this one split. Do not treat this table as a final ranking.

| Dataset | Strongest observed model(s) in this run | Accuracy | AetherGrow | AetherCascade |
|---|---|---:|---:|---:|
| moons | AetherNet gated expert, AetherCascade | 96.7% | 93.3% | 96.7% |
| circles | Extreme Learning Machine, Random forest, AetherNet gated expert | 93.3% | 91.7% | 91.7% |
| mixed_tabular | Random forest | 80.0% | 71.7% | 75.0% |
| blobs | Several methods (MLP, ELM, RBF, KNN, Random forest and candidate methods) | 100.0% | 100.0% | 100.0% |
| breast_cancer | MLP, Nearest centroid, AetherCascade | 98.3% | 96.7% | 98.3% |
| wine | Extreme Learning Machine, RBF network, AetherGrow | 100.0% | 100.0% | 88.9% |
| digits | Random forest, AetherGrow | 91.7% | 91.7% | 78.3% |

## Interpretation

- There is no universal winner in this run. AetherGrow tied the best observed score on Wine and the 300-row Digits subset, but underperformed other methods on Mixed Tabular and Circles. AetherCascade varied considerably by dataset.
- Digits was capped at 300 rows and is not a full-dataset digits benchmark. Wine has only 178 total rows.
- Fit/predict timing depends on machine load and hardware. Serialized model size is an artifact-size proxy, not peak memory. Route frequency is not a measurement of energy, FLOPs, or end-to-end savings.
- The Breast Cancer Wisconsin dataset is included solely as a standard classification benchmark, not for diagnosis or medical decisions.
- These are known model families and a prototype composition. This run does not establish a new architecture or a novelty claim.

## Reproduce

```powershell
.\\.venv\\Scripts\\python.exe benchmarks_v2.py --repeats 3 --samples 500 --output results_v3.json
```

For a short CI smoke run:

```powershell
python benchmarks_v2.py --repeats 1 --samples 300 --dataset wine --output ci-results.json
```

For research-grade conclusions, use repeated/nested validation, more public datasets, per-seed results, matched latency/storage budgets, ablations, and independent replication.
