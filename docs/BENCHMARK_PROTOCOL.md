# Benchmark protocol for AetherNet 0.3.0

## Task and model suite

The task is supervised classification. The suite includes an MLP, ELM-style fixed random feature readout, hyperdimensional prototypes, an RBF network, KNN, nearest centroid, random forest, AetherNet's selective KNN hybrid, AetherGrow, and AetherCascade. These are comparison methods; their inclusion is not evidence of novelty.

## Dataset registry

| Key | Kind | Source | Notes |
|---|---|---|---|
| `moons` | Synthetic | `sklearn.datasets.make_moons` | Two nonlinear classes |
| `circles` | Synthetic | `sklearn.datasets.make_circles` | Two concentric classes |
| `mixed_tabular` | Synthetic | `sklearn.datasets.make_classification` | Generated 12-feature binary task |
| `blobs` | Synthetic | `sklearn.datasets.make_blobs` | Four generated clusters mapped to two classes |
| `breast_cancer` | Bundled dataset | `sklearn.datasets.load_breast_cancer` | 569 rows, 30 features, 2 classes; benchmark only, not medical use |
| `wine` | Bundled dataset | `sklearn.datasets.load_wine` | 178 rows, 13 features, 3 classes |
| `digits` | Bundled dataset | `sklearn.datasets.load_digits` | 1,797 rows, 64 pixel features, 10 classes |

Sources and dataset notes: https://scikit-learn.org/stable/datasets.html. Real datasets are bundled with scikit-learn; benchmark code does not fetch data from the internet. `--samples` is a maximum. When a bundled source is smaller than the cap, all its rows are used; when larger, a deterministic stratified subset is used.

## Procedure

1. For each dataset and repeat, generate/load data using a fixed seed.
2. When needed, make a deterministic stratified subset capped at the requested sample count.
3. Split 80% train / 20% held-out test with stratification.
4. Fit the benchmark-level scaler on the training split only.
5. Fit all baseline models on training only. AetherNet variants reserve an internal calibration subset from the training split for their gate/width choice.
6. Freeze fitted choices, then score only on the held-out test split. No test labels select gates or widths.
7. Store individual trial rows and means/standard deviations for accuracy, balanced accuracy, macro-F1, wall-clock fit/predict timing and serialized model size.

## Interpretation

- Accuracy, balanced accuracy and macro-F1 are predictive metrics, not proof of a model's overall intelligence.
- Timings vary by hardware, operating system, dependency version and machine load.
- Pickle size measures serialized artifact size, not peak runtime memory.
- Route percentages count requests sent to a specialist; they are not measured energy, FLOPs, or total latency savings.
- Synthetic and bundled datasets are small benchmarks, not evidence of broad real-world utility.
- Multiple comparisons, small samples and hyperparameter choices can make isolated wins misleading. Follow-up work should use repeated/nested validation, stronger baselines, confidence intervals, ablations, and real hardware measurements.
