from __future__ import annotations
import json
import time
import pickle
import warnings
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from sklearn.datasets import (
    make_moons, make_circles, make_classification, make_blobs,
    load_breast_cancer, load_wine, load_digits,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.exceptions import ConvergenceWarning

from architectures import build_model_suite
from aethernet import AetherNet
from adaptive_cascade import AetherCascade
from adaptive_grow import AetherGrow

DATASETS = (
    "moons", "circles", "mixed_tabular", "blobs",
    "breast_cancer", "wine", "digits",
)
REAL_DATASETS = ("breast_cancer", "wine", "digits")
DATASET_INFO = {
    "moons": {"kind": "synthetic", "source": "scikit-learn make_moons", "description": "Two interleaving nonlinear classes."},
    "circles": {"kind": "synthetic", "source": "scikit-learn make_circles", "description": "Concentric nonlinear classes."},
    "mixed_tabular": {"kind": "synthetic", "source": "scikit-learn make_classification", "description": "Generated tabular classification with informative and redundant features."},
    "blobs": {"kind": "synthetic", "source": "scikit-learn make_blobs", "description": "Generated clustered samples converted to a binary task."},
    "breast_cancer": {"kind": "bundled-real", "source": "scikit-learn load_breast_cancer (Wisconsin Diagnostic dataset)", "description": "569 samples, 30 numeric features, binary classification. Research benchmark only; not for medical decisions."},
    "wine": {"kind": "bundled-real", "source": "scikit-learn load_wine (UCI Wine recognition dataset)", "description": "178 samples, 13 numeric features, three classes."},
    "digits": {"kind": "bundled-real", "source": "scikit-learn load_digits", "description": "1,797 small 8x8 handwritten-digit images represented by 64 pixel features, ten classes."},
}


def make_dataset(name: str, seed: int, n_samples: int = 900):
    """Load or generate a dataset; for bundled datasets n_samples is a maximum.

    Real datasets ship with scikit-learn, so these benchmark paths do not need to
    download data. If a source is larger than the requested cap, use a deterministic
    stratified subset. Small sources (e.g. Wine with 178 samples) are used in full.
    """
    if name == "moons":
        return make_moons(n_samples=n_samples, noise=0.25, random_state=seed)
    if name == "circles":
        return make_circles(n_samples=n_samples, noise=0.20, factor=0.42, random_state=seed)
    if name == "mixed_tabular":
        return make_classification(n_samples=n_samples, n_features=12, n_informative=6,
            n_redundant=3, n_repeated=0, n_classes=2, n_clusters_per_class=2,
            class_sep=0.9, flip_y=0.07, random_state=seed)
    if name == "blobs":
        X, y = make_blobs(n_samples=n_samples, centers=4, n_features=8,
                          cluster_std=2.2, random_state=seed)
        return X, (y >= 2).astype(int)
    loaders = {
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
        "digits": load_digits,
    }
    if name in loaders:
        X, y = loaders[name](return_X_y=True)
        if n_samples < len(y):
            X, _, y, _ = train_test_split(
                X, y, train_size=n_samples, random_state=seed, stratify=y
            )
        return np.asarray(X, dtype=float), np.asarray(y)
    raise ValueError(f"Unknown dataset: {name}")


def _architecture_detail(model: Any, name: str) -> str:
    if getattr(model, "selected_hidden_units_", None) is not None:
        return f"selected width={model.selected_hidden_units_}"
    if name == "AetherCascade" and getattr(model, "mlp_threshold_", None) is not None:
        return f"gate MLP<{model.mlp_threshold_:.3f}; ELM<{model.elm_threshold_:.3f}"
    if name == "Extreme Learning Machine":
        return f"fixed width={model.hidden_units}"
    if name == "Hyperdimensional prototype":
        return f"dimensions={model.dimensions}"
    if name == "RBF network":
        return f"centers={model.centers}"
    return "—"


def _measure_model(name: str, family: str, model: Any, X_train: np.ndarray,
                   y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    activation_rate = None
    elm_call_rate = None
    knn_call_rate = None
    start = time.perf_counter()
    if name == "AetherNet gated expert":
        report = model.predict_report(X_test)
        predictions = report.predictions
        activation_rate = report.activation_rate
    elif name == "AetherCascade":
        report = model.predict_report(X_test)
        predictions = report.predictions
        elm_call_rate = report.elm_call_rate
        knn_call_rate = report.knn_call_rate
        activation_rate = report.elm_call_rate + report.knn_call_rate
    else:
        predictions = model.predict(X_test)
    predict_seconds = time.perf_counter() - start
    try:
        serialized_kb = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)) / 1024.0
    except Exception:
        serialized_kb = None
    return {
        "model": name,
        "family": family,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "serialized_model_kb": serialized_kb,
        "specialist_call_rate": activation_rate,
        "elm_call_rate": elm_call_rate,
        "knn_call_rate": knn_call_rate,
        "architecture_detail": _architecture_detail(model, name),
    }


def run_suite(repeats: int = 1, n_samples: int = 900, dataset_names: list[str] | None = None) -> dict[str, Any]:
    if repeats < 1 or repeats > 20:
        raise ValueError("repeats must be between 1 and 20")
    if n_samples < 200 or n_samples > 10000:
        raise ValueError("samples must be between 200 and 10000")
    selected_datasets = dataset_names or list(DATASETS)
    unknown = set(selected_datasets) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unsupported datasets: {sorted(unknown)}")
    results: dict[str, Any] = {
        "project": "AetherNet Research Lab",
        "version": "0.3.0",
        "status": "experimental; benchmarks run locally on the host computer",
        "warning": "These are established model families and experimental combinations. This does not establish first-ever novelty or superiority.",
        "metrics_note": "Fit and predict time are wall-clock timings on this run and can vary by machine. Serialized model size is a storage proxy, not peak RAM. No energy measurement is made. Bundled real datasets may contain fewer rows than the requested maximum sample count.",
        "protocol": "Each repeat uses a stratified held-out test split. Scaling is fit on the training split only. Candidate selection inside AetherGrow/AetherCascade uses internal training/calibration splits; test labels are not used to choose their settings.",
        "repeats": repeats,
        "max_samples_per_dataset": n_samples,
        "datasets": {},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for dataset_name in selected_datasets:
            trials = []
            for repeat in range(repeats):
                seed = 121 + repeat * 31
                X, y = make_dataset(dataset_name, seed, n_samples)
                used_samples = int(len(y))
                feature_count = int(X.shape[1])
                class_count = int(len(np.unique(y)))
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.20, random_state=seed + 3, stratify=y
                )
                scaler = StandardScaler().fit(X_tr)
                X_tr_s = scaler.transform(X_tr)
                X_te_s = scaler.transform(X_te)
                rows = []
                for name, family, model in build_model_suite(seed):
                    rows.append(_measure_model(name, family, model, X_tr_s, y_tr, X_te_s, y_te))
                # Existing AetherNet hybrid; its own fit method creates its core/calibration split.
                hybrid = AetherNet(random_state=seed, n_neighbors=11, compute_penalty=0.025, max_iter=160)
                rows.append(_measure_model("AetherNet gated expert", "Adaptive hybrid", hybrid,
                                           X_tr_s, y_tr, X_te_s, y_te))
                grow = AetherGrow(random_state=seed, tolerance=0.01)
                rows.append(_measure_model("AetherGrow", "Adaptive-width ELM candidate", grow,
                                           X_tr_s, y_tr, X_te_s, y_te))
                cascade = AetherCascade(random_state=seed, compute_penalty=0.025,
                                       mlp_max_iter=120, elm_hidden_units=128, n_neighbors=11)
                rows.append(_measure_model("AetherCascade", "Three-stage adaptive cascade", cascade,
                                           X_tr_s, y_tr, X_te_s, y_te))
                trials.append({
                    "seed": seed,
                    "n_samples_used": used_samples,
                    "n_features": feature_count,
                    "n_classes": class_count,
                    "models": rows,
                })
            aggregate: dict[str, Any] = {}
            model_names = [name for name, _, _ in build_model_suite(121)] + ["AetherNet gated expert", "AetherGrow", "AetherCascade"]
            for model_name in model_names:
                per_trial = [next(m for m in t["models"] if m["model"] == model_name) for t in trials]
                metric_names = ("accuracy", "balanced_accuracy", "macro_f1", "fit_seconds", "predict_seconds", "serialized_model_kb")
                aggregate[model_name] = {
                    metric: {"mean": float(mean([float(r[metric]) for r in per_trial if r[metric] is not None])),
                             "std": float(pstdev([float(r[metric]) for r in per_trial if r[metric] is not None]))}
                    for metric in metric_names
                }
                for rate_field in ("specialist_call_rate", "elm_call_rate", "knn_call_rate"):
                    rates = [r[rate_field] for r in per_trial if r.get(rate_field) is not None]
                    if rates:
                        aggregate[model_name][rate_field] = {"mean": float(mean(rates)), "std": float(pstdev(rates))}
                aggregate[model_name]["family"] = per_trial[0]["family"]
            # Include factual source metadata in exported JSON so a result remains interpretable.
            X_meta, y_meta = make_dataset(dataset_name, seed=121, n_samples=n_samples)
            meta = dict(DATASET_INFO[dataset_name])
            meta.update({
                "max_samples_requested": n_samples,
                "samples_used": int(len(y_meta)),
                "features": int(X_meta.shape[1]),
                "classes": int(len(np.unique(y_meta))),
            })
            results["datasets"][dataset_name] = {
                "metadata": meta,
                "trials": trials,
                "summary": aggregate,
            }
    return results


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Compare lightweight local architectures for AetherNet")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--samples", type=int, default=900)
    parser.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    parser.add_argument("--output", type=Path, default=Path("results_v2.json"))
    args = parser.parse_args()
    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    results = run_suite(args.repeats, args.samples, datasets)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"AetherNet Research Lab {results['version']} · {args.repeats} repeat(s) · max {args.samples} samples/dataset")
    print("dataset         model                         accuracy  fit(s) predict(s) size(KB)  specialist")
    print("-" * 104)
    for dataset, block in results["datasets"].items():
        # Use the first repeat's detailed rows for compact, concrete timing output.
        for row in block["trials"][0]["models"]:
            if row.get("elm_call_rate") is not None:
                call = f"E {row['elm_call_rate']:.1%}/K {row['knn_call_rate']:.1%}"
            else:
                call = "—" if row["specialist_call_rate"] is None else f"{row['specialist_call_rate']:.1%}"
            size = "n/a" if row["serialized_model_kb"] is None else f"{row['serialized_model_kb']:.1f}"
            print(f"{dataset:15s} {row['model'][:28]:28s} {row['accuracy']:7.1%}  {row['fit_seconds']:6.3f} {row['predict_seconds']:9.4f} {size:>8s} {call:>11s}")
    print(f"\nSaved full results: {args.output.resolve()}")
    print("Reminder: timings are machine-dependent; serialized size is not RAM; no energy is measured. See dataset metadata for source and actual sample count.")

if __name__ == "__main__":
    main()
