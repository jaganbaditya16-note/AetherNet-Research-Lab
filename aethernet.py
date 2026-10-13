"""AetherNet research candidate: budgeted neural core + selective local expert.

This is an experimental hybrid, not a claim of a proven novel neural-network family.
The point is to measure whether a validation-tuned gate can trade accuracy against
extra local inference work on small classification tasks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


@dataclass
class PredictionReport:
    predictions: np.ndarray
    local_expert_activated: np.ndarray
    confidence: np.ndarray

    @property
    def activation_rate(self) -> float:
        return float(np.mean(self.local_expert_activated)) if len(self.local_expert_activated) else 0.0


class AetherNet:
    """A small neural core with a selectively activated nearest-neighbor expert.

    Training protocol:
      1. Split provided training data into a core-fit set and a gate-calibration set.
      2. Fit a small MLP and a local KNN expert on only the core-fit set.
      3. Choose a confidence threshold on calibration data to maximize
         validation_accuracy - compute_penalty * local_expert_activation_rate.
      4. At inference, run KNN only for inputs below the selected confidence threshold.

    No test labels are used to choose the threshold. The KNN branch is a known
    technique; the research question is whether this particular selective gate
    gives a better accuracy/extra-compute trade-off on defined workloads.
    """

    def __init__(
        self,
        random_state: int = 7,
        n_neighbors: int = 11,
        compute_penalty: float = 0.025,
        hidden_layer_sizes: tuple[int, ...] = (32, 16),
        max_iter: int = 220,
    ) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        if compute_penalty < 0:
            raise ValueError("compute_penalty must be >= 0")
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.compute_penalty = compute_penalty
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.scaler: StandardScaler | None = None
        self.neural_core: MLPClassifier | None = None
        self.local_expert: KNeighborsClassifier | None = None
        self.threshold: float = 0.0
        self.classes_: np.ndarray | None = None
        self.validation_score: float | None = None
        self.validation_accuracy: float | None = None
        self.validation_activation_rate: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AetherNet":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2D and y must be 1D with matching sample counts")
        if len(X) < 40:
            raise ValueError("Need at least 40 samples to fit and calibrate the gate")
        if len(np.unique(y)) < 2:
            raise ValueError("At least two classes are required")

        X_core, X_cal, y_core, y_cal = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=self.random_state,
            stratify=y,
        )
        self.scaler = StandardScaler().fit(X_core)
        X_core_s = self.scaler.transform(X_core)
        X_cal_s = self.scaler.transform(X_cal)
        self.classes_ = np.unique(y_core)

        self.neural_core = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.004,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=self.random_state,
        )
        self.neural_core.fit(X_core_s, y_core)

        k = min(self.n_neighbors, len(X_core))
        self.local_expert = KNeighborsClassifier(n_neighbors=k, weights="distance")
        self.local_expert.fit(X_core_s, y_core)

        core_probs = self.neural_core.predict_proba(X_cal_s)
        core_confidence = np.max(core_probs, axis=1)
        core_pred = self.classes_[np.argmax(core_probs, axis=1)]
        local_pred = self.local_expert.predict(X_cal_s)

        # The calibration stage evaluates a finite set of gates without touching test labels.
        quantiles = np.quantile(core_confidence, np.linspace(0.0, 1.0, 21))
        thresholds = np.unique(np.concatenate(([0.0], quantiles, [1.000001])))
        best_objective = -float("inf")
        best_threshold = 0.0
        best_accuracy = float(accuracy_score(y_cal, core_pred))
        best_rate = 0.0
        for threshold in thresholds:
            use_local = core_confidence < threshold
            predictions = core_pred.copy()
            predictions[use_local] = local_pred[use_local]
            accuracy = float(accuracy_score(y_cal, predictions))
            activation_rate = float(np.mean(use_local))
            objective = accuracy - self.compute_penalty * activation_rate
            # Deterministic tie-break: prefer fewer local-expert calls.
            if objective > best_objective + 1e-12 or (
                abs(objective - best_objective) <= 1e-12 and activation_rate < best_rate
            ):
                best_objective = objective
                best_threshold = float(threshold)
                best_accuracy = accuracy
                best_rate = activation_rate

        self.threshold = best_threshold
        self.validation_score = float(best_objective)
        self.validation_accuracy = best_accuracy
        self.validation_activation_rate = best_rate
        return self

    def _check_fitted(self) -> None:
        if self.scaler is None or self.neural_core is None or self.local_expert is None:
            raise RuntimeError("Call fit(X, y) before prediction")

    def predict_report(self, X: np.ndarray) -> PredictionReport:
        self._check_fitted()
        assert self.scaler is not None
        assert self.neural_core is not None
        assert self.local_expert is not None
        X_s = self.scaler.transform(np.asarray(X, dtype=float))
        probs = self.neural_core.predict_proba(X_s)
        confidence = np.max(probs, axis=1)
        assert self.classes_ is not None
        predictions = self.classes_[np.argmax(probs, axis=1)]
        use_local = confidence < self.threshold
        # Critical: execute local retrieval only on gated samples.
        if np.any(use_local):
            predictions[use_local] = self.local_expert.predict(X_s[use_local])
        return PredictionReport(predictions, use_local, confidence)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_report(X).predictions


def evaluate_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                    seed: int, compute_penalty: float = 0.025) -> dict[str, Any]:
    """Fit shared baseline and proposed models; return test metrics."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test)

    X_core, X_cal, y_core, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=seed, stratify=y_train
    )
    scaler = StandardScaler().fit(X_core)
    X_core_s = scaler.transform(X_core)
    X_cal_s = scaler.transform(X_cal)
    X_test_s = scaler.transform(X_test)
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
                        alpha=0.001, learning_rate_init=0.004, max_iter=220,
                        early_stopping=True, validation_fraction=0.15,
                        n_iter_no_change=15, random_state=seed)
    mlp.fit(X_core_s, y_core)
    knn = KNeighborsClassifier(n_neighbors=min(11, len(X_core)), weights="distance")
    knn.fit(X_core_s, y_core)

    # Baseline 1: small neural network only.
    neural_test_pred = mlp.predict(X_test_s)
    neural_acc = float(accuracy_score(y_test, neural_test_pred))

    # Baseline 2: local KNN on every test query.
    knn_test_pred = knn.predict(X_test_s)
    knn_acc = float(accuracy_score(y_test, knn_test_pred))

    # Proposed candidate: calibrate only on the held-out calibration split.
    cal_probs = mlp.predict_proba(X_cal_s)
    cal_conf = np.max(cal_probs, axis=1)
    cal_base = mlp.predict(X_cal_s)
    cal_local = knn.predict(X_cal_s)
    thresholds = np.unique(np.concatenate(([0.0], np.quantile(cal_conf, np.linspace(0, 1, 21)), [1.000001])))
    best_obj = -float("inf")
    chosen_threshold = 0.0
    for threshold in thresholds:
        active = cal_conf < threshold
        candidate = cal_base.copy()
        candidate[active] = cal_local[active]
        score = float(accuracy_score(y_cal, candidate)) - compute_penalty * float(np.mean(active))
        if score > best_obj + 1e-12:
            best_obj, chosen_threshold = score, float(threshold)

    test_probs = mlp.predict_proba(X_test_s)
    test_conf = np.max(test_probs, axis=1)
    use_local = test_conf < chosen_threshold
    adaptive_pred = mlp.predict(X_test_s)
    # The local branch is invoked only on the selected subset.
    if np.any(use_local):
        adaptive_pred[use_local] = knn.predict(X_test_s[use_local])
    adaptive_acc = float(accuracy_score(y_test, adaptive_pred))

    # Calibration utility is reported separately so its objective is transparent.
    cal_active = cal_conf < chosen_threshold
    cal_pred = cal_base.copy()
    cal_pred[cal_active] = cal_local[cal_active]
    return {
        "seed": int(seed),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "neural_core_accuracy": neural_acc,
        "knn_all_accuracy": knn_acc,
        "aethernet_accuracy": adaptive_acc,
        "aethernet_local_activation_rate": float(np.mean(use_local)),
        "aethernet_gate_threshold": float(chosen_threshold),
        "calibration_accuracy": float(accuracy_score(y_cal, cal_pred)),
        "calibration_activation_rate": float(np.mean(cal_active)),
        "mlp_iterations": int(mlp.n_iter_),
        "compute_penalty": float(compute_penalty),
        "notes": "Activation rate is a proxy for added KNN calls, not measured energy or FLOPs.",
    }
