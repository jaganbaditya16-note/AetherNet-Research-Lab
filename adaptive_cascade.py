"""AetherCascade: experimental low-cost-to-specialist prediction cascade.

This research candidate composes established components (small MLP, ELM-style
readout, KNN). Its potential contribution is the calibrated routing policy and its
measured budget/accuracy behavior, if experiments establish a useful advantage.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from architectures import ExtremeLearningMachine


@dataclass
class CascadeReport:
    predictions: np.ndarray
    mlp_confidence: np.ndarray
    elm_activated: np.ndarray
    knn_activated: np.ndarray

    @property
    def elm_call_rate(self) -> float:
        return float(np.mean(self.elm_activated)) if len(self.elm_activated) else 0.0

    @property
    def knn_call_rate(self) -> float:
        return float(np.mean(self.knn_activated)) if len(self.knn_activated) else 0.0


class AetherCascade:
    """MLP -> ELM-style expert -> KNN fallback, with thresholds fit on calibration data.

    The cascade is intentionally small and CPU-oriented. Routing thresholds are
    tuned on a separate calibration split, never on test labels. It is an experiment,
    not a claim that cascaded classifiers are a new family or always cheaper.
    """
    def __init__(self, random_state: int = 7, compute_penalty: float = 0.025,
                 mlp_max_iter: int = 140, elm_hidden_units: int = 128,
                 n_neighbors: int = 11):
        if compute_penalty < 0:
            raise ValueError("compute_penalty must be >= 0")
        self.random_state = int(random_state)
        self.compute_penalty = float(compute_penalty)
        self.mlp_max_iter = int(mlp_max_iter)
        self.elm_hidden_units = int(elm_hidden_units)
        self.n_neighbors = int(n_neighbors)
        self.scaler: StandardScaler | None = None
        self.mlp: MLPClassifier | None = None
        self.elm: ExtremeLearningMachine | None = None
        self.knn: KNeighborsClassifier | None = None
        self.classes_: np.ndarray | None = None
        self.mlp_threshold_ = 0.0
        self.elm_threshold_ = 0.0
        self.calibration_accuracy_ = 0.0
        self.calibration_weighted_calls_ = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AetherCascade":
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2D and y must be 1D with matching sample counts")
        if len(X) < 60 or len(np.unique(y)) < 2:
            raise ValueError("Need at least 60 examples and at least two classes")
        X_core, X_cal, y_core, y_cal = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state, stratify=y
        )
        self.scaler = StandardScaler().fit(X_core)
        Xc = self.scaler.transform(X_core)
        Xv = self.scaler.transform(X_cal)
        self.classes_ = np.unique(y_core)
        self.mlp = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            alpha=1e-3, learning_rate_init=0.004, max_iter=self.mlp_max_iter,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=12,
            random_state=self.random_state)
        self.mlp.fit(Xc, y_core)
        self.elm = ExtremeLearningMachine(hidden_units=self.elm_hidden_units,
            alpha=0.02, random_state=self.random_state).fit(Xc, y_core)
        self.knn = KNeighborsClassifier(n_neighbors=min(self.n_neighbors, len(Xc)), weights="distance").fit(Xc, y_core)

        mlp_probs = self.mlp.predict_proba(Xv)
        mlp_conf = np.max(mlp_probs, axis=1)
        mlp_pred = self.classes_[np.argmax(mlp_probs, axis=1)]
        elm_probs = self.elm.predict_proba(Xv)
        elm_conf = np.max(elm_probs, axis=1)
        elm_pred = self.elm.classes_[np.argmax(elm_probs, axis=1)]
        knn_pred = self.knn.predict(Xv)
        q1 = np.quantile(mlp_conf, np.linspace(0.0, 1.0, 9))
        q2 = np.quantile(elm_conf, np.linspace(0.0, 1.0, 9))
        thresholds1 = np.unique(np.concatenate(([0.0], q1, [1.000001])))
        thresholds2 = np.unique(np.concatenate(([0.0], q2, [1.000001])))
        best_obj = -float("inf")
        best_t1, best_t2 = 0.0, 0.0
        best_acc, best_cost = float(accuracy_score(y_cal, mlp_pred)), 0.0
        for t1 in thresholds1:
            route_elm = mlp_conf < t1
            for t2 in thresholds2:
                route_knn = route_elm & (elm_conf < t2)
                pred = mlp_pred.copy()
                pred[route_elm] = elm_pred[route_elm]
                pred[route_knn] = knn_pred[route_knn]
                acc = float(accuracy_score(y_cal, pred))
                # Explicit routing-work proxy: ELM calls cost 0.5 unit and KNN calls 1 unit.
                # This is a design penalty, not a hardware-energy measurement.
                weighted_calls = float(np.mean(route_elm) * 0.5 + np.mean(route_knn))
                objective = acc - self.compute_penalty * weighted_calls
                if objective > best_obj + 1e-12 or (abs(objective - best_obj) <= 1e-12 and weighted_calls < best_cost):
                    best_obj, best_t1, best_t2 = objective, float(t1), float(t2)
                    best_acc, best_cost = acc, weighted_calls
        self.mlp_threshold_, self.elm_threshold_ = best_t1, best_t2
        self.calibration_accuracy_ = best_acc
        self.calibration_weighted_calls_ = best_cost
        return self

    def _check(self):
        if any(v is None for v in (self.scaler, self.mlp, self.elm, self.knn, self.classes_)):
            raise RuntimeError("Call fit(X, y) before prediction")

    def predict_report(self, X: np.ndarray) -> CascadeReport:
        self._check()
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        probs = self.mlp.predict_proba(Xs)
        conf = np.max(probs, axis=1)
        pred = self.classes_[np.argmax(probs, axis=1)]
        route_elm = conf < self.mlp_threshold_
        route_knn = np.zeros(len(Xs), dtype=bool)
        if np.any(route_elm):
            indices = np.flatnonzero(route_elm)
            elm_probs = self.elm.predict_proba(Xs[indices])
            elm_conf = np.max(elm_probs, axis=1)
            elm_pred = self.elm.classes_[np.argmax(elm_probs, axis=1)]
            pred[indices] = elm_pred
            route_knn_local = elm_conf < self.elm_threshold_
            if np.any(route_knn_local):
                knn_indices = indices[route_knn_local]
                pred[knn_indices] = self.knn.predict(Xs[knn_indices])
                route_knn[knn_indices] = True
        return CascadeReport(pred, conf, route_elm, route_knn)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_report(X).predictions
