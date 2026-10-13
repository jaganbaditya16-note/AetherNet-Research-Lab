"""Small CPU-friendly architecture experiments for AetherNet Research Lab 0.2.0.

These are implementations of established model families for fair comparison, not
claims of newly invented neural architectures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class ExtremeLearningMachine:
    """Fixed random nonlinear hidden layer with a ridge-fitted readout.

    This is a lightweight Extreme Learning Machine-style baseline. Hidden weights
    are not trained by backpropagation; only the output readout is fitted.
    """
    def __init__(self, hidden_units: int = 256, alpha: float = 1e-2, random_state: int = 7):
        self.hidden_units = int(hidden_units)
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.scaler: StandardScaler | None = None
        self.weights_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None
        self.readout_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def _hidden(self, X: np.ndarray) -> np.ndarray:
        assert self.weights_ is not None and self.bias_ is not None
        return np.tanh(X @ self.weights_ + self.bias_)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtremeLearningMachine":
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        rng = np.random.default_rng(self.random_state)
        self.weights_ = rng.normal(0.0, 1.0 / np.sqrt(max(1, Z.shape[1])), (Z.shape[1], self.hidden_units))
        self.bias_ = rng.normal(0.0, 0.5, self.hidden_units)
        H = self._hidden(Z)
        self.classes_ = np.unique(y)
        Y = (y[:, None] == self.classes_[None, :]).astype(float)
        gram = H.T @ H + self.alpha * np.eye(H.shape[1])
        self.readout_ = np.linalg.solve(gram, H.T @ Y)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.readout_ is None:
            raise RuntimeError("Call fit before predict")
        return self._hidden(self.scaler.transform(np.asarray(X, dtype=float))) @ self.readout_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp = np.exp(np.clip(scores, -50, 50))
        return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("Call fit before predict")
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]


class HyperdimensionalClassifier:
    """Random-projection hypervector encoder with class prototype bundling.

    A practical baseline inspired by hyperdimensional/vector-symbolic computing.
    Encoding and prototype aggregation are explicit so the operations can be audited.
    """
    def __init__(self, dimensions: int = 2048, random_state: int = 7):
        if dimensions < 64:
            raise ValueError("dimensions must be >= 64")
        self.dimensions = int(dimensions)
        self.random_state = int(random_state)
        self.scaler: StandardScaler | None = None
        self.projection_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None
        self.prototypes_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def _encode(self, X: np.ndarray) -> np.ndarray:
        assert self.projection_ is not None and self.bias_ is not None
        # Boolean hypervectors represented by -1/+1 for fast dot-product similarity.
        return np.where(X @ self.projection_ + self.bias_ >= 0.0, 1.0, -1.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HyperdimensionalClassifier":
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        rng = np.random.default_rng(self.random_state)
        self.projection_ = rng.normal(0.0, 1.0 / np.sqrt(max(1, Z.shape[1])), (Z.shape[1], self.dimensions))
        self.bias_ = rng.normal(0.0, 0.1, self.dimensions)
        H = self._encode(Z)
        self.classes_ = np.unique(y)
        # Bundling: sum the class's hypervectors into a prototype.
        self.prototypes_ = np.vstack([H[y == cls].mean(axis=0) for cls in self.classes_])
        # Normalize prototype magnitude so classes with more examples do not win by size.
        norms = np.linalg.norm(self.prototypes_, axis=1, keepdims=True)
        self.prototypes_ = self.prototypes_ / np.maximum(norms, 1e-12)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.prototypes_ is None or self.classes_ is None:
            raise RuntimeError("Call fit before predict")
        Z = self.scaler.transform(np.asarray(X, dtype=float))
        H = self._encode(Z)
        H = H / np.sqrt(self.dimensions)
        scores = H @ self.prototypes_.T
        return self.classes_[np.argmax(scores, axis=1)]


class RBFNetwork:
    """Radial-basis feature network with learned linear readout."""
    def __init__(self, centers: int = 48, alpha: float = 1.0, random_state: int = 7):
        self.centers = int(centers)
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.scaler: StandardScaler | None = None
        self.kmeans: KMeans | None = None
        self.gamma_: float | None = None
        self.readout: RidgeClassifier | None = None

    def _rbf_features(self, X: np.ndarray) -> np.ndarray:
        assert self.kmeans is not None and self.gamma_ is not None
        # ||x-c||^2 without allocating a 3D tensor.
        x2 = np.sum(X * X, axis=1, keepdims=True)
        c2 = np.sum(self.kmeans.cluster_centers_ ** 2, axis=1)[None, :]
        d2 = np.maximum(x2 + c2 - 2.0 * X @ self.kmeans.cluster_centers_.T, 0.0)
        return np.exp(-self.gamma_ * d2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RBFNetwork":
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        k = max(2, min(self.centers, len(Z)))
        self.kmeans = KMeans(n_clusters=k, n_init=3, random_state=self.random_state).fit(Z)
        centers = self.kmeans.cluster_centers_
        if len(centers) > 1:
            dist = np.sqrt(np.maximum(np.sum(centers * centers, axis=1)[:, None] +
                                      np.sum(centers * centers, axis=1)[None, :] -
                                      2 * centers @ centers.T, 0.0))
            nonzero = dist[dist > 1e-10]
            width = float(np.median(nonzero)) if nonzero.size else 1.0
        else:
            width = 1.0
        self.gamma_ = 1.0 / max(2.0 * width * width, 1e-8)
        Phi = self._rbf_features(Z)
        self.readout = RidgeClassifier(alpha=self.alpha).fit(Phi, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.readout is None:
            raise RuntimeError("Call fit before predict")
        Z = self.scaler.transform(np.asarray(X, dtype=float))
        return self.readout.predict(self._rbf_features(Z))


def build_model_suite(seed: int = 7) -> list[tuple[str, str, Any]]:
    """Return model name, family label and estimator factory result."""
    return [
        ("MLP", "Neural", MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu",
            solver="adam", alpha=1e-3, learning_rate_init=0.004, max_iter=160,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=12,
            random_state=seed)),
        ("Extreme Learning Machine", "Neural / fixed random features",
            ExtremeLearningMachine(hidden_units=192, alpha=0.02, random_state=seed)),
        ("Hyperdimensional prototype", "Alternative representation",
            HyperdimensionalClassifier(dimensions=1024, random_state=seed)),
        ("RBF network", "Neural / radial basis",
            RBFNetwork(centers=40, alpha=1.0, random_state=seed)),
        ("KNN", "Instance-based", KNeighborsClassifier(n_neighbors=11, weights="distance")),
        ("Nearest centroid", "Prototype-based", NearestCentroid()),
        ("Random forest", "Non-neural baseline", RandomForestClassifier(n_estimators=80,
            max_depth=None, min_samples_leaf=2, random_state=seed, n_jobs=1)),
    ]
