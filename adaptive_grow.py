"""AetherGrow: validation-selected width for a fixed-random-feature learner.

This is a research candidate built from an established ELM-style method plus a
validation-based capacity-selection rule. Adaptive model sizing is an established
idea, so this implementation is not claimed to be a first-ever architecture.
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class AetherGrow:
    """Choose the smallest ELM hidden width within tolerance of best calibration accuracy."""
    def __init__(self, widths=(32, 64, 96, 128, 192, 256), tolerance: float = 0.01,
                 alpha: float = 0.02, random_state: int = 7):
        widths = tuple(sorted(set(int(w) for w in widths)))
        if not widths or widths[0] < 4:
            raise ValueError("widths must contain integers >= 4")
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        self.widths = widths
        self.tolerance = float(tolerance)
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.scaler_: StandardScaler | None = None
        self.weights_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None
        self.readout_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.selected_hidden_units_: int | None = None
        self.calibration_scores_: dict[int, float] = {}

    def _hidden(self, X: np.ndarray, width: int) -> np.ndarray:
        assert self.weights_ is not None and self.bias_ is not None
        return np.tanh(X @ self.weights_[:, :width] + self.bias_[:width])

    def _fit_readout(self, H: np.ndarray, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
        Y = (y[:, None] == classes[None, :]).astype(float)
        gram = H.T @ H + self.alpha * np.eye(H.shape[1])
        return np.linalg.solve(gram, H.T @ Y)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AetherGrow":
        X = np.asarray(X, dtype=float); y = np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
            raise ValueError("X must be 2D and y must be 1D with matching sample counts")
        if len(X) < 60 or len(np.unique(y)) < 2:
            raise ValueError("Need at least 60 examples and at least two classes")
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X, y, test_size=0.25, random_state=self.random_state, stratify=y
        )
        scaler_cal = StandardScaler().fit(X_fit)
        Xf = scaler_cal.transform(X_fit); Xc = scaler_cal.transform(X_cal)
        max_width = max(self.widths)
        rng = np.random.default_rng(self.random_state)
        W = rng.normal(0, 1.0 / np.sqrt(max(1, X.shape[1])), (X.shape[1], max_width))
        b = rng.normal(0, 0.5, max_width)
        classes = np.unique(y_fit)
        scores = {}
        for width in self.widths:
            Hf = np.tanh(Xf @ W[:, :width] + b[:width])
            readout = self._fit_readout(Hf, y_fit, classes)
            Hc = np.tanh(Xc @ W[:, :width] + b[:width])
            pred = classes[np.argmax(Hc @ readout, axis=1)]
            scores[width] = float(accuracy_score(y_cal, pred))
        best = max(scores.values())
        eligible = [width for width, score in scores.items() if score >= best - self.tolerance]
        selected = min(eligible)

        # Refit selected capacity using all training rows while preserving reproducible
        # prefix random features, then train the output readout on all available training data.
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        rng = np.random.default_rng(self.random_state)
        full_weights = rng.normal(0, 1.0 / np.sqrt(max(1, Z.shape[1])), (Z.shape[1], max_width))
        full_bias = rng.normal(0, 0.5, max_width)
        # Persist only the selected prefix. This is the actual compact deployed model,
        # rather than keeping unused maximum-width weights in the saved artifact.
        self.weights_ = full_weights[:, :selected].copy()
        self.bias_ = full_bias[:selected].copy()
        self.classes_ = np.unique(y)
        H = self._hidden(Z, selected)
        self.readout_ = self._fit_readout(H, y, self.classes_)
        self.selected_hidden_units_ = selected
        self.calibration_scores_ = scores
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.scaler_ is None or self.readout_ is None or self.classes_ is None or self.selected_hidden_units_ is None:
            raise RuntimeError("Call fit(X, y) before prediction")
        Z = self.scaler_.transform(np.asarray(X, dtype=float))
        scores = self._hidden(Z, self.selected_hidden_units_) @ self.readout_
        return self.classes_[np.argmax(scores, axis=1)]
