import unittest
import numpy as np
from sklearn.datasets import make_moons
from architectures import ExtremeLearningMachine, HyperdimensionalClassifier, RBFNetwork, build_model_suite
from adaptive_cascade import AetherCascade
from adaptive_grow import AetherGrow
from benchmarks_v2 import run_suite


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_moons(n_samples=260, noise=0.22, random_state=17)
        cls.X, cls.y = X, y

    def test_elm_shape(self):
        model = ExtremeLearningMachine(hidden_units=48, random_state=3).fit(self.X[:200], self.y[:200])
        self.assertEqual(model.predict(self.X[200:]).shape, self.y[200:].shape)

    def test_hyperdimensional_shape(self):
        model = HyperdimensionalClassifier(dimensions=256, random_state=3).fit(self.X[:200], self.y[:200])
        self.assertEqual(model.predict(self.X[200:]).shape, self.y[200:].shape)

    def test_rbf_shape(self):
        model = RBFNetwork(centers=12, random_state=3).fit(self.X[:200], self.y[:200])
        self.assertEqual(model.predict(self.X[200:]).shape, self.y[200:].shape)

    def test_adaptive_grow_selects_supported_width(self):
        model = AetherGrow(widths=(16, 32, 64), random_state=2).fit(self.X[:200], self.y[:200])
        self.assertIn(model.selected_hidden_units_, (16, 32, 64))
        self.assertEqual(model.predict(self.X[200:]).shape, self.y[200:].shape)

    def test_adaptive_cascade_produces_routing_report(self):
        model = AetherCascade(random_state=2, mlp_max_iter=60, elm_hidden_units=48).fit(self.X[:200], self.y[:200])
        report = model.predict_report(self.X[200:])
        self.assertEqual(report.predictions.shape, self.y[200:].shape)
        self.assertGreaterEqual(report.elm_call_rate, 0.0)
        self.assertLessEqual(report.elm_call_rate, 1.0)
        self.assertGreaterEqual(report.knn_call_rate, 0.0)
        self.assertLessEqual(report.knn_call_rate, 1.0)

    def test_all_suite_models_are_named(self):
        names = [name for name, _, _ in build_model_suite(5)]
        self.assertIn("MLP", names)
        self.assertIn("Extreme Learning Machine", names)
        self.assertIn("Hyperdimensional prototype", names)
        self.assertIn("RBF network", names)

    def test_quick_benchmark_contains_models_and_metrics(self):
        result = run_suite(repeats=1, n_samples=220, dataset_names=["moons"])
        rows = result["datasets"]["moons"]["trials"][0]["models"]
        self.assertGreaterEqual(len(rows), 10)
        self.assertTrue(all(0 <= row["accuracy"] <= 1 for row in rows))
        self.assertTrue(all(row["fit_seconds"] >= 0 for row in rows))

if __name__ == "__main__":
    unittest.main()
