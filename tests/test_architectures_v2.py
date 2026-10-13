import unittest
import numpy as np
from sklearn.datasets import make_moons
from architectures import ExtremeLearningMachine, HyperdimensionalClassifier, RBFNetwork, build_model_suite
from adaptive_cascade import AetherCascade
from adaptive_grow import AetherGrow
from benchmarks_v2 import run_suite, make_dataset, DATASETS, REAL_DATASETS


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


    def test_real_dataset_registry_and_bundled_sources(self):
        self.assertEqual(len(DATASETS), 7)
        self.assertEqual(set(REAL_DATASETS), {"breast_cancer", "wine", "digits"})
        for name in REAL_DATASETS:
            X, y = make_dataset(name, seed=23, n_samples=300)
            self.assertEqual(X.ndim, 2)
            self.assertEqual(y.ndim, 1)
            self.assertEqual(len(X), len(y))
            self.assertGreaterEqual(len(np.unique(y)), 2)
            self.assertTrue(np.isfinite(X).all())

    def test_bundled_small_dataset_uses_all_available_rows(self):
        X, y = make_dataset("wine", seed=23, n_samples=900)
        self.assertEqual(X.shape, (178, 13))
        self.assertEqual(len(np.unique(y)), 3)

    def test_real_dataset_benchmark_records_source_metadata(self):
        result = run_suite(repeats=1, n_samples=300, dataset_names=["wine"])
        block = result["datasets"]["wine"]
        self.assertEqual(result["version"], "0.3.0")
        self.assertEqual(block["metadata"]["kind"], "bundled-real")
        self.assertEqual(block["metadata"]["samples_used"], 178)
        self.assertEqual(block["metadata"]["classes"], 3)
        self.assertIn("source", block["metadata"])

    def test_quick_benchmark_contains_models_and_metrics(self):
        result = run_suite(repeats=1, n_samples=220, dataset_names=["moons"])
        rows = result["datasets"]["moons"]["trials"][0]["models"]
        self.assertGreaterEqual(len(rows), 10)
        self.assertTrue(all(0 <= row["accuracy"] <= 1 for row in rows))
        self.assertTrue(all(row["fit_seconds"] >= 0 for row in rows))

if __name__ == "__main__":
    unittest.main()
