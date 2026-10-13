import unittest
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from aethernet import AetherNet


class AetherNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_moons(n_samples=420, noise=0.2, random_state=5)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.25, random_state=11, stratify=y
        )

    def test_fit_and_predict_shape(self):
        model = AetherNet(random_state=3, max_iter=80).fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        self.assertEqual(predictions.shape, self.y_test.shape)

    def test_activation_rate_is_bounded(self):
        model = AetherNet(random_state=4, max_iter=80).fit(self.X_train, self.y_train)
        report = model.predict_report(self.X_test)
        self.assertGreaterEqual(report.activation_rate, 0.0)
        self.assertLessEqual(report.activation_rate, 1.0)

    def test_no_test_labels_are_needed_for_prediction(self):
        model = AetherNet(random_state=5, max_iter=80).fit(self.X_train, self.y_train)
        preds_a = model.predict(self.X_test)
        preds_b = model.predict(self.X_test)
        np.testing.assert_array_equal(preds_a, preds_b)

    def test_tiny_dataset_rejected(self):
        with self.assertRaises(ValueError):
            AetherNet().fit(self.X_train[:20], self.y_train[:20])

if __name__ == "__main__":
    unittest.main()
