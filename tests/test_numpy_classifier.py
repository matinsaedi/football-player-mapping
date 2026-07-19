"""Numerical smoke tests for TensorFlow-free model inference."""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from numpy_classifier import NumpyClassifier  # noqa: E402


class NumpyClassifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = NumpyClassifier(ROOT / "models" / "model.h5")

    def test_prediction_is_a_probability_distribution(self):
        batch = np.zeros((1, 45, 45, 3), dtype=np.uint8)
        prediction = self.model.predict(batch)

        self.assertEqual(prediction.shape, (1, 3))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertAlmostEqual(float(prediction.sum()), 1.0, places=5)

    def test_batch_prediction(self):
        batch = np.stack(
            [
                np.zeros((45, 45, 3), dtype=np.uint8),
                np.full((45, 45, 3), 255, dtype=np.uint8),
            ]
        )
        prediction = self.model.predict(batch)
        self.assertEqual(prediction.shape, (2, 3))


if __name__ == "__main__":
    unittest.main()
