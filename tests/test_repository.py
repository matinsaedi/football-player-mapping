"""Dependency-free checks for the preserved repository artifacts."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryIntegrityTests(unittest.TestCase):
    def test_preserved_model_has_expected_input_and_classes(self):
        payload = json.loads((ROOT / "models" / "model.json").read_text(encoding="utf-8"))
        layers = payload["config"]["layers"]
        input_layer = layers[0]
        output_layer = layers[-1]

        self.assertEqual(input_layer["config"]["batch_input_shape"], [None, 45, 45, 3])
        self.assertEqual(output_layer["config"]["units"], 3)
        self.assertEqual(output_layer["config"]["activation"], "softmax")

    def test_required_small_artifacts_exist(self):
        expected = [
            ROOT / "assets" / "field-map.png",
            ROOT / "assets" / "demo.gif",
            ROOT / "assets" / "result-screenshot.png",
            ROOT / "models" / "model.h5",
        ]
        for path in expected:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_demo_is_small_enough_for_a_readme(self):
        demo = ROOT / "assets" / "demo.gif"
        self.assertLess(demo.stat().st_size, 10 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
