"""Small in-memory checks; never fit a model or write experiment outputs."""

import importlib
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


class SharedModuleTests(unittest.TestCase):
    def common(self):
        try:
            return importlib.import_module("gmm_experiment_common")
        except ModuleNotFoundError as error:
            if error.name != "gmm_experiment_common":
                raise
            self.fail("The shared experiment module has not been extracted yet.")

    def test_distance_and_coverage_distinguish_identical_and_disjoint_sets(self):
        common = self.common()
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]], dtype=float)
        same = common.evaluate(x, x, 1)
        self.assertAlmostEqual(same["fid_2d"], 0)
        self.assertEqual((same["precision"], same["recall"]), (1, 1))
        shifted = common.evaluate(x, x + [10, 0], 1)
        self.assertAlmostEqual(shifted["fid_2d"], 100)
        self.assertEqual((shifted["precision"], shifted["recall"]), (0, 0))
        missing = common.evaluate(x, np.empty((0, 2)), 1)
        self.assertEqual(missing["recall"], 0)
        self.assertIsNone(missing["precision"])

    def test_class_assignment_uses_true_centers_and_resolves_ties(self):
        common = self.common()
        actual = common.assign_classes(np.array([[1, 0], [3, 0], [2, 0]]), np.array([[0, 0], [4, 0]]))
        np.testing.assert_array_equal(actual, [0, 1, 0])

    def test_training_composition_and_seeded_reproducibility(self):
        common = self.common()
        synthetic = {"weights": np.array([1.]), "means": np.array([[-100., -100.]]),
                     "covariances": np.array([np.eye(2) * 0.01])}
        true = {**synthetic, "means": np.array([[100., 100.]])}
        for fraction, expected_fresh in ((0., 0), (0.1, 200), (1., 2000)):
            with self.subTest(fraction=fraction):
                args = SimpleNamespace(fresh_fraction=fraction, true_parameters=true)
                x = common.training_sample(synthetic, 3, 20, args)
                self.assertEqual(x.shape, (2000, 2))
                self.assertEqual(int((x[:, 0] > 0).sum()), expected_fresh)
                np.testing.assert_array_equal(x, common.training_sample(synthetic, 3, 20, args))


if __name__ == "__main__":
    unittest.main()
