"""In-memory scientific checks; no model fitting or experiment outputs."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import gmm_experiment_common as common


class ReproductionTests(unittest.TestCase):
    def test_bias_scales_standard_deviation_not_weights_or_mean(self):
        p = {"weights": np.array([1.]), "means": np.array([[3., 4.]]),
             "covariances": np.array([np.eye(2)])}
        original, labels = common.sample(p, np.random.default_rng(3), 8)
        biased, other = common.sample(p, np.random.default_rng(3), 8, covariance_scale=0.25)
        np.testing.assert_array_equal(labels, other)
        np.testing.assert_allclose(biased, p["means"][0] + 0.5 * (original - p["means"][0]))
        np.testing.assert_array_equal(p["covariances"], [np.eye(2)])

    def test_fixed_and_fresh_recipes_allow_different_training_sizes(self):
        p = {"weights": np.array([1.]), "means": np.array([[-100., -100.]]),
             "covariances": np.array([np.eye(2) * .01])}
        args = SimpleNamespace(recipe={"mode": "fixed", "n_synthetic": 3, "lambda": 1.},
                               fixed_real=np.array([[100., 100.], [101., 101.]]),
                               true_parameters={**p, "means": np.array([[100., 100.]])})
        x = common.training_sample(p, 0, 2, args)
        self.assertEqual(x.shape, (5, 2))
        np.testing.assert_array_equal(x[x[:, 0] > 0], args.fixed_real)
        args.recipe = {"mode": "fresh", "n_fresh": 2, "n_synthetic": 3, "lambda": 1.}
        x = common.training_sample(p, 0, 2, args)
        self.assertEqual((len(x), int((x[:, 0] > 0).sum())), (5, 2))

    def test_sliced_distance_translation(self):
        x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        self.assertAlmostEqual(common.sliced_w2_squared(x, x), 0)
        self.assertAlmostEqual(common.sliced_w2_squared(x, x + [2., 0.]), 2)

    def test_effective_size_requires_unique_interpolation(self):
        result = common.match_effective_size([100, 200, 400], [4., 2., 1.], 1.5)
        self.assertAlmostEqual(result["n_effective"], np.sqrt(200 * 400))
        ambiguous = common.match_effective_size([100, 200, 400], [4., 1., 3.], 2.)
        self.assertIsNone(ambiguous["n_effective"])
        self.assertEqual(ambiguous["status"], "ambiguous")

    def test_tail_screen_rejects_short_runs_and_continuing_drift(self):
        from run_effective_sample_size import tail_summary
        def reports(generations, values):
            return [{"generation": g, "overall": {"sliced_w2_squared": v}}
                    for g, v in zip(generations, values)]
        short = reports([1, 2, 5, 10, 20], [1.] * 5)
        self.assertFalse(tail_summary(short, 500, .2)["stability_screen_passed"])
        stable = reports(range(520, 1001, 20), [1.] * 25)
        self.assertTrue(tail_summary(stable, 500, .2)["stability_screen_passed"])
        drifting = reports(range(520, 1001, 20), range(1, 26))
        self.assertFalse(tail_summary(drifting, 500, .2)["stability_screen_passed"])


if __name__ == "__main__":
    unittest.main()
