"""Fixed-real augmentation: original 2000 points + 2000 new synthetic points.

Reuses saved M1; total training size is 4000 from generation 2. No new true
data, no historical synthetic pool. Reference remains the original 2000.
This adapts the paper's fixed-real/latest-synthetic setup to unbalanced GMMs;
it does not assume the deep-model conclusions must hold here. Scores on the
reused real training reference measure empirical retention, not held-out fit.

Default CUDA, 2000 generations, five seeds. Commands from GAOJiashen:
    uv run --locked --cache-dir .uv-cache code/run_fixed_real.py --generations 20 --seeds 0
    uv run --locked --cache-dir .uv-cache code/run_fixed_real.py --resume
Outputs: outputs/fixed_real; figures: outputs/figures/fixed_real.
"""
import gmm_experiment_common as common


def main():
    args = common.suite_parser(__doc__).parse_args()
    common.run_suite_case("fixed_real", {"mode": "fixed", "n_synthetic": 2000, "lambda": 1.0}, args)


if __name__ == "__main__":
    main()
