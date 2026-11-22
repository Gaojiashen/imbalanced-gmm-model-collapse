"""Unbalanced GMM full-synthetic loops: lambda = 1, 0.8, 0.5.

Training samples use covariance lambda*Sigma (standard deviation sqrt(lambda)).
Weights/means are unchanged during sampling; all parameters are relearned.
Main evaluations always sample without bias from the fitted model. Snapshot
NPZ files contain X_train (biased) and X (unbiased evaluation) for comparison.
All cases start from the same M1 and use paired generation-specific RNG streams.
The lambda=1 case is a fresh backend-matched control, separate from old results.

    uv run --locked --cache-dir .uv-cache code/run_sampling_bias.py --generations 20 --seeds 0
    uv run --locked --cache-dir .uv-cache code/run_sampling_bias.py --resume
Optional --lambdas 1 0.8 0.5. Outputs/figures under sampling_bias/lambda_*.
"""
import gmm_experiment_common as common


def main():
    parser = common.suite_parser(__doc__)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[1.0, .8, .5])
    args = parser.parse_args()
    if any(not 0 < value <= 1 for value in args.lambdas):
        parser.error("Each lambda must be in (0, 1].")
    cases = {}
    for value in dict.fromkeys(args.lambdas):
        condition = f"sampling_bias/lambda_{value:g}"
        cases[f"lambda={value:g}"] = condition
        common.run_suite_case(condition, {"mode": "fresh", "n_fresh": 0,
                                          "n_synthetic": 2000, "lambda": value}, args)
    common.compare_suite("sampling_bias", cases, args.seeds)


if __name__ == "__main__":
    main()
