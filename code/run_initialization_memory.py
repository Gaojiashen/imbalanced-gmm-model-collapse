"""Compare n_initial=500,2000,8000; subsequent fits use 200 fresh +1800 synthetic.

For each seed, generate one nested initial pool (at least 8000 points), train
all three initial models afresh, then pair all subsequent RNG streams across
initial sizes. Original 2000 points remain the evaluation reference, not the
source of new initial/training points. Evaluation tests loss of initial-quality
effects on finite-horizon performance; it cannot prove convergence in law.

    uv run --locked --cache-dir .uv-cache code/run_initialization_memory.py --generations 20 --seeds 0
    uv run --locked --cache-dir .uv-cache code/run_initialization_memory.py --resume
Optional --initial-sizes 500 2000 8000. Outputs under initialization_memory/n_*.
"""
import gmm_experiment_common as common


def main():
    parser = common.suite_parser(__doc__)
    parser.add_argument("--initial-sizes", type=int, nargs="+", default=[500, 2000, 8000])
    args = parser.parse_args()
    if min(args.initial_sizes) < common.FIT["n_components"]:
        parser.error("Initial samples must be at least the number of GMM components.")
    cases = {}
    pool_size = max(8000, max(args.initial_sizes))
    for size in dict.fromkeys(args.initial_sizes):
        condition = f"initialization_memory/n_{size}"
        cases[f"initial n={size}"] = condition
        common.run_suite_case(condition, {"mode": "fresh", "n_fresh": 200,
                                          "n_synthetic": 1800, "lambda": 1.0,
                                          "initial_size": size, "initial_pool_size": pool_size}, args)
    common.compare_suite("initialization_memory", cases, args.seeds, tail_window=500)


if __name__ == "__main__":
    main()
