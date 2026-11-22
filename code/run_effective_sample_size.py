"""Empirical distance-specific effective sample size and synthetic-data boundary.

Default: n_fresh=200, n_synthetic=200,800,1800,3800; lambda=1,0.8; 5 seeds;
2000 generations. Calibration: 50 independent real-only fits at n=100,200,
400,800,1600,3200,6400,12800. These direct fits are not recursive trajectories.
All evaluations use 2000 generated points versus the fixed original 2000.

Distance: squared sliced W2 on 128 fixed directions in raw 2D, NOT exact GMM
Wasserstein distance. Match tail-mean distance to calibration via unique local
log(n) interpolation. No extrapolation or forced monotonicity. Nonunique,
out-of-range, failed or short runs do not yield a claimed effective size.
Tail stability uses a declared diagnostic: at least 5 tail evaluations spanning
part of a complete 500-generation tail, and first/second-half mean difference
<=20% of tail mean for every trajectory. Passing is not a proof of stationarity.
Loop uncertainty is summarized over seed-level tail means, never independent
time points. Calibration SD/SE and per-seed inversions are saved; ratios near
one are descriptive and do not establish a statistically significant benefit.

Commands from GAOJiashen (nothing runs on import):
    uv run --locked --cache-dir .uv-cache code/run_effective_sample_size.py --phase calibrate
    uv run --locked --cache-dir .uv-cache code/run_effective_sample_size.py --phase loops
    uv run --locked --cache-dir .uv-cache code/run_effective_sample_size.py --phase analyze
Or use default --phase all; --resume reuses individual calibration fits and
loop checkpoints. Pilot: --generations 20 --seeds 0 --calibration-repeats 3;
use separate --study-name effective_sample_size_pilot for different grids.
"""
from pathlib import Path
import numpy as np
import gmm_experiment_common as common


def select_backend(device):
    if device == "cpu":
        return "cpu", common.fit_cpu
    try:
        import torch
    except ImportError:
        if device == "cuda":
            raise RuntimeError("CUDA PyTorch is required.")
    else:
        if torch.cuda.is_available():
            return "cuda", common.fit_cuda
        if device == "cuda":
            raise RuntimeError("CUDA is not available.")
    return "cpu", common.fit_cpu


def case_grid(args):
    return {f"lambda={value:g}, ns={size}":
            (f"{args.study_name}/lambda_{value:g}/ns_{size}",
             {"mode": "fresh", "n_fresh": args.n_fresh, "n_synthetic": size, "lambda": value})
            for value in dict.fromkeys(args.lambdas) for size in dict.fromkeys(args.synthetic_sizes)}


def calibrate(args):
    output = common.ROOT / "outputs" / args.study_name / "calibration"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"sizes": args.calibration_sizes, "repeats": args.calibration_repeats,
                "device": args.device, "fit": common.FIT, "reference": str(common.INITIAL),
                "reference_sha256": common.hashlib.sha256(common.INITIAL.read_bytes()).hexdigest(),
                "config_sha256": common.hashlib.sha256(common.TRUE_CONFIG.read_bytes()).hexdigest(),
                "k": args.k, "distance": "sliced_w2_squared", "directions": 128}
    manifest_path = output / "settings.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError("Calibration exists; use --resume or a different --study-name.")
        if common.read_json(manifest_path) != manifest:
            raise ValueError("Calibration settings changed; use a different --study-name.")
    else:
        common.write_json(manifest_path, manifest)
    _, fitter = select_backend(args.device)
    true = common.parameters(common.read_json(common.TRUE_CONFIG))
    real = common.read_input(common.INITIAL)
    rows = []
    for size in args.calibration_sizes:
        for repeat in range(args.calibration_repeats):
            path = output / f"n_{size}_repeat_{repeat}.json"
            if path.exists() and args.resume:
                row = common.read_json(path)["metrics"]
            else:
                # A nested true pool is shared across sample sizes for each repeat.
                pool, _ = common.sample(true, common.random_stream(2026, repeat, 51), max(args.calibration_sizes))
                x = pool[:size]
                fit_seed = int(common.random_stream(2026, repeat, 52).integers(0, 2**32))
                p, training = fitter(x, fit_seed, args)
                generated, _ = common.sample(p, common.random_stream(2026, repeat, 53))
                row = {"n_real": size, "repeat": repeat, **common.evaluate(real, generated, args.k),
                       "sliced_w2_squared": common.sliced_w2_squared(real, generated),
                       "nll_initial": common.nll(p, real)}
                common.write_json(path, {"metrics": row, "training": training,
                                         **{key: value.tolist() for key, value in p.items()}})
                np.savez_compressed(output / f"n_{size}_repeat_{repeat}.npz", X=generated, X_train=x)
                print(f"calibration n={size} repeat={repeat + 1}/{args.calibration_repeats}", flush=True)
            rows.append(row)
    common.write_csv(output / "all_fits.csv", rows)
    summary = []
    for size in args.calibration_sizes:
        values = [r["sliced_w2_squared"] for r in rows if r["n_real"] == size]
        summary.append({"n_real": size, "n_fits": len(values), "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                        "standard_error": float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None})
    common.write_json(output / "summary.json", {"distance": "sliced_w2_squared", "rows": summary})
    common.write_csv(output / "summary.csv", summary)


def tail_summary(reports, window, tolerance):
    last = reports[-1]["generation"]
    tail = [r for r in reports if r["generation"] > last - window]
    values = np.array([r["overall"]["sliced_w2_squared"] for r in tail])
    mean = float(values.mean())
    half = len(values) // 2
    shift = float(abs(values[:half].mean() - values[half:].mean()) / max(mean, 1e-12)) if half else None
    stable = last >= window and len(values) >= 5 and shift is not None and shift <= tolerance
    return {"last_generation": last, "tail_evaluations": len(values), "tail_mean": mean,
            "relative_half_shift": shift, "stability_screen_passed": stable}


def analyze(args):
    output = common.ROOT / "outputs" / args.study_name
    summary_path = output / "calibration/summary.json"
    calibration = common.read_json(summary_path)["rows"]
    sizes, errors = [r["n_real"] for r in calibration], [r["mean"] for r in calibration]
    nr_baseline = next(r for r in calibration if r["n_real"] == args.n_fresh)
    case_rows, per_seed = [], []
    for label, (condition, recipe) in case_grid(args).items():
        settings_path = common.ROOT / "outputs" / condition / "settings/experiment.json"
        if settings_path.exists() and common.read_json(settings_path).get("recipe") != recipe:
            raise ValueError(f"Analysis recipe differs from saved training settings: {condition}")
        details = []
        for seed in args.seeds:
            directory = common.ROOT / "outputs" / condition / f"seed_{seed}"
            reports = [common.read_json(path) for path in sorted((directory / "metrics").glob("generation_*.json"))]
            if not reports:
                continue
            detail = {"condition": label, "seed": seed,
                      **tail_summary(reports, args.tail_window, args.tail_tolerance)}
            failure = (directory / "failure.json").exists()
            detail["complete"] = reports[-1]["generation"] >= args.generations and not failure
            if detail["stability_screen_passed"] and detail["complete"]:
                detail.update(common.match_effective_size(sizes, errors, detail["tail_mean"]))
            else:
                detail.update(n_effective=None, status="incomplete_or_tail_not_stable", candidates=[])
            details.append(detail)
            per_seed.append(detail)
        if not details:
            continue
        values = [r["tail_mean"] for r in details]
        eligible = (len(details) == len(args.seeds)
                    and len({r["last_generation"] for r in details}) == 1
                    and all(r["stability_screen_passed"] and r["complete"] for r in details))
        estimate = common.match_effective_size(sizes, errors, float(np.mean(values))) if eligible else {
            "n_effective": None, "status": "incomplete_or_tail_not_stable", "candidates": []}
        ne = estimate["n_effective"]
        case_rows.append({"lambda": recipe["lambda"], "n_fresh": args.n_fresh,
                          "n_synthetic": recipe["n_synthetic"], "n_seeds": len(details),
                          "min_last_generation": min(r["last_generation"] for r in details),
                          "max_last_generation": max(r["last_generation"] for r in details),
                          "mean_distance": float(np.mean(values)), "min_seed_distance": float(min(values)),
                          "max_seed_distance": float(max(values)),
                          "seed_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                          "real_only_distance": nr_baseline["mean"],
                          "distance_improvement": nr_baseline["mean"] - float(np.mean(values)),
                          "n_effective": ne, "effective_ratio": ne / args.n_fresh if ne is not None else None,
                          "status": estimate["status"], "exploratory_only": len(details) < 5,
                          "calibration_standard_error_at_nr": nr_baseline["standard_error"]})
    output.mkdir(parents=True, exist_ok=True)
    common.write_json(output / "effective_size.json", {
        "definition": "Distance-specific empirical size relative to the fixed finite reference; not exact GMM W2.",
        "uncertainty": "Calibration SD/SE, seed-level tail ranges, per-seed inversions. No significance claim from ratio alone.",
        "tail_window": args.tail_window, "tail_relative_shift_tolerance": args.tail_tolerance,
        "conditions": case_rows, "per_seed": per_seed})
    common.write_csv(output / "effective_size.csv", case_rows)
    common.write_csv(output / "tail_per_seed.csv", per_seed)
    plot_analysis(args, calibration, case_rows)


def plot_analysis(args, calibration, rows):
    fig, axes = common.plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    axes[0].errorbar([r["n_real"] for r in calibration], [r["mean"] for r in calibration],
                     yerr=[r["standard_error"] or 0 for r in calibration], marker="o")
    axes[0].set(xscale="log", xlabel="Real-only training samples", ylabel="Squared sliced W2", title="Calibration mean and SE")
    for value in args.lambdas:
        selected = sorted([r for r in rows if r["lambda"] == value], key=lambda r: r["n_synthetic"])
        axes[1].plot([r["n_synthetic"] for r in selected], [r["mean_distance"] for r in selected],
                     marker="o", label=f"lambda={value:g}")
        axes[2].plot([r["n_synthetic"] for r in selected],
                     [np.nan if r["effective_ratio"] is None else r["effective_ratio"] for r in selected],
                     marker="o", label=f"lambda={value:g}")
    baseline = next(r["mean"] for r in calibration if r["n_real"] == args.n_fresh)
    axes[1].axhline(baseline, color="black", linestyle="--", label=f"Real-only n={args.n_fresh}")
    axes[1].set(xlabel="Synthetic points per generation", ylabel="Tail mean squared sliced W2", title="Benefit at fixed real budget")
    axes[2].axhline(1, color="black", linestyle="--")
    axes[2].set(xlabel="Synthetic points per generation", ylabel="Effective size / fresh size", title="Unique interpolation only; gaps = N/A")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    directory = common.ROOT / "outputs/figures" / args.study_name
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / "effective_size_boundary.png", dpi=200)
    common.plt.close(fig)


def main():
    parser = common.suite_parser(__doc__)
    parser.add_argument("--phase", choices=("all", "calibrate", "loops", "analyze"), default="all")
    parser.add_argument("--study-name", default="effective_sample_size")
    parser.add_argument("--n-fresh", type=int, default=200)
    parser.add_argument("--synthetic-sizes", type=int, nargs="+", default=[200, 800, 1800, 3800])
    parser.add_argument("--lambdas", type=float, nargs="+", default=[1., .8])
    parser.add_argument("--calibration-sizes", type=int, nargs="+", default=[100, 200, 400, 800, 1600, 3200, 6400, 12800])
    parser.add_argument("--calibration-repeats", type=int, default=50)
    parser.add_argument("--tail-window", type=int, default=500)
    parser.add_argument("--tail-tolerance", type=float, default=.2)
    args = parser.parse_args()
    if (min(args.calibration_sizes) < 25 or args.n_fresh < 25 or min(args.synthetic_sizes) < 0
            or args.calibration_repeats < 1 or args.tail_window < 1 or args.tail_tolerance < 0
            or any(not 0 < value <= 1 for value in args.lambdas)):
        parser.error("Use calibration/fresh sizes >=25, nonnegative synthetic sizes, positive repeats/window and 0<lambda<=1.")
    if Path(args.study_name).name != args.study_name or args.study_name in (".", ".."):
        parser.error("--study-name must be a single folder name.")
    args.calibration_sizes = sorted(set([*args.calibration_sizes, args.n_fresh]))
    if args.phase != "analyze":
        args.device, _ = select_backend(args.device)
    if args.phase in ("all", "calibrate"):
        calibrate(args)
    if args.phase in ("all", "loops"):
        for condition, recipe in case_grid(args).values():
            common.run_suite_case(condition, recipe, args)
        common.compare_suite(args.study_name, {label: c for label, (c, _) in case_grid(args).items()}, args.seeds, args.tail_window)
    if args.phase in ("all", "analyze"):
        analyze(args)


if __name__ == "__main__":
    main()
