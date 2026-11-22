"""Fit generation 1, draw 2,000 samples, and evaluate against the input data.

This module only performs work when main() is called. See README.md for metric
definitions, class assignments, edge cases, and output descriptions.
"""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import scipy
import sklearn

from gmm_experiment_common import (
    assign_classes, create_cpu_model, draw_samples, evaluate, frechet_distance,
    manifold_coverage, plot_class_metrics, plot_comparison, read_input, write_json,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/datas/initial_samples_n2000_seed42.npz"
CONFIG = ROOT / "settings/gao_unbalanced_true_gmm.json"
N_SAMPLES = 2000

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--fit-seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=43)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if not 0 <= args.fit_seed < 2**32 or args.sample_seed < 0:
        parser.error("--fit-seed must be in [0, 2**32); --sample-seed must be nonnegative")
    if not 1 <= args.k < N_SAMPLES:
        parser.error("--k must be between 1 and 1999")
    input_path = args.input.resolve()
    real = read_input(input_path)
    raw_config = CONFIG.read_bytes()
    config = json.loads(raw_config.decode("utf-8-sig"))
    centers = np.asarray(config["means"], dtype=np.float64)
    n_classes = int(config["n_components"])
    if centers.shape != (n_classes, 2) or not np.isfinite(centers).all():
        raise ValueError("True centers must be a finite (n_components, 2) array.")

    # Only model size is specified from the experimental design. Neither true
    # parameters nor labels initialize or constrain the unsupervised fit.
    model = create_cpu_model(args.fit_seed, n_classes)
    model.fit(real)
    generated, fitted_ids = draw_samples(model, args.sample_seed)
    real_classes = assign_classes(real, centers)
    generated_classes = assign_classes(generated, centers)
    overall = evaluate(real, generated, args.k)
    rare = set(config.get("rare_component_indices_zero_based", []))
    class_rows = []
    for class_id in range(n_classes):
        row = {
            "class_id": class_id,
            "center_x": float(centers[class_id, 0]),
            "center_y": float(centers[class_id, 1]),
            "is_rare": class_id in rare,
            **evaluate(real[real_classes == class_id], generated[generated_classes == class_id], args.k),
        }
        row["initial_fraction"] = row["n_initial"] / len(real)
        row["generated_fraction"] = row["n_generated"] / len(generated)
        row["fraction_difference"] = row["generated_fraction"] - row["initial_fraction"]
        class_rows.append(row)

    directories = {name: ROOT / "outputs" / name for name in ("models", "datas", "metrics", "figures")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    stem = f"generation1_fit{args.fit_seed}_sample{args.sample_seed}_k{args.k}"
    training = {
        "generation": 1, "parameters": model.get_params(),
        "converged": bool(model.converged_), "n_iter": int(model.n_iter_),
        "lower_bound": float(model.lower_bound_),
        "mean_train_log_likelihood": float(model.score(real)),
        "status": "converged" if model.converged_ else "WARNING: EM did not converge; inspect before interpretation",
    }
    provenance = {
        "input_path": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "config_path": str(CONFIG), "config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "fit_seed": args.fit_seed, "sample_seed": args.sample_seed,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    }
    write_json(directories["models"] / f"{stem}_model.json", {
        "weights": model.weights_.tolist(), "means": model.means_.tolist(),
        "covariances": model.covariances_.tolist(), "training": training, "provenance": provenance,
    })
    np.savez_compressed(directories["datas"] / f"{stem}_generated.npz", X=generated,
                        fitted_component_ids=fitted_ids, eval_class_ids=generated_classes)
    np.savetxt(directories["datas"] / f"{stem}_generated.csv",
               np.column_stack((generated, fitted_ids, generated_classes)), delimiter=",",
               header="x,y,fitted_component_id,eval_class_id", comments="", fmt=["%.17g", "%.17g", "%d", "%d"])
    report = {
        "provenance": provenance, "training": training,
        "definitions": {
            "reference": "Original training data; this is an in-sample diagnostic, not a held-out estimate.",
            "fid_2d": "FID formula on raw (x,y), unbiased sample covariance ddof=1; no Inception network.",
            "precision": "Fraction of generated points inside union of initial-data k-NN balls.",
            "recall": "Fraction of initial points inside union of generated-data k-NN balls.",
            "k": args.k, "exclude_self": True,
            "classes": "Both sets assigned to nearest true center (Euclidean); ties use smallest index.",
            "per_class": "Recompute FID and k-NN support within each class; fixed k; counts <= k yield null support metric.",
            "undefined": "JSON null / blank CSV cell; empty generated class has recall=0 if initial class is nonempty.",
        },
        "overall": overall, "per_class": class_rows,
        "class_mass_total_variation": 0.5 * sum(abs(row["fraction_difference"]) for row in class_rows),
    }
    write_json(directories["metrics"] / f"{stem}_metrics.json", report)
    with (directories["metrics"] / f"{stem}_per_class.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    with (directories["metrics"] / f"{stem}_overall.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(overall))
        writer.writeheader()
        writer.writerow(overall)
    plot_comparison(real, generated, centers, overall, directories["figures"] / f"{stem}_scatter.png")
    plot_class_metrics(class_rows, args.k, directories["figures"] / f"{stem}_class_metrics.png")
    print(training["status"])
    print(json.dumps(overall, indent=2))
    print(f"Saved model, generated data, metrics and figures under {ROOT / 'outputs'}")

if __name__ == "__main__":
    main()
