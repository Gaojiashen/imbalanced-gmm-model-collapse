"""Shared GMM sampling, CPU/CUDA fitting, metrics, plots and experiment driver.

Import this module from the small experiment entry scripts. Importing never
starts training or writes experiment outputs. Keep the original grouped
initial sampler to preserve existing seed-42 samples, and the vectorized loop
sampler to preserve recursive trajectories. Training/evaluation random streams,
output paths, metric definitions and checkpoint schemas are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from scipy.special import logsumexp
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.mixture import GaussianMixture



ROOT = Path(__file__).resolve().parents[1]
INITIAL = ROOT / "outputs/datas/initial_samples_n2000_seed42.npz"
FIRST_MODEL = ROOT / "outputs/models/generation1_fit42_sample43_k5_model.json"
TRUE_CONFIG = ROOT / "settings/gao_unbalanced_true_gmm.json"
OUTPUT = ROOT / "outputs/full_synthetic"
FIGURES = ROOT / "outputs/figures/full_synthetic"
N = 2000
SNAPSHOTS = {1, 20, 100, 200, 500, 1000, 2000}
FIT = dict(n_components=25, covariance_type="full", n_init=3, init_params="kmeans",
           max_iter=300, tol=1e-4, reg_covar=1e-6)
METRIC_KEYS = ("fid_2d", "precision", "recall", "nll_initial", "total_variance",
               "mean_component_variance", "weighted_component_variance",
               "rare_mean_recall", "common_mean_recall", "rare_coverage", "common_coverage",
               "class_mass_total_variation")
CLASS_KEYS = ("fid_2d", "precision", "recall", "generated_fraction", "fraction_difference",
              "neighborhood_fraction")




PROJECT_ROOT = ROOT
CONFIG_PATH = CONFIG = TRUE_CONFIG
DEFAULT_INPUT = INITIAL
N_SAMPLES = N
BLUE, ORANGE = "#0072B2", "#D55E00"



def load_gmm(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load weights, means and full covariance matrices without altering them."""
    raw = path.read_bytes()
    config = json.loads(raw.decode("utf-8-sig"))
    weights = np.asarray(config["weights"], dtype=np.float64)
    means = np.asarray(config["means"], dtype=np.float64)
    covariances = np.asarray(config["covariances"], dtype=np.float64)
    k = config["n_components"]
    d = config["dimension"]
    if d != 2:
        raise ValueError("The scatter plot requires a two-dimensional GMM.")
    if weights.shape != (k,) or means.shape != (k, d):
        raise ValueError("Weights or means do not match the declared GMM dimensions.")
    if covariances.shape != (k, d, d):
        raise ValueError("Covariances do not match the declared GMM dimensions.")
    if not all(np.isfinite(a).all() for a in (weights, means, covariances)):
        raise ValueError("GMM parameters must be finite.")
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Weights must be nonnegative and sum to one.")
    if not np.allclose(covariances, covariances.swapaxes(1, 2), rtol=0, atol=1e-12):
        raise ValueError("Covariance matrices must be symmetric.")
    if np.any(np.linalg.eigvalsh(covariances) <= 0):
        raise ValueError("Covariance matrices must be positive definite.")
    return weights, means, covariances, hashlib.sha256(raw).hexdigest()

def sample_gmm(
    weights: np.ndarray, means: np.ndarray, covariances: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Choose each component independently, then sample its Gaussian.

    Component counts are random; expected counts are not imposed as quotas.
    Preserve the original sampled order rather than sorting by component.
    """
    rng = np.random.default_rng(seed)
    component_ids = rng.choice(len(weights), size=N_SAMPLES, p=weights)
    samples = np.empty((N_SAMPLES, means.shape[1]), dtype=np.float64)
    for component_id in range(len(weights)):
        mask = component_ids == component_id
        samples[mask] = rng.multivariate_normal(
            means[component_id], covariances[component_id], size=int(mask.sum()),
            check_valid="raise",
        )
    return samples, component_ids

def save_scatter(samples: np.ndarray, means: np.ndarray, seed: int, path: Path) -> None:
    """Plot all observations on equal spatial scales, with true centers marked."""
    with plt.rc_context({"font.size": 11, "axes.titlesize": 14}):
        fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
        ax.scatter(
            samples[:, 0], samples[:, 1], s=9, alpha=0.6,
            color="#0072B2", edgecolors="none", label="Initial samples", zorder=2,
        )
        ax.scatter(
            means[:, 0], means[:, 1], s=35, marker="x", linewidths=1.2,
            color="#222222", label="True component centers", zorder=3,
        )
        ax.set_title(f"Unbalanced GMM: {len(samples):,} initial samples\nSeed = {seed}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.2, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncols=2, frameon=False)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

def read_input(path: Path) -> np.ndarray:
    """Only coordinates are used for unsupervised training."""
    with np.load(path, allow_pickle=False) as archive:
        x = np.asarray(archive["X"], dtype=np.float64)
    if x.shape != (N_SAMPLES, 2) or not np.isfinite(x).all():
        raise ValueError("Input X must contain exactly 2,000 finite two-dimensional points.")
    return x

def assign_classes(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Use a fixed common partition; never interpret fitted component IDs as classes."""
    return cdist(x, centers, metric="sqeuclidean").argmin(axis=1)

def frechet_distance(real: np.ndarray, generated: np.ndarray) -> float | None:
    """FID formula on raw 2D features, with unbiased sample covariance (ddof=1).

    Evaluate the covariance term through a symmetric positive-semidefinite
    sandwich to avoid complex results from sqrtm(C_real @ C_generated).
    """
    if min(len(real), len(generated)) < 2:
        return None
    mean_difference = real.mean(axis=0) - generated.mean(axis=0)
    cr = np.cov(real, rowvar=False, ddof=1)
    cg = np.cov(generated, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh((cr + cr.T) / 2)
    root_real = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
    sandwich = root_real @ cg @ root_real
    cross_trace = np.sqrt(
        np.maximum(np.linalg.eigvalsh((sandwich + sandwich.T) / 2), 0)
    ).sum()
    value = mean_difference @ mean_difference + np.trace(cr) + np.trace(cg) - 2 * cross_trace
    # Only numerical roundoff can make this nonnegative distance slightly negative.
    return float(max(value, 0.0))

def manifold_coverage(query: np.ndarray, reference: np.ndarray, k: int) -> float | None:
    """Fraction of query points in the union of reference k-NN balls.

    Exclude each reference point itself from its neighbors. Keep k fixed rather
    than silently changing the metric for small classes. A class with no
    reference support has zero coverage; an empty query has no defined fraction.
    """
    if len(query) == 0:
        return None
    if len(reference) == 0:
        return 0.0
    if len(reference) <= k:
        return None
    within = cdist(reference, reference)
    np.fill_diagonal(within, np.inf)
    radii = np.partition(within, kth=k - 1, axis=1)[:, k - 1]
    covered = 0
    for start in range(0, len(query), 256):
        distances = cdist(query[start : start + 256], reference)
        covered += int((distances <= radii[None, :]).any(axis=1).sum())
    return covered / len(query)

def evaluate(real: np.ndarray, generated: np.ndarray, k: int) -> dict:
    notes = []
    if min(len(real), len(generated)) < 2:
        notes.append("FID undefined: each set needs at least two points.")
    if len(real) == 0:
        notes.append("No initial samples in this class; recall is undefined.")
    elif len(real) <= k:
        notes.append("Too few initial samples for fixed-k precision support.")
    if len(generated) == 0:
        notes.append("No generated samples: precision undefined; recall zero if initial set nonempty.")
    elif len(generated) <= k:
        notes.append("Too few generated samples for fixed-k recall support.")
    return {
        "n_initial": len(real),
        "n_generated": len(generated),
        "fid_2d": frechet_distance(real, generated),
        "precision": manifold_coverage(generated, real, k),
        "recall": manifold_coverage(real, generated, k),
        "status": "ok" if not notes else " | ".join(notes),
    }

def draw_samples(model: GaussianMixture, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Use a separate RNG for generation, preserving iid draw order."""
    rng = np.random.default_rng(seed)
    ids = rng.choice(model.n_components, size=N_SAMPLES, p=model.weights_)
    noise = rng.standard_normal((N_SAMPLES, 2))
    factors = np.linalg.cholesky(model.covariances_)
    x = model.means_[ids] + np.einsum("nij,nj->ni", factors[ids], noise)
    return x, ids

def plot_comparison(
    real: np.ndarray, generated: np.ndarray, centers: np.ndarray, overall: dict, path: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), layout="constrained")
    combined = np.vstack((real, generated, centers))
    low, high = combined.min(axis=0) - 0.15, combined.max(axis=0) + 0.15
    for ax, title in zip(axes, ("Initial data (n=2,000)", "Generation 1 (n=2,000)", "Overlay")):
        ax.set(title=title, xlabel="x", ylabel="y", xlim=(low[0], high[0]), ylim=(low[1], high[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in (axes[0], axes[2]):
        ax.scatter(real[:, 0], real[:, 1], s=9, alpha=0.5, color=BLUE, label="Initial", edgecolors="none")
    for ax in (axes[1], axes[2]):
        ax.scatter(generated[:, 0], generated[:, 1], s=12, alpha=0.5, color=ORANGE, marker="^",
                   label="Generated", edgecolors="none")
    for ax in axes:
        ax.scatter(centers[:, 0], centers[:, 1], s=24, marker="x", color="black", label="True centers")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False, ncols=2, fontsize=9)
    fig.suptitle(
        f"Generation 1 vs. initial data | FID (2D) = {overall['fid_2d']:.5f} | "
        f"Precision = {overall['precision']:.3f} | Recall = {overall['recall']:.3f}"
    )
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_class_metrics(rows: list[dict], k: int, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, layout="constrained")
    ids = np.arange(len(rows))
    colors = [ORANGE if row["is_rare"] else BLUE for row in rows]
    for ax, key, label in zip(axes, ("fid_2d", "precision", "recall"), ("FID (2D)", "Precision", "Recall")):
        values = [np.nan if row[key] is None else row[key] for row in rows]
        ax.bar(ids, values, color=colors)
        for i, value in enumerate(values):
            if np.isnan(value):
                ax.text(i, 0.02, "N/A", rotation=90, ha="center", transform=ax.get_xaxis_transform())
        ax.set_ylabel(label)
        if key != "fid_2d":
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch

    axes[0].legend(handles=[Patch(color=BLUE, label="Common class"), Patch(color=ORANGE, label="Rare class")],
                   frameon=False, loc="upper right")
    axes[-1].set_xticks(ids, [f"{i}*" if row["is_rare"] else str(i) for i, row in enumerate(rows)])
    axes[-1].set_xlabel("Class ID (zero-based); * marks rare classes")
    fig.suptitle(f"Per-class metrics in 2D | fixed k = {k} | N/A denotes undefined metrics")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def create_cpu_model(seed, n_components=25):
    """Use the same EM settings for the first fit and recursive CPU fits."""
    return GaussianMixture(**{**FIT, "n_components": n_components}, random_state=seed)


def fit_cuda(x, seed, args):
    """Fit full-covariance 2D GMM using three GPU-batched EM starts.

    CPU k-means uses the same n_init=1 and advancing RandomState as sklearn's
    GaussianMixture. EM follows E/M updates and lower-bound convergence; all
    parameters are relearned every generation, with no warm start or true labels.
    """
    import torch

    settings = FIT
    starts, components = settings["n_init"], settings["n_components"]
    rng = np.random.RandomState(seed)
    labels = np.stack([KMeans(n_clusters=components, n_init=1, random_state=rng).fit(x).labels_
                       for _ in range(starts)])
    with torch.no_grad():
        data = torch.as_tensor(x, dtype=torch.float64, device="cuda")
        responsibilities = torch.nn.functional.one_hot(
            torch.as_tensor(labels, dtype=torch.int64, device="cuda"), num_classes=components
        ).to(torch.float64)
        identity = torch.eye(2, dtype=data.dtype, device=data.device)

        def maximize(resp):
            counts = resp.sum(dim=1) + 10 * torch.finfo(data.dtype).eps
            weights = counts / counts.sum(dim=1, keepdim=True)
            means = torch.einsum("rnk,nd->rkd", resp, data) / counts[..., None]
            diff = data[None, :, None, :] - means[:, None, :, :]
            cov = torch.einsum("rnk,rnki,rnkj->rkij", resp, diff, diff) / counts[..., None, None]
            return weights, means, cov + settings["reg_covar"] * identity

        def log_density(weights, means, cov):
            # Explicit 2x2 solve avoids launching many small matrix solvers.
            diff = data[None, :, None, :] - means[:, None, :, :]
            a, b, c = cov[..., 0, 0][:, None], cov[..., 0, 1][:, None], cov[..., 1, 1][:, None]
            determinant = a * c - b.square()
            dx, dy = diff[..., 0], diff[..., 1]
            quadratic = (c * dx.square() - 2 * b * dx * dy + a * dy.square()) / determinant
            return weights[:, None].log() - np.log(2 * np.pi) - 0.5 * (determinant.log() + quadratic)

        weights, means, cov = maximize(responsibilities)
        previous = torch.full((starts,), -torch.inf, dtype=data.dtype, device=data.device)
        converged = torch.zeros(starts, dtype=torch.bool, device=data.device)
        iterations = torch.full((starts,), settings["max_iter"], dtype=torch.int64, device=data.device)
        for iteration in range(1, settings["max_iter"] + 1):
            densities = log_density(weights, means, cov)
            normalizer = densities.logsumexp(dim=-1)
            lower_bound = normalizer.mean(dim=1)
            updated = maximize((densities - normalizer[..., None]).exp())
            active = ~converged
            weights = torch.where(active[:, None], updated[0], weights)
            means = torch.where(active[:, None, None], updated[1], means)
            cov = torch.where(active[:, None, None, None], updated[2], cov)
            newly_converged = active & ((lower_bound - previous).abs() < settings["tol"])
            iterations = torch.where(newly_converged, iteration, iterations)
            converged |= newly_converged
            previous = lower_bound
            if bool(converged.all()):
                break
        likelihood = log_density(weights, means, cov).logsumexp(dim=-1).mean(dim=1)
        best = int(likelihood.argmax().item())
        if not bool(torch.isfinite(likelihood).all()):
            raise FloatingPointError("Nonfinite CUDA EM likelihood.")
        p = {"weights": weights[best].cpu().numpy(), "means": means[best].cpu().numpy(),
             "covariances": cov[best].cpu().numpy()}
        training = {"converged": bool(converged[best].item()), "n_iter": int(iterations[best].item()),
                    "mean_train_log_likelihood": float(likelihood[best].item())}
        return p, training

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))

def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

def parameters(saved):
    return {key: np.asarray(saved[key], dtype=np.float64)
            for key in ("weights", "means", "covariances")}

def random_stream(seed, generation, purpose):
    # purpose: 0=training sample, 1=EM initialization, 2=evaluation sample.
    return np.random.default_rng(np.random.SeedSequence([seed, generation, purpose]))

def sample(p, rng, size=N, covariance_scale=1.0):
    labels = rng.choice(len(p["weights"]), size=size, p=p["weights"])
    factors = np.linalg.cholesky(p["covariances"])
    if covariance_scale != 1.0:
        factors = factors * np.sqrt(covariance_scale)
    x = p["means"][labels] + np.einsum(
        "nij,nj->ni", factors[labels], rng.standard_normal((size, 2)))
    return x, labels

def training_sample(p, seed, generation, args):
    recipe = getattr(args, "recipe", None)
    if recipe:
        synthetic, _ = sample(p, random_stream(seed, generation, 0),
                              recipe["n_synthetic"], recipe.get("lambda", 1.0))
        if recipe["mode"] == "fixed":
            real = args.fixed_real
        else:
            real, _ = sample(args.true_parameters, random_stream(seed, generation, 3), recipe["n_fresh"])
        return np.vstack((real, synthetic))
    fresh_count = round(N * args.fresh_fraction)
    if fresh_count == 0:
        return sample(p, random_stream(seed, generation, 0))[0]
    fresh, _ = sample(args.true_parameters, random_stream(seed, generation, 3), fresh_count)
    if fresh_count == N:
        return fresh
    synthetic, _ = sample(p, random_stream(seed, generation, 0), N - fresh_count)
    mixed = np.vstack((synthetic, fresh))
    return mixed[random_stream(seed, generation, 4).permutation(N)]

def fit_cpu(x, seed, args):
    model = create_cpu_model(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(x)
    return ({"weights": model.weights_, "means": model.means_, "covariances": model.covariances_},
            {"converged": bool(model.converged_), "n_iter": int(model.n_iter_),
             "mean_train_log_likelihood": float(model.score(x))})

def model_statistics(p):
    weights, means, covariances = p["weights"], p["means"], p["covariances"]
    mean = weights @ means
    traces = np.trace(covariances, axis1=1, axis2=2)
    return {
        "total_variance": float(weights @ (traces + ((means - mean) ** 2).sum(axis=1))),
        "mean_component_variance": float(traces.mean()),
        "weighted_component_variance": float(weights @ traces),
        "min_covariance_eigenvalue": float(np.linalg.eigvalsh(covariances).min()),
    }

def nll(p, x):
    # Score the original reference without needing to reconstruct a sklearn estimator.
    densities = []
    for weight, mean, covariance in zip(p["weights"], p["means"], p["covariances"]):
        factor = np.linalg.cholesky(covariance)
        z = np.linalg.solve(factor, (x - mean).T)
        densities.append(np.log(weight) - np.log(np.diag(factor)).sum()
                         - np.log(2 * np.pi) - 0.5 * (z * z).sum(axis=0))
    return float(-logsumexp(np.column_stack(densities), axis=1).mean())

def mean_defined(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None

def assess(p, generated, real, real_labels, centers, rare_ids, k):
    labels = assign_classes(generated, centers)
    counts_in_radius = (cdist(generated, centers) <= 0.4).sum(axis=0)
    per_class = []
    for c in range(len(centers)):
        row = {"class_id": c, "center_x": float(centers[c, 0]), "center_y": float(centers[c, 1]),
               "is_rare": c in rare_ids,
               **evaluate(real[real_labels == c], generated[labels == c], k)}
        row.update(initial_fraction=row["n_initial"] / N,
                   generated_fraction=row["n_generated"] / N,
                   fraction_difference=(row["n_generated"] - row["n_initial"]) / N,
                   neighborhood_count=int(counts_in_radius[c]),
                   neighborhood_fraction=float(counts_in_radius[c] / N))
        per_class.append(row)
    rare = [row for row in per_class if row["is_rare"]]
    common = [row for row in per_class if not row["is_rare"]]
    overall = {
        **evaluate(real, generated, k), **model_statistics(p), "nll_initial": nll(p, real),
        "class_mass_total_variation": 0.5 * sum(abs(row["fraction_difference"]) for row in per_class),
    }
    for name, group in (("rare", rare), ("common", common)):
        overall[f"{name}_mean_recall"] = mean_defined([row["recall"] for row in group])
        overall[f"{name}_recall_defined_classes"] = sum(row["recall"] is not None for row in group)
        overall[f"{name}_covered_modes"] = sum(row["neighborhood_count"] > 0 for row in group)
        overall[f"{name}_coverage"] = overall[f"{name}_covered_modes"] / len(group)
    return overall, per_class, labels

def save_checkpoint(directory, generation, p, training):
    payload = {key: value.tolist() for key, value in p.items()}
    payload.update(generation=generation, training=training)
    write_json(directory / f"generation_{generation:04d}.json", payload)
    # Replace only after the complete new checkpoint has been written.
    temporary = directory / "latest.tmp"
    write_json(temporary, payload)
    temporary.replace(directory / "latest.json")

def record_evaluation(seed, generation, p, training_x, training, context, args, directory):
    real, real_labels, centers, rare_ids = context
    # All trajectories have exactly the same generation-1 evaluation baseline.
    rng = random_stream(0 if generation == 1 else seed, generation, 2)
    generated, fitted_ids = sample(p, rng)
    overall, classes, labels = assess(p, generated, real, real_labels, centers, rare_ids, args.k)
    if getattr(args, "recipe", None):
        overall["sliced_w2_squared"] = sliced_w2_squared(real, generated)
    report = {"seed": seed, "generation": generation, "training": training,
              "overall": overall, "per_class": classes}
    write_json(directory / "metrics" / f"generation_{generation:04d}.json", report)
    if generation in SNAPSHOTS or generation == args.generations:
        np.savez_compressed(directory / "datas" / f"generation_{generation:04d}.npz",
                            X=generated, X_train=training_x, fitted_component_ids=fitted_ids,
                            eval_class_ids=labels)
    save_checkpoint(directory / "models", generation, p, training)
    print(f"seed={seed} generation={generation} FID={overall['fid_2d']:.5g} "
          f"P={overall['precision']:.3f} R={overall['recall']:.3f} "
          f"rare={overall['rare_covered_modes']}/5 common={overall['common_covered_modes']}/20",
          flush=True)

def run_seed(seed, first, context, args, fit_function):
    directory = OUTPUT / f"seed_{seed}"
    checkpoint = directory / "models/latest.json"
    history_path = directory / "metrics/training.csv"
    if args.resume and checkpoint.exists():
        saved = read_json(checkpoint)
        p, start = parameters(saved), saved["generation"]
        # A stopped run may have logged fits beyond its latest saved checkpoint.
        with history_path.open(encoding="utf-8-sig", newline="") as f:
            history = [row for row in csv.DictReader(f) if int(row["generation"]) <= start]
        write_csv(history_path, history)
    else:
        directory.mkdir(parents=True, exist_ok=False)
        for name in ("models", "datas", "metrics"):
            (directory / name).mkdir()
        p, start = parameters(first), 1
        initial_x = context[0]
        recipe = getattr(args, "recipe", None)
        if recipe and recipe.get("initial_size"):
            pool, _ = sample(args.true_parameters, random_stream(seed, 1, 5), recipe["initial_pool_size"])
            initial_x = pool[:recipe["initial_size"]]
            initial_seed = int(random_stream(seed, 1, 1).integers(0, 2**32))
            p, training = fit_function(initial_x, initial_seed, args)
            np.savez_compressed(directory / "datas/initial_training.npz", X=initial_x)
        else:
            training = {"converged": first["training"]["converged"],
                        "n_iter": first["training"]["n_iter"],
                        "mean_train_log_likelihood": first["training"]["mean_train_log_likelihood"]}
        write_csv(history_path, [{"generation": 1, **training, **model_statistics(p)}])
        record_evaluation(seed, 1, p, initial_x, training, context, args, directory)
    failure_path = directory / "failure.json"
    for generation in range(start + 1, args.generations + 1):
        try:
            training_x = training_sample(p, seed, generation, args)
            fit_seed = int(random_stream(seed, generation, 1).integers(0, 2**32))
            p, training = fit_function(training_x, fit_seed, args)
            stats = model_statistics(p)
            if not all(np.isfinite(a).all() for a in p.values()) or stats["min_covariance_eigenvalue"] <= 0:
                raise FloatingPointError("Nonfinite model parameters or nonpositive covariance.")
            with history_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=["generation", *training, *stats]).writerow(
                    {"generation": generation, **training, **stats})
            if (generation in {2, 5, 10} or generation % args.eval_every == 0
                    or generation in SNAPSHOTS or generation == args.generations):
                record_evaluation(seed, generation, p, training_x, training, context, args, directory)
        except (np.linalg.LinAlgError, FloatingPointError, ValueError) as error:
            write_json(failure_path, {"seed": seed, "generation": generation,
                                     "error_type": type(error).__name__, "error": str(error),
                                     "last_checkpoint": str(checkpoint)})
            print(f"Stopped seed {seed} at generation {generation}: {error}", flush=True)
            return
    if failure_path.exists():
        failure_path.unlink()

def load_reports(seed):
    return [read_json(path) for path in sorted((OUTPUT / f"seed_{seed}/metrics").glob("generation_*.json"))]

def export_tables(seed, reports):
    directory = OUTPUT / f"seed_{seed}/metrics"
    baseline = reports[0]["overall"]
    overall, classes = [], []
    for report in reports:
        row = {"seed": seed, "generation": report["generation"], **report["overall"]}
        for key in ("fid_2d", "precision", "recall", "nll_initial", "total_variance"):
            row[f"delta_{key}_vs_generation1"] = row[key] - baseline[key]
        overall.append(row)
        classes.extend({"seed": seed, "generation": report["generation"], **c} for c in report["per_class"])
    write_csv(directory / "overall.csv", overall)
    write_csv(directory / "per_class.csv", classes)

def plot_trajectory(seed, reports):
    directory = FIGURES / f"seed_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    generations = [r["generation"] for r in reports]
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), layout="constrained")
    for ax, key, label in zip(axes.flat, METRIC_KEYS[:5],
                             ("FID (2D)", "Precision", "Recall", "NLL on initial data", "Total variance")):
        ax.plot(generations, [r["overall"][key] for r in reports], color="#0072B2")
        ax.set_ylabel(label)
        if key in ("precision", "recall"):
            ax.set_ylim(0, 1.03)
    for group, color in (("rare", "#D55E00"), ("common", "#0072B2")):
        axes.flat[-1].plot(generations, [r["overall"][f"{group}_coverage"] for r in reports],
                           color=color, label=group)
    axes.flat[-1].set(ylabel="Mode coverage within radius 0.4", ylim=(0, 1.03))
    axes.flat[-1].legend()
    for ax in axes.flat:
        ax.set_xlabel("Generation")
        ax.grid(alpha=0.2)
    fig.suptitle(f"{OUTPUT.name} | seed {seed} | reference: original 2,000 points")
    fig.savefig(directory / "trajectory.png", dpi=200)
    plt.close(fig)

    # Use real generation coordinates, including the more frequent early evaluations.
    g = np.asarray(generations)
    edges = np.r_[g[0] - 0.5, (g[:-1] + g[1:]) / 2, g[-1] + 0.5]
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, layout="constrained")
    for ax, key, title in zip(axes, ("recall", "generated_fraction", "fid_2d"),
                              ("Per-class recall", "Generated class fraction", "Per-class FID (2D)")):
        values = np.array([[np.nan if c[key] is None else c[key] for c in r["per_class"]]
                           for r in reports], dtype=float).T
        image = ax.pcolormesh(edges, np.arange(26) - 0.5, np.ma.masked_invalid(values),
                              cmap="viridis", vmin=0, vmax=1 if key == "recall" else None)
        ax.set(title=title, ylabel="Class ID")
        ax.set_yticks(range(25), [f"{i}*" if reports[0]["per_class"][i]["is_rare"] else str(i)
                                 for i in range(25)], fontsize=8)
        fig.colorbar(image, ax=ax)
    axes[-1].set_xlabel("Generation (* marks rare classes; blank cells are undefined)")
    fig.savefig(directory / "class_heatmaps.png", dpi=200)
    plt.close(fig)

def plot_snapshots(seeds, real, centers):
    paths = [(seed, path) for seed in seeds
             for path in sorted((OUTPUT / f"seed_{seed}/datas").glob("generation_*.npz"))]
    low, high = real.min(axis=0), real.max(axis=0)
    for _, path in paths:
        with np.load(path) as data:
            low = np.minimum(low, data["X"].min(axis=0))
            high = np.maximum(high, data["X"].max(axis=0))
    low, high = low - 0.15, high + 0.15
    for seed, path in paths:
        with np.load(path) as data:
            generated = data["X"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), layout="constrained")
        for ax, title in zip(axes, ("Initial data (n=2,000)", "Generated (n=2,000)", "Overlay")):
            ax.set(title=title, xlabel="x", ylabel="y", xlim=(low[0], high[0]), ylim=(low[1], high[1]))
            ax.set_aspect("equal", adjustable="box")
        for ax in (axes[0], axes[2]):
            ax.scatter(real[:, 0], real[:, 1], color="#0072B2", s=8, alpha=0.5,
                       edgecolors="none", label="Initial")
        for ax in (axes[1], axes[2]):
            ax.scatter(generated[:, 0], generated[:, 1], color="#D55E00", marker="^", s=11,
                       alpha=0.5, edgecolors="none", label="Generated")
        for ax in axes:
            ax.scatter(centers[:, 0], centers[:, 1], color="black", marker="x", s=20, label="True centers")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2, fontsize=8, frameon=False)
        fig.suptitle(f"Seed {seed} | {path.stem.replace('_', ' ')} | common limits across all snapshots")
        fig.savefig(FIGURES / f"seed_{seed}" / f"{path.stem}_scatter.png", dpi=200)
        plt.close(fig)

def summarize(all_reports):
    directory = OUTPUT / "summary"
    directory.mkdir(exist_ok=True)
    generations = sorted({r["generation"] for reports in all_reports.values() for r in reports})
    indexed = {(seed, r["generation"]): r for seed, reports in all_reports.items() for r in reports}
    rows, class_rows = [], []
    for generation in generations:
        available = [indexed[seed, generation] for seed in all_reports if (seed, generation) in indexed]
        for class_id, keys, destination in [(None, METRIC_KEYS, rows)] + [(c, CLASS_KEYS, class_rows) for c in range(25)]:
            for key in keys:
                values = [(r["overall"] if class_id is None else r["per_class"][class_id])[key] for r in available]
                values = [v for v in values if v is not None]
                result = {"generation": generation}
                if class_id is not None:
                    result["class_id"] = class_id
                result.update(metric=key, n_valid=len(values),
                              mean=float(np.mean(values)) if values else None,
                              std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                              minimum=float(np.min(values)) if values else None,
                              maximum=float(np.max(values)) if values else None)
                destination.append(result)
    write_csv(directory / "overall_across_seeds.csv", rows)
    write_csv(directory / "per_class_across_seeds.csv", class_rows)
    final = []
    for seed, reports in all_reports.items():
        with (OUTPUT / f"seed_{seed}/metrics/training.csv").open(encoding="utf-8-sig", newline="") as f:
            training = list(csv.DictReader(f))
        final.append({"seed": seed, "last_evaluated_generation": reports[-1]["generation"],
                      "initial": reports[0]["overall"], "final": reports[-1]["overall"],
                      "nonconverged_generations": [int(r["generation"]) for r in training if r["converged"] == "False"],
                      "failure": read_json(OUTPUT / f"seed_{seed}/failure.json")
                      if (OUTPUT / f"seed_{seed}/failure.json").exists() else None})
    write_json(directory / "final.json", {"reference": str(INITIAL), "trajectories": final})
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), layout="constrained")
    keys = ("fid_2d", "precision", "recall", "nll_initial", "total_variance",
            "rare_mean_recall", "common_mean_recall", "rare_coverage", "common_coverage")
    for ax, key in zip(axes.flat, keys):
        for seed, reports in all_reports.items():
            ax.plot([r["generation"] for r in reports],
                    [r["overall"][key] for r in reports], alpha=0.35, linewidth=1, label=f"seed {seed}")
        selected = [r for r in rows if r["metric"] == key and r["n_valid"]]
        ax.plot([r["generation"] for r in selected], [r["mean"] for r in selected],
                color="black", linewidth=2, label="Mean")
        ax.fill_between([r["generation"] for r in selected], [r["minimum"] for r in selected],
                        [r["maximum"] for r in selected], color="gray", alpha=0.15, label="Min-max")
        ax.set(title=key.replace("_", " "), xlabel="Generation")
        if key in ("precision", "recall", "rare_mean_recall", "common_mean_recall", "rare_coverage", "common_coverage"):
            ax.set_ylim(0, 1.03)
        ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{OUTPUT.name} | individual trajectories, mean and range")
    (FIGURES / "summary").mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "summary/trajectories.png", dpi=200)
    plt.close(fig)

def run_experiment(fresh_fraction=0.0, condition="full_synthetic", default_device="cpu", *, argv=None, recipe=None):
    global OUTPUT, FIGURES
    OUTPUT = ROOT / "outputs" / condition
    FIGURES = ROOT / "outputs/figures" / condition
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=2000, help="Final generation, including existing M1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=default_device)
    args = parser.parse_args(argv)
    if args.generations < 1 or args.eval_every < 1 or not 1 <= args.k < N or min(args.seeds) < 0:
        parser.error("Use positive generations/eval-every, 1 <= k < 2000, and nonnegative seeds.")
    args.seeds = list(dict.fromkeys(args.seeds))
    requested = vars(args).copy()
    args.recipe = recipe
    args.fresh_fraction = fresh_fraction
    fit_function = fit_cpu
    if args.device != "cpu":
        try:
            import torch
        except ImportError:
            if args.device == "cuda":
                parser.error("CUDA requires CUDA-enabled PyTorch; see run_fresh10.py for installation commands.")
            args.device = "cpu"
        else:
            if torch.cuda.is_available():
                fit_function, args.device = fit_cuda, "cuda"
            elif args.device == "cuda":
                parser.error("PyTorch cannot access CUDA; install a CUDA-enabled build or use --device cpu.")
            else:
                args.device = "cpu"
    composition = str(recipe) if recipe else f"fresh={round(N * fresh_fraction)}/{N}"
    print(f"condition={condition} device={args.device} {composition}", flush=True)
    first, config = read_json(FIRST_MODEL), read_json(TRUE_CONFIG)
    args.true_parameters = parameters(config)
    with np.load(INITIAL) as data:
        real = np.asarray(data["X"], dtype=np.float64)
    args.fixed_real = real
    centers = np.asarray(config["means"], dtype=np.float64)
    rare_ids = set(config["rare_component_indices_zero_based"])
    context = real, assign_classes(real, centers), centers, rare_ids
    settings_dir = OUTPUT / "settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    settings_path = settings_dir / "experiment.json"
    if settings_path.exists() and not args.resume:
        parser.error("Results already exist; use --resume to continue without replacing the experiment settings.")
    if settings_path.exists() and args.resume:
        saved = read_json(settings_path)
        if saved["k"] != args.k or saved["eval_every"] != args.eval_every:
            parser.error("Resume with the original k and eval-every settings.")
        if saved.get("fresh_fraction", 0) != fresh_fraction or saved.get("device", "cpu") != args.device:
            parser.error("Resume with the original fresh fraction and CPU/CUDA backend.")
        if saved.get("recipe") != recipe:
            parser.error("Resume with the original training recipe.")
        for seed in args.seeds:
            path = OUTPUT / f"seed_{seed}/models/latest.json"
            if path.exists() and read_json(path)["generation"] > args.generations:
                parser.error("The requested final generation precedes a saved checkpoint.")
    else:
        write_json(settings_path, {
            "initial_data": str(INITIAL), "first_model": str(FIRST_MODEL), "true_config": str(TRUE_CONFIG),
            "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (INITIAL, FIRST_MODEL, TRUE_CONFIG)},
            "fit": FIT,
            "training_samples": (recipe["n_synthetic"] + (N if recipe["mode"] == "fixed" else recipe["n_fresh"])) if recipe else N,
            "evaluation_samples": N, "fresh_fraction": fresh_fraction,
            "device": args.device, "dtype": "float64",
            "recipe": recipe,
            "torch_version": torch.__version__ if args.device == "cuda" else None,
            "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
            "k": args.k, "eval_every": args.eval_every, "coverage_radius": 0.4,
            "classes": "Nearest true center, same rule for initial and generated points",
            "reference": "Original 2000 points only; no new true reference sample",
            "evaluation": "Independent of training; generation-1 evaluation shared across seeds",
            "rng": "SeedSequence([trajectory_seed, generation, purpose]); 0=synthetic, 1=fit, 2=eval, 3=fresh, 4=shuffle",
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        })
    # Each invocation is retained, including short runs and later continuations.
    invocation = settings_dir / f"run_to_{args.generations}_seeds_{'_'.join(map(str, args.seeds))}.json"
    write_json(invocation, {**requested, "fresh_fraction": fresh_fraction, "resolved_device": args.device})
    all_reports = {}
    for seed in args.seeds:
        run_seed(seed, first, context, args, fit_function)
        reports = load_reports(seed)
        export_tables(seed, reports)
        plot_trajectory(seed, reports)
        all_reports[seed] = reports
    summarize(all_reports)
    plot_snapshots(args.seeds, real, centers)
    print(f"Saved results to {OUTPUT}; figures to {FIGURES}")


def sliced_w2_squared(real, generated, directions=128):
    """Empirical sliced W2 squared, equal-size samples, fixed 2D directions.

    This is not exact multivariate W2 or a population-level GMM distance.
    """
    angles = np.arange(directions) * np.pi / directions
    vectors = np.column_stack((np.cos(angles), np.sin(angles)))
    return float(np.mean((np.sort(real @ vectors.T, axis=0)
                          - np.sort(generated @ vectors.T, axis=0)) ** 2))


def match_effective_size(sizes, errors, target):
    """Unique local interpolation in log(sample size); never extrapolate.

    Nonmonotone calibration can give several crossings: return ambiguous
    rather than imposing a decreasing curve that the observations do not show.
    """
    sizes, errors = np.asarray(sizes, float), np.asarray(errors, float)
    candidates = [float(n) for n, error in zip(sizes, errors) if np.isclose(error, target, rtol=1e-10, atol=1e-12)]
    for a, b, ea, eb in zip(sizes[:-1], sizes[1:], errors[:-1], errors[1:]):
        if min(ea, eb) < target < max(ea, eb):
            fraction = (target - ea) / (eb - ea)
            candidates.append(float(np.exp(np.log(a) + fraction * np.log(b / a))))
    candidates = np.unique(np.round(candidates, 8)).tolist()
    if len(candidates) == 1:
        return {"n_effective": candidates[0], "status": "interpolated", "candidates": candidates}
    if candidates:
        return {"n_effective": None, "status": "ambiguous", "candidates": candidates}
    return {"n_effective": None, "status": "outside_calibration_range", "candidates": []}


def suite_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--generations", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser


def run_suite_case(condition, recipe, args):
    argv = ["--generations", str(args.generations), "--seeds", *map(str, args.seeds),
            "--eval-every", str(args.eval_every), "--k", str(args.k), "--device", args.device]
    if args.resume:
        argv.append("--resume")
    fraction = recipe.get("n_fresh", 0) / (recipe.get("n_fresh", 0) + recipe["n_synthetic"]) if recipe["mode"] == "fresh" else 0.0
    run_experiment(fresh_fraction=fraction, condition=condition, default_device="cuda", argv=argv, recipe=recipe)


def compare_suite(name, cases, seeds, tail_window=500):
    """Per-seed summaries and cross-condition plots, using saved evaluations."""
    output, figures = ROOT / "outputs" / name, ROOT / "outputs/figures" / name
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for label, condition in cases.items():
        curves[label] = []
        for seed in seeds:
            paths = sorted((ROOT / "outputs" / condition / f"seed_{seed}/metrics").glob("generation_*.json"))
            reports = [read_json(path) for path in paths]
            if not reports:
                continue
            curves[label].append(reports)
            last = reports[-1]["generation"]
            tail = [r for r in reports if r["generation"] > last - tail_window]
            row = {"condition": label, "seed": seed, "last_generation": last,
                   "tail_start": tail[0]["generation"], "tail_evaluations": len(tail)}
            for key in (*METRIC_KEYS, "sliced_w2_squared"):
                values = [r["overall"].get(key) for r in tail]
                row[key] = mean_defined(values)
            rows.append(row)
    write_csv(output / "comparison_tail_per_seed.csv", rows)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    for ax, key in zip(axes.flat, ("fid_2d", "recall", "precision", "rare_mean_recall", "total_variance", "sliced_w2_squared")):
        for label, reports_list in curves.items():
            indexed = {}
            for reports in reports_list:
                for r in reports:
                    value = r["overall"].get(key)
                    if value is not None:
                        indexed.setdefault(r["generation"], []).append(value)
            generations = sorted(indexed)
            ax.plot(generations, [np.mean(indexed[g]) for g in generations], label=label)
            ax.fill_between(generations, [min(indexed[g]) for g in generations],
                            [max(indexed[g]) for g in generations], alpha=.1)
        ax.set(title=key.replace("_", " "), xlabel="Generation")
        ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{name}: seed mean and range | fixed original reference")
    fig.savefig(figures / "condition_comparison.png", dpi=200)
    plt.close(fig)
    return rows
