"""Sample 2,000 independent observations from the configured true GMM.

Run from GAOJiashen with: uv run code/generate_initial_samples.py
Paths are anchored to this file, so outputs do not depend on the working directory.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gmm_experiment_common import load_gmm, sample_gmm, save_scatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "settings" / "gao_unbalanced_true_gmm.json"
N_SAMPLES = 2000

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Nonnegative random seed (default: 42)")
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed must be nonnegative")

    weights, means, covariances, config_sha256 = load_gmm(CONFIG_PATH)
    samples, component_ids = sample_gmm(weights, means, covariances, args.seed)
    data_dir = PROJECT_ROOT / "outputs" / "datas"
    figure_dir = PROJECT_ROOT / "outputs" / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = f"initial_samples_n{N_SAMPLES}_seed{args.seed}"

    # X contains coordinates only. Labels are ground truth for later evaluation,
    # not inputs to a fitted model. Component IDs follow the JSON's zero-based order.
    np.savez_compressed(data_dir / f"{stem}.npz", X=samples, component_ids=component_ids)
    np.savetxt(
        data_dir / f"{stem}.csv",
        np.column_stack((samples, component_ids)),
        delimiter=",", header="x,y,component_id", comments="", fmt=["%.17g", "%.17g", "%d"],
    )
    metadata = {
        "n_samples": N_SAMPLES,
        "seed": args.seed,
        "config_path": CONFIG_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "config_sha256": config_sha256,
        "numpy_version": np.__version__,
        "bit_generator": "PCG64",
        "component_index_base": 0,
        "weights": weights.tolist(),
        "component_counts": np.bincount(component_ids, minlength=len(weights)).tolist(),
        "sample_mean": samples.mean(axis=0).tolist(),
        "sample_covariance_ddof1": np.cov(samples, rowvar=False, ddof=1).tolist(),
    }
    (data_dir / f"{stem}_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    figure_path = figure_dir / f"{stem}.png"
    save_scatter(samples, means, args.seed, figure_path)
    print(f"Saved {N_SAMPLES} samples (NPZ, CSV and metadata) to: {data_dir}")
    print(f"Saved scatter plot to: {figure_path}")

if __name__ == "__main__":
    main()
