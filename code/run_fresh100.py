"""100% fresh-data control: draw 2000 NEW true points each generation.

No synthetic points enter training. Each fit learns a fresh 25-component GMM;
the previous model is not used to initialize it. The saved generation-1 model
is shared with the 0%/10% conditions. Evaluation always uses the original 2000
points, with 2000 independently generated evaluation points per evaluation.
Defaults, metrics, snapshots, seeds, and resume behavior match run_fresh10.py.

Outputs: outputs/fresh100; figures: outputs/figures/fresh100.
Run later:
    uv run --locked --cache-dir .uv-cache code/run_fresh100.py --generations 20 --seeds 0
    uv run --locked --cache-dir .uv-cache code/run_fresh100.py --resume

CUDA PyTorch is managed by pyproject.toml/uv.lock. GPU run:
    uv run --locked --cache-dir .uv-cache code/run_fresh100.py --device cuda
--device auto defaults to CUDA when available, otherwise sklearn CPU.
The shared driver and CUDA fitter reside in gmm_experiment_common.py;
keep these project scripts together. This file does not run on import.
"""

from gmm_experiment_common import run_experiment


if __name__ == "__main__":
    run_experiment(fresh_fraction=1.0, condition="fresh100", default_device="auto")
