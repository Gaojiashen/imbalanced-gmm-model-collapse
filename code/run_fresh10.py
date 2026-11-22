"""Fixed 10% fresh-data experiment: 200 new true + 1800 synthetic points/fit.

The original 2000 points remain the ONLY evaluation reference. Fresh points
are drawn independently each generation from settings/gao_unbalanced_true_gmm.json,
not resampled from that reference. Defaults: 2000 generations, seeds 0..4,
25 full-covariance components, 3 starts, 300 EM steps, tol=1e-4, reg=1e-6.
Training, metrics, figures and resume logic reuse gmm_experiment_common.py.
Outputs: outputs/fresh10; figures: outputs/figures/fresh10.

Run later (no experiment was run while writing this file):
    uv run --locked --cache-dir .uv-cache code/run_fresh10.py --generations 20 --seeds 0
    uv run --locked --cache-dir .uv-cache code/run_fresh10.py --resume

--device auto (default): use CUDA if CUDA-enabled PyTorch is installed, else CPU.
--device cuda: require GPU; --device cpu: use the original sklearn CPU fit.
CUDA-enabled PyTorch is a locked Windows project dependency (official cu128 index).
Restore the environment with uv; no separate pip install or --inexact is needed:
    uv sync --locked --cache-dir .uv-cache
    uv run --locked --cache-dir .uv-cache code/run_fresh10.py --device cuda

GPU implementation: sklearn k-means initialization on CPU, batched float64 EM
for 3 independent starts on CUDA. This preserves the initialization method,
but CUDA roundoff can change fitted trajectories. Evaluate speed empirically;
this small 2D problem and consumer-GPU float64 need not outperform CPU.
For a backend-matched 0% control, the existing full-synthetic script also
accepts --device cuda. Do not resume an existing CPU trajectory on CUDA.
"""

from gmm_experiment_common import fit_cuda, run_experiment


if __name__ == "__main__":
    run_experiment(fresh_fraction=0.10, condition="fresh10", default_device="auto")
