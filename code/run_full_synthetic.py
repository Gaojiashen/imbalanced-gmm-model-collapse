"""Full-synthetic GMM recursion, starting from the saved generation-1 model.

Run later from GAOJiashen (these commands have NOT been executed):
    uv run --locked --cache-dir .uv-cache code/run_full_synthetic.py --generations 20 --seeds 0
    uv run --locked --cache-dir .uv-cache code/run_full_synthetic.py --resume

Defaults: generations 1..2000 (1999 new fits), seeds 0..4, 2000 training
points/generation, and 2000 independent evaluation points/evaluation generation.
The original 2000 points are the ONLY fixed reference set. Evaluation is at
1, 2, 5, 10, every 20 generations, and the requested final generation.

Outputs: outputs/full_synthetic/{settings,seed_*/{models,datas,metrics},summary}
Figures: outputs/figures/full_synthetic/seed_* and summary.
Snapshots: 1,20,100,200,500,1000,2000 plus the requested final generation.

Reuse the driver, fitting, metrics and plots from gmm_experiment_common.py
in this same directory. FID uses raw 2D features;
precision/recall use fixed k=5 nearest-neighbor balls. Classes are assigned
to the nearest true center. Undefined class metrics remain null, never zero-
filled. No true points, historical samples, or evaluation points enter fits
after generation 1. Numerical failures stop that trajectory and are recorded;
ordinary nonconvergence is recorded without changing the experiment.

All five trajectories share M1: uncertainty is conditional on this saved model.
Independent, generation-addressed random streams make resumed training and
training under different evaluation schedules identical. A resume keeps the
same seeds, k, and evaluation interval; only the final generation may increase.
"""

from gmm_experiment_common import run_experiment as main


if __name__ == "__main__":
    main()
