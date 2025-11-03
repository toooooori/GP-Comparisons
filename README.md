# GP-Comparsion

This repository contains code for comparing different shape-constrained Gaussian process (GP) models, including monotonicity-constrained and convexity-constrained regression. It includes both MATLAB and R implementations, synthetic and real datasets, and experiment scripts for evaluation.

---

## Repository Structure

### Folders

- `datasets_car/`  
  Car mileage/price datasets used in monotone regression experiments.

- `datasets_cosh/`  
  Synthetic datasets generated from cosh for testing convex.

- `datasets_parabola/`  
  Synthetic datasets generated from parabola for testing convex.

- `datasets_sigmoid/`  
  Synthetic datasets generated from sigmoid for testing monotone.

- `datasets_sine/`  
  Synthetic datasets generated from sine for testing monotone.

- `datasets_stepwise_convex/`  
  Synthetic datasets generated from stepwise for testing convex.

- `datasets_stepwise_monotone/`  
  Synthetic datasets generated from stepwise for testing monotone.

- `plots/`  
  Figures for fitted curves/surfaces, credible intervals, and diagnostics.

- `results_convex/`  
  Saved experiment outputs for convexity-constrained model runs.

- `results_convex_stats/`  
  Aggregated statistics and summaries computed from `results_convex`.

- `results_monotone/`  
  Saved experiment outputs for monotonicity-constrained model runs.

- `results_monotone_stats/`  
  Aggregated statistics and summaries computed from `results_monotone`.

---

### Files

- `BF_convex_1d.R`  
  BF on 1D convex datasets
  Requires LineqGPR to be installed.

- `BF_monotone_1d.R`  
  BF on 1D monotone datasets.
  Requires LineqGPR to be installed.

- `BF_monotone_2d.R`  
  BF on 2D monotone datasets.
  Requires LineqGPR to be installed.

- `CGP_convex.m`  
  CGP on 1D convex datasets
  Requires GPML to be installed.

- `CGP_monotone.m`  
  CGP on 1D monotone datasets
  Requires GPML to be installed.

- `CGP_monotone_2D.m`  
  CGP on 2D monotone datasets
  Requires GPML to be installed.

- `IP.m`  
  IP on 1D monotone datasets
  Requires GPstuff to be installed.

- `IP_2D.m`  
  IP on 2D monotone datasets
  Requires GPstuff to be installed.
  
- `SR_convex_1d.R`  
  SR on 1D convex datasets
  Requires bsamGP to be installed.

- `SR_monotone_1d.R`  
  SR on 1D monotone datasets
  Requires bsamGP to be installed.
  
- `baseline_convex.R`  
  Baseline convex regression model in R; fits and evaluates a convex-constrained function.

- `baseline_monotone.r`  
  Baseline monotone regression model in R; fits and evaluates a monotone-constrained function.

- `data_generate.R`  
  R script for generating synthetic monotone datasets.

- `data_generate_convex.R`  
  R script for generating synthetic convex datasets.

- `fit_plot.m`  
  MATLAB plotting helper to visualize fitted means, credible intervals, and observed data.

- `main.m`  
  Runs repeated monotone/convex-regression simulations and collect performance statistics for IP and CGP.

- `simulate_convex.R`  
  Run repeated convex-regression simulations and collect performance statistics for SR, BF, and baselines.

- `simulate_monotone.R`  
  Run repeated convex-regression simulations and collect performance statistics for SR, BF, and baselines.

- `test_convex.R`  
  Hypothesis testing of the convex-regression metrics for IP. CGP, SR, and BF.

- `test_monotone.R`  
  Hypothesis testing of the monotone-regression metrics for IP. CGP, SR, and BF.


