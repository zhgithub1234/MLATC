# MLATC

Code for paper 'Mode-wise low-rank autoregressive tensor completion for traffic data imputation'

## Data

Dataset files should be placed under `datasets/<Dataset>-data-set/`. 
Generated missing masks are stored under `datasets/generated_missing_data/`.

Guangzhou dataset: https://doi.org/10.5281/zenodo.1205229
Hangzhou dataset: https://tianchi.aliyun.com/competition/entrance/231708/information
Seattle dataset: https://github.com/zhiyongc/Seattle-Loop-Data

## Code for baseline models

BGCP, BTTF, LRTC-TNN, and LATC: https://github.com/xinychen/transdim
LRTC-TMCP: https://github.com/peterchen96/LRTC-TMCP

## Requirements

```bash
pip install -r requirements.txt
```

## Project Structure

```text
MLATC/
|-- README.md
|-- requirements.txt
|-- models/
|   |-- mlatc.py
|   `-- mlatc_multi_mode.py
|-- baseline_models/
|   |-- BGCP/BGCP.py
|   |-- BTTF/BTTF.py
|   |-- LATC/LATC.py
|   `-- LRTC_TNN/LRTC_TNN.py
|-- experiments/
|   |-- get_all_results_for_all_models.py
|   |-- reproduce_mlatc_best_results.py
|   |-- optimize_mlatc_parameters_rs_tpe.py
|   |-- optimize_mlatc_parameters_gatpe.py
|   |-- generate_slice_missing_data.py
|   `-- generate_models_recovered_data_for_slicemissing.py
|-- figures/
|   |-- figure1_mode_vis.py
|   |-- figure3_convergence_analysis.py
|   `-- figure5_slice_missing.py
|-- utils/
|   `-- utils_for_results_save.py
|-- datasets/
|   |-- Guangzhou-data-set/
|   |-- Hangzhou-data-set/
|   |-- Seattle-data-set/
|   `-- generated_missing_data/
|-- results/
    |-- model_results/
    |-- mlatc_best_results/
    |-- slice_missing_data/
    |-- slice_recovered_data/
    `-- figure_results/


## Main Scripts

- models/mlatc.py: MLATC model implementation.
- models/mlatc_multi_mode.py: MLATC-Multi-mode model implementation.
- experiments/get_all_results_for_all_models.py: grid-search experiments for MLATC and baselines.
- experiments/reproduce_mlatc_best_results.py: reloads saved `.npy` parameter packages and reproduces MLATC results.
- experiments/optimize_mlatc_parameters_rs_tpe.py: MLATC parameter optimization with RS and TPE.
- experiments/optimize_mlatc_parameters_gatpe.py: MLATC parameter optimization with GA-TPE.
- experiments/generate_slice_missing_data.py: fixed slice-missing data generation for Figure 5.
- experiments/generate_models_recovered_data_for_slicemissing.py: recovered tensor generation for Figure 5.
- figures/figure1_mode_vis.py: mode-wise unfolding and Pearson correlation plots.
- figures/figure3_convergence_analysis.py: primal residual curves for MLATC.
- figures/figure5_slice_missing.py: imputation visualizations for slice-missing patterns.
