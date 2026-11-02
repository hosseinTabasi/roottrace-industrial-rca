# RootTrace — Results

All values are copied from `experiments/results/*.csv` (seeds 0,1,2). See README for discussion.

## e1_e5_rootcause_summary

| method                      |   top1_mean |   top1_std |   top3_mean |   top3_std |   top5_mean |   top5_std |   mrr_mean |   mrr_std |
|:----------------------------|------------:|-----------:|------------:|-----------:|------------:|-----------:|-----------:|----------:|
| PCA-SPE contribution        |       0.44  |      0     |       0.56  |      0     |       0.653 |      0.012 |      0.551 |     0.002 |
| Ablation: no propagation    |       0.393 |      0.023 |       0.493 |      0.05  |       0.607 |      0.023 |      0.494 |     0.027 |
| Forecast-residual ranking   |       0.36  |      0     |       0.493 |      0.042 |       0.653 |      0.095 |      0.489 |     0.016 |
| Sensitivity: restart=0.8    |       0.393 |      0.023 |       0.44  |      0.053 |       0.587 |      0.031 |      0.486 |     0.022 |
| PCA-T2 contribution         |       0.38  |      0     |       0.54  |      0     |       0.54  |      0     |      0.485 |     0     |
| Sensitivity: restart=0.5    |       0.38  |      0     |       0.427 |      0.031 |       0.62  |      0.02  |      0.481 |     0.004 |
| RootTrace (full)            |       0.347 |      0.012 |       0.48  |      0     |       0.667 |      0.061 |      0.48  |     0.01  |
| Ablation: linear graph      |       0.387 |      0.012 |       0.46  |      0.02  |       0.593 |      0.046 |      0.473 |     0.005 |
| Linear Granger + PageRank   |       0.187 |      0.012 |       0.42  |      0.035 |       0.493 |      0.031 |      0.356 |     0.005 |
| Sensitivity: restart=0.1    |       0.02  |      0     |       0.5   |      0.069 |       0.667 |      0.023 |      0.309 |     0.01  |
| Ablation: no counterfactual |       0     |      0     |       0.5   |      0.04  |       0.66  |      0.04  |      0.307 |     0.01  |
| Random                      |       0.033 |      0.023 |       0.113 |      0.046 |       0.213 |      0.058 |      0.136 |     0.029 |

## e2_diagnosis_summary

| embedding                |   k |   acc_mean |   acc_std |   acc_excl_3_9_15_mean |   acc_excl_3_9_15_std |
|:-------------------------|----:|-----------:|----------:|-----------------------:|----------------------:|
| Graph-residual embedding |   1 |      0.356 |     0.012 |                  0.386 |                 0.013 |
| Graph-residual embedding |   5 |      0.501 |     0.015 |                  0.551 |                 0.019 |
| Graph-residual embedding |  10 |      0.569 |     0.003 |                  0.626 |                 0.004 |
| PCA scores               |   1 |      0.382 |     0.008 |                  0.424 |                 0.011 |
| PCA scores               |   5 |      0.537 |     0.009 |                  0.603 |                 0.009 |
| PCA scores               |  10 |      0.599 |     0.003 |                  0.674 |                 0.003 |
| Raw mean/std             |   1 |      0.337 |     0.01  |                  0.368 |                 0.012 |
| Raw mean/std             |   5 |      0.483 |     0.013 |                  0.533 |                 0.016 |
| Raw mean/std             |  10 |      0.552 |     0.009 |                  0.612 |                 0.011 |

## e3_delay_far_summary

| family         |   quantile |   far_mean |   far_std |   median_delay_mean |   median_delay_std |   detected_faults_mean |   detected_faults_std |
|:---------------|-----------:|-----------:|----------:|--------------------:|-------------------:|-----------------------:|----------------------:|
| Graph residual |      0.9   |      0.04  |     0.01  |              10     |              0     |                 21     |                 0     |
| Graph residual |      0.95  |      0.013 |     0.003 |              11.333 |              2.309 |                 21     |                 0     |
| Graph residual |      0.98  |      0.001 |     0.001 |              14.667 |              2.93  |                 19     |                 1     |
| Graph residual |      0.99  |      0     |     0     |              15.5   |              2.291 |                 19     |                 1     |
| Graph residual |      0.995 |      0     |     0     |              15.5   |              2.291 |                 19     |                 1     |
| Graph residual |      0.999 |      0     |     0     |              15     |              3     |                 18.667 |                 0.577 |
| PCA-SPE        |      0.9   |      0.078 |   nan     |              10     |            nan     |                 21     |               nan     |
| PCA-SPE        |      0.95  |      0.031 |   nan     |              16     |            nan     |                 21     |               nan     |
| PCA-SPE        |      0.98  |      0.007 |   nan     |              16     |            nan     |                 19     |               nan     |
| PCA-SPE        |      0.99  |      0.004 |   nan     |              16     |            nan     |                 19     |               nan     |
| PCA-SPE        |      0.995 |      0     |   nan     |              14     |            nan     |                 17     |               nan     |
| PCA-SPE        |      0.999 |      0     |   nan     |              14     |            nan     |                 17     |               nan     |
| PCA-T2         |      0.9   |      0.04  |   nan     |              20     |            nan     |                 21     |               nan     |
| PCA-T2         |      0.95  |      0.013 |   nan     |              28     |            nan     |                 21     |               nan     |
| PCA-T2         |      0.98  |      0.004 |   nan     |              22     |            nan     |                 17     |               nan     |
| PCA-T2         |      0.99  |      0.001 |   nan     |              22     |            nan     |                 17     |               nan     |
| PCA-T2         |      0.995 |      0.001 |   nan     |              22     |            nan     |                 17     |               nan     |
| PCA-T2         |      0.999 |      0     |   nan     |              22     |            nan     |                 17     |               nan     |

## e4_graph_sanity

| graph                     |   known_pair_recall_top150 |   nonzero_edges |
|:--------------------------|---------------------------:|----------------:|
| NeuralGranger seed 0      |                      0.667 |             467 |
| NeuralGranger seed 1      |                      0.667 |             499 |
| NeuralGranger seed 2      |                      0.667 |             470 |
| Linear Granger (F-test)   |                      0.5   |              71 |
| Random top-150 (expected) |                      0.11  |             150 |

## e4_graph_stability

| pair   |   jaccard_top150 |
|:-------|-----------------:|
| 0-1    |            0.775 |
| 0-2    |            0.754 |
| 1-2    |            0.714 |

## e4_sparsity_selection

|   sparsity |   val_mse |   nonzero_offdiag_edges | selected   |
|-----------:|----------:|------------------------:|:-----------|
|      0.002 |     1.387 |                    2652 | False      |
|      0.2   |     1.267 |                    2652 | False      |
|      1     |     0.868 |                    2587 | False      |
|      2     |     0.592 |                    2396 | False      |
|      5     |     0.531 |                     467 | True       |
|     10     |     0.681 |                       0 | False      |
|     20     |     0.681 |                       0 | False      |
|     50     |     0.681 |                       0 | False      |

