# Eval Results

timestamp: 2026-09-26 16:33:38.181984
checkpoint: tabular_chemeleon_logs/inductive_tabular_chemeleon/version_8/checkpoints/best-epoch=72-val_loss=0.0000.ckpt

## `polaris/pkis2-ret-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |       Score |
|---:|:-----------|:---------------|:--------------------|------------:|
|  0 | test       | RET            | explained_var       |    0.143201 |
|  1 | test       | RET            | mean_absolute_error |   26.359    |
|  2 | test       | RET            | pearsonr            |    0.463838 |
|  3 | test       | RET            | spearmanr           |    0.483545 |
|  4 | test       | RET            | r2                  |    0.142826 |
|  5 | test       | RET            | mean_squared_error  | 1017.81     |

### Leaderboard Comparison
| Name                                 |   mean_squared_error |
|:-------------------------------------|---------------------:|
| 1B_MolGPS-ens_LargeMix-and-Phenomics |              589.944 |
| 3B_e50_MPNN_LargeMix-and-Phenomics   |              609.399 |
| CheMeleon                            |              684.319 |
| CheMeleon                            |              724.347 |
| TabularCheMeleon                     |             1017.81  |

## `polaris/pkis2-kit-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |       Score |
|---:|:-----------|:---------------|:--------------------|------------:|
|  0 | test       | KIT            | explained_var       |    0.118103 |
|  1 | test       | KIT            | mean_absolute_error |   29.3166   |
|  2 | test       | KIT            | pearsonr            |    0.353586 |
|  3 | test       | KIT            | spearmanr           |    0.31595  |
|  4 | test       | KIT            | r2                  |    0.053611 |
|  5 | test       | KIT            | mean_squared_error  | 1141.99     |

### Leaderboard Comparison
| Name             |   mean_squared_error |
|:-----------------|---------------------:|
| CheMeleon        |              849.611 |
| TabularCheMeleon |             1141.99  |

## `polaris/pkis2-egfr-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |       Score |
|---:|:-----------|:---------------|:--------------------|------------:|
|  0 | test       | EGFR           | explained_var       |   0.123786  |
|  1 | test       | EGFR           | mean_absolute_error |  21.7946    |
|  2 | test       | EGFR           | pearsonr            |   0.411765  |
|  3 | test       | EGFR           | spearmanr           |   0.289193  |
|  4 | test       | EGFR           | r2                  |   0.0911648 |
|  5 | test       | EGFR           | mean_squared_error  | 732.311     |

### Leaderboard Comparison
| Name                                   |   mean_squared_error |
|:---------------------------------------|---------------------:|
| aether-pharmaos-pkis2-egfr-wt-ensemble |              430.821 |
| CheMeleon                              |              459.929 |
| TabularCheMeleon                       |              732.311 |

## `polaris/adme-fang-solu-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_SOLUBILITY | explained_var       | 0.173361 |
|  1 | test       | LOG_SOLUBILITY | mean_absolute_error | 0.449415 |
|  2 | test       | LOG_SOLUBILITY | pearsonr            | 0.477491 |
|  3 | test       | LOG_SOLUBILITY | spearmanr           | 0.456024 |
|  4 | test       | LOG_SOLUBILITY | r2                  | 0.126561 |
|  5 | test       | LOG_SOLUBILITY | mean_squared_error  | 0.47356  |

### Leaderboard Comparison
| Name                        |   pearsonr |
|:----------------------------|-----------:|
| 1B_MPNN_MolGPS-ens_LargeMix |   0.77     |
| 1B_MPNN_LargeMix-Phenomics  |   0.764    |
| CheMeleonMOE                |   0.729    |
| CheMeleon                   |   0.682    |
| it-works-now                |   0.669    |
| ML4DD-team16                |   0.654    |
| ML4DD-team9                 |   0.651    |
| team9_submission_2          |   0.651    |
| ExactTanimotoGP             |   0.635    |
| ML4DD-team25                |   0.632    |
| TabularCheMeleon            |   0.477491 |

## `polaris/adme-fang-rppb-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_RPPB       | explained_var       | 0.487161 |
|  1 | test       | LOG_RPPB       | mean_absolute_error | 0.561573 |
|  2 | test       | LOG_RPPB       | pearsonr            | 0.7271   |
|  3 | test       | LOG_RPPB       | spearmanr           | 0.843478 |
|  4 | test       | LOG_RPPB       | r2                  | 0.474887 |
|  5 | test       | LOG_RPPB       | mean_squared_error  | 0.466554 |

### Leaderboard Comparison
| Name                                          |   pearsonr |
|:----------------------------------------------|-----------:|
| 1B_MPNN_LargeMix-and-Phenomics                |     0.908  |
| 1B_MPNN_MolGPS-ens_LargeMix                   |     0.886  |
| nepare                                        |     0.863  |
| TabPFNv2-rdkit                                |     0.816  |
| nepare_chemprop                               |     0.78   |
| chemma-2b-sft                                 |     0.747  |
| TabularCheMeleon                              |     0.7271 |
| adme-fang-RPPB-1_desc2D_RandomForestRegressor |     0.722  |
| adme-fang-RPPB-1-GIRAFFE-wae                  |     0.68   |
| CheMeleon                                     |     0.662  |
| chemlactica-1b-sft                            |     0.614  |

## `polaris/adme-fang-hppb-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_HPPB       | explained_var       | 0.578942 |
|  1 | test       | LOG_HPPB       | mean_absolute_error | 0.436405 |
|  2 | test       | LOG_HPPB       | pearsonr            | 0.800539 |
|  3 | test       | LOG_HPPB       | spearmanr           | 0.79288  |
|  4 | test       | LOG_HPPB       | r2                  | 0.560099 |
|  5 | test       | LOG_HPPB       | mean_squared_error  | 0.266421 |

### Leaderboard Comparison
| Name                                            |   pearsonr |
|:------------------------------------------------|-----------:|
| 1B_MPNN_LargeMix-and-Phenomics                  |   0.888    |
| 1B_MPNN_MolGPS-ens_LargeMix                     |   0.884    |
| TabPFNv2-rdkit                                  |   0.827    |
| adme-fang-HPPB-1-GIRAFFE-wae                    |   0.815    |
| agentomics-ml-adme-fang-hppb-1                  |   0.815    |
| nepare                                          |   0.809    |
| TabularCheMeleon                                |   0.800539 |
| CheMeleon                                       |   0.793    |
| chemlactica-125m-sft                            |   0.774    |
| adme-fang-HPPB-1_atompair_RandomForestRegressor |   0.69     |
| chemma-2b-sft                                   |   0.636    |

## `polaris/adme-fang-perm-1`

### Model Performance
|    | Test set   | Target label     | Metric              |    Score |
|---:|:-----------|:-----------------|:--------------------|---------:|
|  0 | test       | LOG_MDR1-MDCK_ER | explained_var       | 0.395134 |
|  1 | test       | LOG_MDR1-MDCK_ER | mean_absolute_error | 0.454619 |
|  2 | test       | LOG_MDR1-MDCK_ER | pearsonr            | 0.635694 |
|  3 | test       | LOG_MDR1-MDCK_ER | spearmanr           | 0.648178 |
|  4 | test       | LOG_MDR1-MDCK_ER | r2                  | 0.380486 |
|  5 | test       | LOG_MDR1-MDCK_ER | mean_squared_error  | 0.306902 |

### Leaderboard Comparison
| Name                                          |   pearsonr |
|:----------------------------------------------|-----------:|
| 1B_MPNN_MolGPS-ens_LargeMix                   |   0.879    |
| 1B_MPNN_LargeMix-and-Phenomics                |   0.86     |
| CheMeleon                                     |   0.822    |
| MolEncoder                                    |   0.804    |
| TabPFNv2-rdkit                                |   0.798    |
| adme-fang-PERM-1-GIRAFFE-wae_s                |   0.772    |
| chemlactica-1b-sft                            |   0.762    |
| optimized-random-forest-rdkit-descriptors     |   0.727    |
| adme-fang-PERM-1_desc2D_RandomForestRegressor |   0.716    |
| chemlactica-125m-sft                          |   0.714    |
| TabularCheMeleon                              |   0.635694 |

## `polaris/adme-fang-rclint-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_RLM_CLint  | explained_var       | 0.289799 |
|  1 | test       | LOG_RLM_CLint  | mean_absolute_error | 0.517682 |
|  2 | test       | LOG_RLM_CLint  | pearsonr            | 0.543303 |
|  3 | test       | LOG_RLM_CLint  | spearmanr           | 0.54443  |
|  4 | test       | LOG_RLM_CLint  | r2                  | 0.28739  |
|  5 | test       | LOG_RLM_CLint  | mean_squared_error  | 0.402299 |

### Leaderboard Comparison
| Name                                            |   pearsonr |
|:------------------------------------------------|-----------:|
| 1B_MPNN_MolGPS-ens_LargeMix                     |   0.798    |
| 1B_MPNN_LargeMix-and-Phenomics                  |   0.784    |
| CheMeleon                                       |   0.757    |
| chemlactica-125m-sft                            |   0.714    |
| chemlactica-1b-sft                              |   0.698    |
| TabPFNv2-rdkit                                  |   0.694    |
| chemma-2b-sft                                   |   0.66     |
| adme-fang-RCLint-1_desc2D_RandomForestRegressor |   0.64     |
| adme-fang-RCLint-1_desc2D_RandomForestRegressor |   0.631    |
| adme-fang-RCLint-1_desc2D_FCModel               |   0.544    |
| TabularCheMeleon                                |   0.543303 |

## `polaris/adme-fang-hclint-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_HLM_CLint  | explained_var       | 0.243247 |
|  1 | test       | LOG_HLM_CLint  | mean_absolute_error | 0.445188 |
|  2 | test       | LOG_HLM_CLint  | pearsonr            | 0.493755 |
|  3 | test       | LOG_HLM_CLint  | spearmanr           | 0.513323 |
|  4 | test       | LOG_HLM_CLint  | r2                  | 0.241969 |
|  5 | test       | LOG_HLM_CLint  | mean_squared_error  | 0.294427 |

### Leaderboard Comparison
| Name                                            |   pearsonr |
|:------------------------------------------------|-----------:|
| seqera-gradient                                 |   0.796    |
| 1B_MPNN_LargeMix-and-Phenomics                  |   0.778    |
| chemlactica-1b-sft                              |   0.72     |
| CheMeleon                                       |   0.72     |
| chemlactica-125m-sft                            |   0.717    |
| MolEncoder                                      |   0.714    |
| agentomics-ml-adme-fang-hclint-1                |   0.697    |
| chemma-2b-sft                                   |   0.674    |
| TabPFNv2-rdkit                                  |   0.662    |
| adme-fang-HCLint-1_desc2D_RandomForestRegressor |   0.639    |
| TabularCheMeleon                                |   0.493755 |

## `tdcommons/lipophilicity-astrazeneca`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.808804 |

### Leaderboard Comparison
| Name                                      |   mean_absolute_error |
|:------------------------------------------|----------------------:|
| 1B_MPNN_MolGPS-ens_LargeMix-and-Phenomics |              0.392    |
| 1B_MPNN_LargeMix-and-Phenomics            |              0.411    |
| 3B_e50_MPNN_LargeMix-and-Phenomics        |              0.424    |
| CheMeleon                                 |              0.443    |
| MolEncoder                                |              0.497    |
| agentomics-ml-lipophilicity-astrazeneca   |              0.497    |
| TabPFNv2-rdkit                            |              0.499    |
| TabPFNv2-rdkit                            |              0.502    |
| TabPFNv2-maplight_gnn                     |              0.502    |
| TabPFNv2-rdkit-3D                         |              0.524    |
| TabularCheMeleon                          |              0.808804 |

## `tdcommons/ppbr-az`

### Model Performance
|    | Test set   | Target label   | Metric              |   Score |
|---:|:-----------|:---------------|:--------------------|--------:|
|  0 | test       | Y              | mean_absolute_error |  8.5775 |

### Leaderboard Comparison
| Name                               |   mean_absolute_error |
|:-----------------------------------|----------------------:|
| 3B_e50_MPNN_LargeMix-and-Phenomics |                7.004  |
| 1B_MPNN_LargeMix-and-Phenomics     |                7.026  |
| TabPFNv2-rdkit-3D                  |                7.033  |
| TabPFNv2-rdkit                     |                7.117  |
| CheMeleon                          |                7.532  |
| TabularCheMeleon                   |                8.5775 |

## `tdcommons/clearance-hepatocyte-az`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.301807 |

### Leaderboard Comparison
| Name                                |   spearmanr |
|:------------------------------------|------------:|
| 1B_MPNN_LargeMix-and-Phenomics      |    0.538    |
| 3B_e50_MPNN_LargeMix-and-Phenomics  |    0.525    |
| clearance-hepatocyte-az-GIRAFFE-wae |    0.429    |
| TabPFNv2-rdkit                      |    0.41     |
| TabPFNv2-rdkit-3D                   |    0.402    |
| CheMeleon                           |    0.387    |
| TabularCheMeleon                    |    0.301807 |

## `tdcommons/half-life-obach`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.483922 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| 1B_MPNN_LargeMix-and-Phenomics     |    0.625    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.573    |
| TabPFNv2-rdkit                     |    0.542    |
| TabPFNv2-rdkit-3D                  |    0.489    |
| TabularCheMeleon                   |    0.483922 |
| MolEncoder                         |    0.417    |
| CheMeleon                          |    0.36     |

## `tdcommons/clearance-microsome-az`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.473698 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.683    |
| 1B_MPNN_LargeMix-and-Phenomics     |    0.653    |
| TabPFNv2-rdkit-3D                  |    0.649    |
| TabPFNv2-rdkit                     |    0.64     |
| CheMeleon                          |    0.59     |
| TabularCheMeleon                   |    0.473698 |

## `tdcommons/vdss-lombardo`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.459358 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| vidraft-tabpfn-vdss                |    0.726    |
| TabPFNv2-rdkit                     |    0.7      |
| TabPFNv2-rdkit-3D                  |    0.69     |
| 1B_MPNN_LargeMix-and-Phenomics     |    0.637    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.6      |
| CheMeleon                          |    0.59     |
| TabularCheMeleon                   |    0.459358 |

## `tdcommons/caco2-wang`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.408835 |

### Leaderboard Comparison
| Name                                      |   mean_absolute_error |
|:------------------------------------------|----------------------:|
| mini-PlanE-Seed-Descriptor                |              0.27     |
| agentomics-ml-caco2-wang                  |              0.277    |
| 1B_MPNN_MolGPS-ens_LargeMix-and-Phenomics |              0.282    |
| drylab                                    |              0.282    |
| TabPFNv2-maplight_gnn                     |              0.284    |
| TabPFNv2-rdkit                            |              0.285    |
| mini-PlanE-EBasePlanE-ensemble-rich-znorm |              0.301    |
| mini-PlanE-E-BasePlanE-seed42             |              0.314    |
| 3B_e50_MPNN_LargeMix-and-Phenomics        |              0.316    |
| 1B_MPNN_LargeMix-and-Phenomics            |              0.319    |
| TabularCheMeleon                          |              0.408835 |

## `tdcommons/ld50-zhu`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.735906 |

### Leaderboard Comparison
| Name                               |   mean_absolute_error |
|:-----------------------------------|----------------------:|
| CheMeleon                          |              0.526    |
| TabPFNv2-rdkit                     |              0.6      |
| TabPFNv2-rdkit-3D                  |              0.605    |
| 1B_MPNN_LargeMix-and-Phenomics     |              0.614    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |              0.625    |
| TabularCheMeleon                   |              0.735906 |

# Summary

Average Rank of TabularCheMeleon across benchmarks with 4+ other entries 15: 8.13

results_dict = {
    "polaris/pkis2-ret-wt-reg-v2": {
        "mean_squared_error": 1017.8131393237215
    },
    "polaris/pkis2-kit-wt-reg-v2": {
        "mean_squared_error": 1141.9924473940541
    },
    "polaris/pkis2-egfr-wt-reg-v2": {
        "mean_squared_error": 732.3107838038151
    },
    "polaris/adme-fang-solu-1": {
        "pearsonr": 0.4774910272985249
    },
    "polaris/adme-fang-rppb-1": {
        "pearsonr": 0.7270996286121887
    },
    "polaris/adme-fang-hppb-1": {
        "pearsonr": 0.8005392133308453
    },
    "polaris/adme-fang-perm-1": {
        "pearsonr": 0.635693666110491
    },
    "polaris/adme-fang-rclint-1": {
        "pearsonr": 0.5433033255983457
    },
    "polaris/adme-fang-hclint-1": {
        "pearsonr": 0.49375540641174814
    },
    "tdcommons/lipophilicity-astrazeneca": {
        "mean_absolute_error": 0.8088035338265555
    },
    "tdcommons/ppbr-az": {
        "mean_absolute_error": 8.57749525803786
    },
    "tdcommons/clearance-hepatocyte-az": {
        "spearmanr": 0.3018066562168246
    },
    "tdcommons/half-life-obach": {
        "spearmanr": 0.48392171897338104
    },
    "tdcommons/clearance-microsome-az": {
        "spearmanr": 0.4736975777501099
    },
    "tdcommons/vdss-lombardo": {
        "spearmanr": 0.45935818315542565
    },
    "tdcommons/caco2-wang": {
        "mean_absolute_error": 0.4088349515312448
    },
    "tdcommons/ld50-zhu": {
        "mean_absolute_error": 0.7359062121311286
    }
}
