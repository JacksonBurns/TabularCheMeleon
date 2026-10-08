# Eval Results

timestamp: 2026-10-07 19:37:56.866561
checkpoint: tabularchemeleonv4.pt

## `polaris/pkis2-ret-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |       Score |
|---:|:-----------|:---------------|:--------------------|------------:|
|  0 | test       | RET            | pearsonr            |    0.577115 |
|  1 | test       | RET            | mean_squared_error  | 1068.09     |
|  2 | test       | RET            | explained_var       |    0.202148 |
|  3 | test       | RET            | spearmanr           |    0.510715 |
|  4 | test       | RET            | mean_absolute_error |   24.0548   |
|  5 | test       | RET            | r2                  |    0.100481 |

### Leaderboard Comparison
| Name                                 |   mean_squared_error |
|:-------------------------------------|---------------------:|
| 1B_MolGPS-ens_LargeMix-and-Phenomics |              589.944 |
| 3B_e50_MPNN_LargeMix-and-Phenomics   |              609.399 |
| CheMeleon                            |              684.319 |
| CheMeleon                            |              724.347 |
| TabularCheMeleon                     |             1068.09  |

## `polaris/pkis2-kit-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |       Score |
|---:|:-----------|:---------------|:--------------------|------------:|
|  0 | test       | KIT            | pearsonr            |    0.418586 |
|  1 | test       | KIT            | mean_squared_error  | 1046.33     |
|  2 | test       | KIT            | explained_var       |    0.133549 |
|  3 | test       | KIT            | spearmanr           |    0.370088 |
|  4 | test       | KIT            | mean_absolute_error |   26.6165   |
|  5 | test       | KIT            | r2                  |    0.132887 |

### Leaderboard Comparison
| Name             |   mean_squared_error |
|:-----------------|---------------------:|
| CheMeleon        |              849.611 |
| TabularCheMeleon |             1046.33  |

## `polaris/pkis2-egfr-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | EGFR           | pearsonr            |   0.355242 |
|  1 | test       | EGFR           | mean_squared_error  | 705.672    |
|  2 | test       | EGFR           | explained_var       |   0.124545 |
|  3 | test       | EGFR           | spearmanr           |   0.185053 |
|  4 | test       | EGFR           | mean_absolute_error |  21.2677   |
|  5 | test       | EGFR           | r2                  |   0.124225 |

### Leaderboard Comparison
| Name                                   |   mean_squared_error |
|:---------------------------------------|---------------------:|
| aether-pharmaos-pkis2-egfr-wt-ensemble |              430.821 |
| CheMeleon                              |              459.929 |
| TabularCheMeleon                       |              705.672 |

## `polaris/adme-fang-solu-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_SOLUBILITY | pearsonr            | 0.488533 |
|  1 | test       | LOG_SOLUBILITY | mean_squared_error  | 0.43646  |
|  2 | test       | LOG_SOLUBILITY | explained_var       | 0.216819 |
|  3 | test       | LOG_SOLUBILITY | spearmanr           | 0.443314 |
|  4 | test       | LOG_SOLUBILITY | mean_absolute_error | 0.435324 |
|  5 | test       | LOG_SOLUBILITY | r2                  | 0.194989 |

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
| TabularCheMeleon            |   0.488533 |

## `polaris/adme-fang-rppb-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_RPPB       | pearsonr            | 0.63103  |
|  1 | test       | LOG_RPPB       | mean_squared_error  | 0.545893 |
|  2 | test       | LOG_RPPB       | explained_var       | 0.388305 |
|  3 | test       | LOG_RPPB       | spearmanr           | 0.746957 |
|  4 | test       | LOG_RPPB       | mean_absolute_error | 0.547008 |
|  5 | test       | LOG_RPPB       | r2                  | 0.385589 |

### Leaderboard Comparison
| Name                                          |   pearsonr |
|:----------------------------------------------|-----------:|
| 1B_MPNN_LargeMix-and-Phenomics                |    0.908   |
| 1B_MPNN_MolGPS-ens_LargeMix                   |    0.886   |
| nepare                                        |    0.863   |
| TabPFNv2-rdkit                                |    0.816   |
| nepare_chemprop                               |    0.78    |
| chemma-2b-sft                                 |    0.747   |
| adme-fang-RPPB-1_desc2D_RandomForestRegressor |    0.722   |
| adme-fang-RPPB-1-GIRAFFE-wae                  |    0.68    |
| CheMeleon                                     |    0.662   |
| TabularCheMeleon                              |    0.63103 |
| chemlactica-1b-sft                            |    0.614   |

## `polaris/adme-fang-hppb-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_HPPB       | pearsonr            | 0.72821  |
|  1 | test       | LOG_HPPB       | mean_squared_error  | 0.32889  |
|  2 | test       | LOG_HPPB       | explained_var       | 0.525664 |
|  3 | test       | LOG_HPPB       | spearmanr           | 0.767668 |
|  4 | test       | LOG_HPPB       | mean_absolute_error | 0.457421 |
|  5 | test       | LOG_HPPB       | r2                  | 0.456954 |

### Leaderboard Comparison
| Name                                            |   pearsonr |
|:------------------------------------------------|-----------:|
| 1B_MPNN_LargeMix-and-Phenomics                  |    0.888   |
| 1B_MPNN_MolGPS-ens_LargeMix                     |    0.884   |
| TabPFNv2-rdkit                                  |    0.827   |
| adme-fang-HPPB-1-GIRAFFE-wae                    |    0.815   |
| agentomics-ml-adme-fang-hppb-1                  |    0.815   |
| nepare                                          |    0.809   |
| CheMeleon                                       |    0.793   |
| chemlactica-125m-sft                            |    0.774   |
| TabularCheMeleon                                |    0.72821 |
| adme-fang-HPPB-1_atompair_RandomForestRegressor |    0.69    |
| chemma-2b-sft                                   |    0.636   |

## `polaris/adme-fang-perm-1`

### Model Performance
|    | Test set   | Target label     | Metric              |    Score |
|---:|:-----------|:-----------------|:--------------------|---------:|
|  0 | test       | LOG_MDR1-MDCK_ER | pearsonr            | 0.665101 |
|  1 | test       | LOG_MDR1-MDCK_ER | mean_squared_error  | 0.283499 |
|  2 | test       | LOG_MDR1-MDCK_ER | explained_var       | 0.436572 |
|  3 | test       | LOG_MDR1-MDCK_ER | spearmanr           | 0.64468  |
|  4 | test       | LOG_MDR1-MDCK_ER | mean_absolute_error | 0.402079 |
|  5 | test       | LOG_MDR1-MDCK_ER | r2                  | 0.427728 |

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
| TabularCheMeleon                              |   0.665101 |

## `polaris/adme-fang-rclint-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_RLM_CLint  | pearsonr            | 0.566449 |
|  1 | test       | LOG_RLM_CLint  | mean_squared_error  | 0.389168 |
|  2 | test       | LOG_RLM_CLint  | explained_var       | 0.31405  |
|  3 | test       | LOG_RLM_CLint  | spearmanr           | 0.568436 |
|  4 | test       | LOG_RLM_CLint  | mean_absolute_error | 0.510206 |
|  5 | test       | LOG_RLM_CLint  | r2                  | 0.310651 |

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
| TabularCheMeleon                                |   0.566449 |
| adme-fang-RCLint-1_desc2D_FCModel               |   0.544    |

## `polaris/adme-fang-hclint-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_HLM_CLint  | pearsonr            | 0.515822 |
|  1 | test       | LOG_HLM_CLint  | mean_squared_error  | 0.294817 |
|  2 | test       | LOG_HLM_CLint  | explained_var       | 0.264783 |
|  3 | test       | LOG_HLM_CLint  | spearmanr           | 0.529911 |
|  4 | test       | LOG_HLM_CLint  | mean_absolute_error | 0.437233 |
|  5 | test       | LOG_HLM_CLint  | r2                  | 0.240964 |

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
| TabularCheMeleon                                |   0.515822 |

## `tdcommons/lipophilicity-astrazeneca`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.869601 |

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
| TabularCheMeleon                          |              0.869601 |

## `tdcommons/ppbr-az`

### Model Performance
|    | Test set   | Target label   | Metric              |   Score |
|---:|:-----------|:---------------|:--------------------|--------:|
|  0 | test       | Y              | mean_absolute_error | 8.72338 |

### Leaderboard Comparison
| Name                               |   mean_absolute_error |
|:-----------------------------------|----------------------:|
| 3B_e50_MPNN_LargeMix-and-Phenomics |               7.004   |
| 1B_MPNN_LargeMix-and-Phenomics     |               7.026   |
| TabPFNv2-rdkit-3D                  |               7.033   |
| TabPFNv2-rdkit                     |               7.117   |
| CheMeleon                          |               7.532   |
| TabularCheMeleon                   |               8.72338 |

## `tdcommons/clearance-hepatocyte-az`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.343935 |

### Leaderboard Comparison
| Name                                |   spearmanr |
|:------------------------------------|------------:|
| 1B_MPNN_LargeMix-and-Phenomics      |    0.538    |
| 3B_e50_MPNN_LargeMix-and-Phenomics  |    0.525    |
| clearance-hepatocyte-az-GIRAFFE-wae |    0.429    |
| TabPFNv2-rdkit                      |    0.41     |
| TabPFNv2-rdkit-3D                   |    0.402    |
| CheMeleon                           |    0.387    |
| TabularCheMeleon                    |    0.343935 |

## `tdcommons/half-life-obach`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.256264 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| 1B_MPNN_LargeMix-and-Phenomics     |    0.625    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.573    |
| TabPFNv2-rdkit                     |    0.542    |
| TabPFNv2-rdkit-3D                  |    0.489    |
| MolEncoder                         |    0.417    |
| CheMeleon                          |    0.36     |
| TabularCheMeleon                   |    0.256264 |

## `tdcommons/clearance-microsome-az`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.439041 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.683    |
| 1B_MPNN_LargeMix-and-Phenomics     |    0.653    |
| TabPFNv2-rdkit-3D                  |    0.649    |
| TabPFNv2-rdkit                     |    0.64     |
| CheMeleon                          |    0.59     |
| TabularCheMeleon                   |    0.439041 |

## `tdcommons/vdss-lombardo`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.388559 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| vidraft-tabpfn-vdss                |    0.726    |
| TabPFNv2-rdkit                     |    0.7      |
| TabPFNv2-rdkit-3D                  |    0.69     |
| 1B_MPNN_LargeMix-and-Phenomics     |    0.637    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.6      |
| CheMeleon                          |    0.59     |
| TabularCheMeleon                   |    0.388559 |

## `tdcommons/caco2-wang`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.458928 |

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
| TabularCheMeleon                          |              0.458928 |

## `tdcommons/ld50-zhu`

### Model Performance
|    | Test set   | Target label   | Metric              |   Score |
|---:|:-----------|:---------------|:--------------------|--------:|
|  0 | test       | Y              | mean_absolute_error |  0.7185 |

### Leaderboard Comparison
| Name                               |   mean_absolute_error |
|:-----------------------------------|----------------------:|
| CheMeleon                          |                0.526  |
| TabPFNv2-rdkit                     |                0.6    |
| TabPFNv2-rdkit-3D                  |                0.605  |
| 1B_MPNN_LargeMix-and-Phenomics     |                0.614  |
| 3B_e50_MPNN_LargeMix-and-Phenomics |                0.625  |
| TabularCheMeleon                   |                0.7185 |

# Summary

Average Rank of TabularCheMeleon across benchmarks with 4+ other entries (15): 8.53

results_dict = {
    "polaris/pkis2-ret-wt-reg-v2": {
        "mean_squared_error": 1068.093061246866
    },
    "polaris/pkis2-kit-wt-reg-v2": {
        "mean_squared_error": 1046.331195038693
    },
    "polaris/pkis2-egfr-wt-reg-v2": {
        "mean_squared_error": 705.6715256476682
    },
    "polaris/adme-fang-solu-1": {
        "pearsonr": 0.48853282029410916
    },
    "polaris/adme-fang-rppb-1": {
        "pearsonr": 0.6310295408390837
    },
    "polaris/adme-fang-hppb-1": {
        "pearsonr": 0.7282095392690612
    },
    "polaris/adme-fang-perm-1": {
        "pearsonr": 0.6651005715017023
    },
    "polaris/adme-fang-rclint-1": {
        "pearsonr": 0.5664491355631013
    },
    "polaris/adme-fang-hclint-1": {
        "pearsonr": 0.515821941376274
    },
    "tdcommons/lipophilicity-astrazeneca": {
        "mean_absolute_error": 0.8696014474630357
    },
    "tdcommons/ppbr-az": {
        "mean_absolute_error": 8.723382632233376
    },
    "tdcommons/clearance-hepatocyte-az": {
        "spearmanr": 0.3439353661474872
    },
    "tdcommons/half-life-obach": {
        "spearmanr": 0.25626382039470935
    },
    "tdcommons/clearance-microsome-az": {
        "spearmanr": 0.4390406730013735
    },
    "tdcommons/vdss-lombardo": {
        "spearmanr": 0.3885590354311097
    },
    "tdcommons/caco2-wang": {
        "mean_absolute_error": 0.4589275611773817
    },
    "tdcommons/ld50-zhu": {
        "mean_absolute_error": 0.7184996248756275
    }
}
