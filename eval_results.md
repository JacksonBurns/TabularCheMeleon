# Eval Results

timestamp: 2026-10-14 22:41:11.665364
checkpoint: /home/jackson/TabularCheMeleon/tabular_chemeleon_logs_g4_v1/chemeleon_pfn/version_4/checkpoints/best-epoch=04-val_loss=1.2016.ckpt

## `polaris/pkis2-ret-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | RET            | mean_absolute_error |  19.6229   |
|  1 | test       | RET            | pearsonr            |   0.674968 |
|  2 | test       | RET            | spearmanr           |   0.638968 |
|  3 | test       | RET            | r2                  |   0.382863 |
|  4 | test       | RET            | explained_var       |   0.450321 |
|  5 | test       | RET            | mean_squared_error  | 732.792    |

### Leaderboard Comparison
| Name                                 |   mean_squared_error |
|:-------------------------------------|---------------------:|
| 1B_MolGPS-ens_LargeMix-and-Phenomics |              589.944 |
| 3B_e50_MPNN_LargeMix-and-Phenomics   |              609.399 |
| CheMeleon                            |              684.319 |
| CheMeleon                            |              724.347 |
| TabularCheMeleon                     |              732.792 |

## `polaris/pkis2-kit-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | KIT            | mean_absolute_error |  21.5684   |
|  1 | test       | KIT            | pearsonr            |   0.586361 |
|  2 | test       | KIT            | spearmanr           |   0.534671 |
|  3 | test       | KIT            | r2                  |   0.299012 |
|  4 | test       | KIT            | explained_var       |   0.321386 |
|  5 | test       | KIT            | mean_squared_error  | 845.871    |

### Leaderboard Comparison
| Name             |   mean_squared_error |
|:-----------------|---------------------:|
| TabularCheMeleon |              845.871 |
| CheMeleon        |              849.611 |

## `polaris/pkis2-egfr-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | EGFR           | mean_absolute_error |  16.9202   |
|  1 | test       | EGFR           | pearsonr            |   0.655471 |
|  2 | test       | EGFR           | spearmanr           |   0.357302 |
|  3 | test       | EGFR           | r2                  |   0.409147 |
|  4 | test       | EGFR           | explained_var       |   0.424425 |
|  5 | test       | EGFR           | mean_squared_error  | 476.091    |

### Leaderboard Comparison
| Name                                   |   mean_squared_error |
|:---------------------------------------|---------------------:|
| aether-pharmaos-pkis2-egfr-wt-ensemble |              430.821 |
| CheMeleon                              |              459.929 |
| TabularCheMeleon                       |              476.091 |

## `polaris/adme-fang-solu-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_SOLUBILITY | mean_absolute_error | 0.382663 |
|  1 | test       | LOG_SOLUBILITY | pearsonr            | 0.65854  |
|  2 | test       | LOG_SOLUBILITY | spearmanr           | 0.533484 |
|  3 | test       | LOG_SOLUBILITY | r2                  | 0.370894 |
|  4 | test       | LOG_SOLUBILITY | explained_var       | 0.408143 |
|  5 | test       | LOG_SOLUBILITY | mean_squared_error  | 0.341088 |

### Leaderboard Comparison
| Name                        |   pearsonr |
|:----------------------------|-----------:|
| 1B_MPNN_MolGPS-ens_LargeMix |    0.77    |
| 1B_MPNN_LargeMix-Phenomics  |    0.764   |
| CheMeleonMOE                |    0.729   |
| CheMeleon                   |    0.682   |
| it-works-now                |    0.669   |
| TabularCheMeleon            |    0.65854 |
| ML4DD-team16                |    0.654   |
| ML4DD-team9                 |    0.651   |
| team9_submission_2          |    0.651   |
| ExactTanimotoGP             |    0.635   |
| ML4DD-team25                |    0.632   |

## `polaris/adme-fang-rppb-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_RPPB       | mean_absolute_error | 0.462295 |
|  1 | test       | LOG_RPPB       | pearsonr            | 0.793028 |
|  2 | test       | LOG_RPPB       | spearmanr           | 0.858261 |
|  3 | test       | LOG_RPPB       | r2                  | 0.545564 |
|  4 | test       | LOG_RPPB       | explained_var       | 0.546844 |
|  5 | test       | LOG_RPPB       | mean_squared_error  | 0.403758 |

### Leaderboard Comparison
| Name                                          |   pearsonr |
|:----------------------------------------------|-----------:|
| 1B_MPNN_LargeMix-and-Phenomics                |   0.908    |
| 1B_MPNN_MolGPS-ens_LargeMix                   |   0.886    |
| nepare                                        |   0.863    |
| TabPFNv2-rdkit                                |   0.816    |
| TabularCheMeleon                              |   0.793028 |
| nepare_chemprop                               |   0.78     |
| chemma-2b-sft                                 |   0.747    |
| adme-fang-RPPB-1_desc2D_RandomForestRegressor |   0.722    |
| adme-fang-RPPB-1-GIRAFFE-wae                  |   0.68     |
| CheMeleon                                     |   0.662    |
| chemlactica-1b-sft                            |   0.614    |

## `polaris/adme-fang-hppb-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_HPPB       | mean_absolute_error | 0.489089 |
|  1 | test       | LOG_HPPB       | pearsonr            | 0.73583  |
|  2 | test       | LOG_HPPB       | spearmanr           | 0.73833  |
|  3 | test       | LOG_HPPB       | r2                  | 0.482099 |
|  4 | test       | LOG_HPPB       | explained_var       | 0.537407 |
|  5 | test       | LOG_HPPB       | mean_squared_error  | 0.313661 |

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
| TabularCheMeleon                                |    0.73583 |
| adme-fang-HPPB-1_atompair_RandomForestRegressor |    0.69    |
| chemma-2b-sft                                   |    0.636   |

## `polaris/adme-fang-perm-1`

### Model Performance
|    | Test set   | Target label     | Metric              |    Score |
|---:|:-----------|:-----------------|:--------------------|---------:|
|  0 | test       | LOG_MDR1-MDCK_ER | mean_absolute_error | 0.314318 |
|  1 | test       | LOG_MDR1-MDCK_ER | pearsonr            | 0.789563 |
|  2 | test       | LOG_MDR1-MDCK_ER | spearmanr           | 0.766084 |
|  3 | test       | LOG_MDR1-MDCK_ER | r2                  | 0.623097 |
|  4 | test       | LOG_MDR1-MDCK_ER | explained_var       | 0.623145 |
|  5 | test       | LOG_MDR1-MDCK_ER | mean_squared_error  | 0.186715 |

### Leaderboard Comparison
| Name                                          |   pearsonr |
|:----------------------------------------------|-----------:|
| 1B_MPNN_MolGPS-ens_LargeMix                   |   0.879    |
| 1B_MPNN_LargeMix-and-Phenomics                |   0.86     |
| CheMeleon                                     |   0.822    |
| MolEncoder                                    |   0.804    |
| TabPFNv2-rdkit                                |   0.798    |
| TabularCheMeleon                              |   0.789563 |
| adme-fang-PERM-1-GIRAFFE-wae_s                |   0.772    |
| chemlactica-1b-sft                            |   0.762    |
| optimized-random-forest-rdkit-descriptors     |   0.727    |
| adme-fang-PERM-1_desc2D_RandomForestRegressor |   0.716    |
| chemlactica-125m-sft                          |   0.714    |

## `polaris/adme-fang-rclint-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_RLM_CLint  | mean_absolute_error | 0.393048 |
|  1 | test       | LOG_RLM_CLint  | pearsonr            | 0.739769 |
|  2 | test       | LOG_RLM_CLint  | spearmanr           | 0.737192 |
|  3 | test       | LOG_RLM_CLint  | r2                  | 0.546331 |
|  4 | test       | LOG_RLM_CLint  | explained_var       | 0.546945 |
|  5 | test       | LOG_RLM_CLint  | mean_squared_error  | 0.256116 |

### Leaderboard Comparison
| Name                                            |   pearsonr |
|:------------------------------------------------|-----------:|
| 1B_MPNN_MolGPS-ens_LargeMix                     |   0.798    |
| 1B_MPNN_LargeMix-and-Phenomics                  |   0.784    |
| CheMeleon                                       |   0.757    |
| TabularCheMeleon                                |   0.739769 |
| chemlactica-125m-sft                            |   0.714    |
| chemlactica-1b-sft                              |   0.698    |
| TabPFNv2-rdkit                                  |   0.694    |
| chemma-2b-sft                                   |   0.66     |
| adme-fang-RCLint-1_desc2D_RandomForestRegressor |   0.64     |
| adme-fang-RCLint-1_desc2D_RandomForestRegressor |   0.631    |
| adme-fang-RCLint-1_desc2D_FCModel               |   0.544    |

## `polaris/adme-fang-hclint-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_HLM_CLint  | mean_absolute_error | 0.334295 |
|  1 | test       | LOG_HLM_CLint  | pearsonr            | 0.718447 |
|  2 | test       | LOG_HLM_CLint  | spearmanr           | 0.723932 |
|  3 | test       | LOG_HLM_CLint  | r2                  | 0.503018 |
|  4 | test       | LOG_HLM_CLint  | explained_var       | 0.504421 |
|  5 | test       | LOG_HLM_CLint  | mean_squared_error  | 0.193033 |

### Leaderboard Comparison
| Name                                            |   pearsonr |
|:------------------------------------------------|-----------:|
| seqera-gradient                                 |   0.796    |
| 1B_MPNN_LargeMix-and-Phenomics                  |   0.778    |
| chemlactica-1b-sft                              |   0.72     |
| CheMeleon                                       |   0.72     |
| TabularCheMeleon                                |   0.718447 |
| chemlactica-125m-sft                            |   0.717    |
| MolEncoder                                      |   0.714    |
| agentomics-ml-adme-fang-hclint-1                |   0.697    |
| chemma-2b-sft                                   |   0.674    |
| TabPFNv2-rdkit                                  |   0.662    |
| adme-fang-HCLint-1_desc2D_RandomForestRegressor |   0.639    |

## `tdcommons/lipophilicity-astrazeneca`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.500264 |

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
| TabularCheMeleon                          |              0.500264 |
| TabPFNv2-rdkit                            |              0.502    |
| TabPFNv2-maplight_gnn                     |              0.502    |
| TabPFNv2-rdkit-3D                         |              0.524    |

## `tdcommons/ppbr-az`

### Model Performance
|    | Test set   | Target label   | Metric              |   Score |
|---:|:-----------|:---------------|:--------------------|--------:|
|  0 | test       | Y              | mean_absolute_error | 7.05784 |

### Leaderboard Comparison
| Name                               |   mean_absolute_error |
|:-----------------------------------|----------------------:|
| 3B_e50_MPNN_LargeMix-and-Phenomics |               7.004   |
| 1B_MPNN_LargeMix-and-Phenomics     |               7.026   |
| TabPFNv2-rdkit-3D                  |               7.033   |
| TabularCheMeleon                   |               7.05784 |
| TabPFNv2-rdkit                     |               7.117   |
| CheMeleon                          |               7.532   |

## `tdcommons/clearance-hepatocyte-az`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.397744 |

### Leaderboard Comparison
| Name                                |   spearmanr |
|:------------------------------------|------------:|
| 1B_MPNN_LargeMix-and-Phenomics      |    0.538    |
| 3B_e50_MPNN_LargeMix-and-Phenomics  |    0.525    |
| clearance-hepatocyte-az-GIRAFFE-wae |    0.429    |
| TabPFNv2-rdkit                      |    0.41     |
| TabPFNv2-rdkit-3D                   |    0.402    |
| TabularCheMeleon                    |    0.397744 |
| CheMeleon                           |    0.387    |

## `tdcommons/half-life-obach`

### Model Performance
|    | Test set   | Target label   | Metric    |   Score |
|---:|:-----------|:---------------|:----------|--------:|
|  0 | test       | Y              | spearmanr | 0.57526 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| 1B_MPNN_LargeMix-and-Phenomics     |     0.625   |
| TabularCheMeleon                   |     0.57526 |
| 3B_e50_MPNN_LargeMix-and-Phenomics |     0.573   |
| TabPFNv2-rdkit                     |     0.542   |
| TabPFNv2-rdkit-3D                  |     0.489   |
| MolEncoder                         |     0.417   |
| CheMeleon                          |     0.36    |

## `tdcommons/clearance-microsome-az`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.609866 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.683    |
| 1B_MPNN_LargeMix-and-Phenomics     |    0.653    |
| TabPFNv2-rdkit-3D                  |    0.649    |
| TabPFNv2-rdkit                     |    0.64     |
| TabularCheMeleon                   |    0.609866 |
| CheMeleon                          |    0.59     |

## `tdcommons/vdss-lombardo`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.712474 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| vidraft-tabpfn-vdss                |    0.726    |
| TabularCheMeleon                   |    0.712474 |
| TabPFNv2-rdkit                     |    0.7      |
| TabPFNv2-rdkit-3D                  |    0.69     |
| 1B_MPNN_LargeMix-and-Phenomics     |    0.637    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.6      |
| CheMeleon                          |    0.59     |

## `tdcommons/caco2-wang`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.301647 |

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
| TabularCheMeleon                          |              0.301647 |
| mini-PlanE-E-BasePlanE-seed42             |              0.314    |
| 3B_e50_MPNN_LargeMix-and-Phenomics        |              0.316    |
| 1B_MPNN_LargeMix-and-Phenomics            |              0.319    |

# Summary

Average Rank of TabularCheMeleon across benchmarks with 4+ other entries (14): 5.36

results_dict = {
    "polaris/pkis2-ret-wt-reg-v2": {
        "mean_squared_error": 732.7922027167672
    },
    "polaris/pkis2-kit-wt-reg-v2": {
        "mean_squared_error": 845.8713689275836
    },
    "polaris/pkis2-egfr-wt-reg-v2": {
        "mean_squared_error": 476.09083660781266
    },
    "polaris/adme-fang-solu-1": {
        "pearsonr": 0.6585395951068622
    },
    "polaris/adme-fang-rppb-1": {
        "pearsonr": 0.7930283357892933
    },
    "polaris/adme-fang-hppb-1": {
        "pearsonr": 0.735830339196074
    },
    "polaris/adme-fang-perm-1": {
        "pearsonr": 0.7895628617738168
    },
    "polaris/adme-fang-rclint-1": {
        "pearsonr": 0.7397686688268965
    },
    "polaris/adme-fang-hclint-1": {
        "pearsonr": 0.7184467286757535
    },
    "tdcommons/lipophilicity-astrazeneca": {
        "mean_absolute_error": 0.5002643788996197
    },
    "tdcommons/ppbr-az": {
        "mean_absolute_error": 7.057839273579005
    },
    "tdcommons/clearance-hepatocyte-az": {
        "spearmanr": 0.397743857740329
    },
    "tdcommons/half-life-obach": {
        "spearmanr": 0.5752596415666381
    },
    "tdcommons/clearance-microsome-az": {
        "spearmanr": 0.6098664221769532
    },
    "tdcommons/vdss-lombardo": {
        "spearmanr": 0.7124738058228707
    },
    "tdcommons/caco2-wang": {
        "mean_absolute_error": 0.3016474253139287
    }
}
