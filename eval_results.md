# Eval Results

timestamp: 2026-09-26 16:56:45.642133
checkpoint: tabular_chemeleon_logs/inductive_tabular_chemeleon/version_8/checkpoints/best-epoch=72-val_loss=0.0000.ckpt

## `polaris/pkis2-ret-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |        Score |
|---:|:-----------|:---------------|:--------------------|-------------:|
|  0 | test       | RET            | pearsonr            |    0.493155  |
|  1 | test       | RET            | mean_squared_error  | 1496.33      |
|  2 | test       | RET            | spearmanr           |    0.575844  |
|  3 | test       | RET            | r2                  |   -0.260168  |
|  4 | test       | RET            | explained_var       |    0.0789423 |
|  5 | test       | RET            | mean_absolute_error |   26.2248    |

### Leaderboard Comparison
| Name                                 |   mean_squared_error |
|:-------------------------------------|---------------------:|
| 1B_MolGPS-ens_LargeMix-and-Phenomics |              589.944 |
| 3B_e50_MPNN_LargeMix-and-Phenomics   |              609.399 |
| CheMeleon                            |              684.319 |
| CheMeleon                            |              724.347 |
| TabularCheMeleon                     |             1496.33  |

## `polaris/pkis2-kit-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |         Score |
|---:|:-----------|:---------------|:--------------------|--------------:|
|  0 | test       | KIT            | pearsonr            |    0.380051   |
|  1 | test       | KIT            | mean_squared_error  | 1204.19       |
|  2 | test       | KIT            | spearmanr           |    0.352688   |
|  3 | test       | KIT            | r2                  |    0.00206847 |
|  4 | test       | KIT            | explained_var       |    0.0634284  |
|  5 | test       | KIT            | mean_absolute_error |   27.7107     |

### Leaderboard Comparison
| Name             |   mean_squared_error |
|:-----------------|---------------------:|
| CheMeleon        |              849.611 |
| TabularCheMeleon |             1204.19  |

## `polaris/pkis2-egfr-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |       Score |
|---:|:-----------|:---------------|:--------------------|------------:|
|  0 | test       | EGFR           | pearsonr            |   0.328298  |
|  1 | test       | EGFR           | mean_squared_error  | 794.187     |
|  2 | test       | EGFR           | spearmanr           |   0.24031   |
|  3 | test       | EGFR           | r2                  |   0.0143732 |
|  4 | test       | EGFR           | explained_var       |   0.0788807 |
|  5 | test       | EGFR           | mean_absolute_error |  20.2448    |

### Leaderboard Comparison
| Name                                   |   mean_squared_error |
|:---------------------------------------|---------------------:|
| aether-pharmaos-pkis2-egfr-wt-ensemble |              430.821 |
| CheMeleon                              |              459.929 |
| TabularCheMeleon                       |              794.187 |

## `polaris/adme-fang-solu-1`

### Model Performance
|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | LOG_SOLUBILITY | pearsonr            |  0.297025  |
|  1 | test       | LOG_SOLUBILITY | mean_squared_error  |  0.69074   |
|  2 | test       | LOG_SOLUBILITY | spearmanr           |  0.363951  |
|  3 | test       | LOG_SOLUBILITY | r2                  | -0.274007  |
|  4 | test       | LOG_SOLUBILITY | explained_var       |  0.0492616 |
|  5 | test       | LOG_SOLUBILITY | mean_absolute_error |  0.477015  |

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
| TabularCheMeleon            |   0.297025 |

## `polaris/adme-fang-rppb-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_RPPB       | pearsonr            | 0.717963 |
|  1 | test       | LOG_RPPB       | mean_squared_error  | 0.455267 |
|  2 | test       | LOG_RPPB       | spearmanr           | 0.836522 |
|  3 | test       | LOG_RPPB       | r2                  | 0.48759  |
|  4 | test       | LOG_RPPB       | explained_var       | 0.507791 |
|  5 | test       | LOG_RPPB       | mean_absolute_error | 0.554901 |

### Leaderboard Comparison
| Name                                          |   pearsonr |
|:----------------------------------------------|-----------:|
| 1B_MPNN_LargeMix-and-Phenomics                |   0.908    |
| 1B_MPNN_MolGPS-ens_LargeMix                   |   0.886    |
| nepare                                        |   0.863    |
| TabPFNv2-rdkit                                |   0.816    |
| nepare_chemprop                               |   0.78     |
| chemma-2b-sft                                 |   0.747    |
| adme-fang-RPPB-1_desc2D_RandomForestRegressor |   0.722    |
| TabularCheMeleon                              |   0.717963 |
| adme-fang-RPPB-1-GIRAFFE-wae                  |   0.68     |
| CheMeleon                                     |   0.662    |
| chemlactica-1b-sft                            |   0.614    |

## `polaris/adme-fang-hppb-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_HPPB       | pearsonr            | 0.799961 |
|  1 | test       | LOG_HPPB       | mean_squared_error  | 0.255508 |
|  2 | test       | LOG_HPPB       | spearmanr           | 0.793644 |
|  3 | test       | LOG_HPPB       | r2                  | 0.578118 |
|  4 | test       | LOG_HPPB       | explained_var       | 0.580971 |
|  5 | test       | LOG_HPPB       | mean_absolute_error | 0.418822 |

### Leaderboard Comparison
| Name                                            |   pearsonr |
|:------------------------------------------------|-----------:|
| 1B_MPNN_LargeMix-and-Phenomics                  |   0.888    |
| 1B_MPNN_MolGPS-ens_LargeMix                     |   0.884    |
| TabPFNv2-rdkit                                  |   0.827    |
| adme-fang-HPPB-1-GIRAFFE-wae                    |   0.815    |
| agentomics-ml-adme-fang-hppb-1                  |   0.815    |
| nepare                                          |   0.809    |
| TabularCheMeleon                                |   0.799961 |
| CheMeleon                                       |   0.793    |
| chemlactica-125m-sft                            |   0.774    |
| adme-fang-HPPB-1_atompair_RandomForestRegressor |   0.69     |
| chemma-2b-sft                                   |   0.636    |

## `polaris/adme-fang-perm-1`

### Model Performance
|    | Test set   | Target label     | Metric              |    Score |
|---:|:-----------|:-----------------|:--------------------|---------:|
|  0 | test       | LOG_MDR1-MDCK_ER | pearsonr            | 0.617908 |
|  1 | test       | LOG_MDR1-MDCK_ER | mean_squared_error  | 0.38149  |
|  2 | test       | LOG_MDR1-MDCK_ER | spearmanr           | 0.626979 |
|  3 | test       | LOG_MDR1-MDCK_ER | r2                  | 0.229924 |
|  4 | test       | LOG_MDR1-MDCK_ER | explained_var       | 0.38181  |
|  5 | test       | LOG_MDR1-MDCK_ER | mean_absolute_error | 0.457865 |

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
| TabularCheMeleon                              |   0.617908 |

## `polaris/adme-fang-rclint-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_RLM_CLint  | pearsonr            | 0.511086 |
|  1 | test       | LOG_RLM_CLint  | mean_squared_error  | 0.419806 |
|  2 | test       | LOG_RLM_CLint  | spearmanr           | 0.511779 |
|  3 | test       | LOG_RLM_CLint  | r2                  | 0.256379 |
|  4 | test       | LOG_RLM_CLint  | explained_var       | 0.259312 |
|  5 | test       | LOG_RLM_CLint  | mean_absolute_error | 0.524726 |

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
| TabularCheMeleon                                |   0.511086 |

## `polaris/adme-fang-hclint-1`

### Model Performance
|    | Test set   | Target label   | Metric              |     Score |
|---:|:-----------|:---------------|:--------------------|----------:|
|  0 | test       | LOG_HLM_CLint  | pearsonr            | 0.494956  |
|  1 | test       | LOG_HLM_CLint  | mean_squared_error  | 0.352142  |
|  2 | test       | LOG_HLM_CLint  | spearmanr           | 0.522303  |
|  3 | test       | LOG_HLM_CLint  | r2                  | 0.0933752 |
|  4 | test       | LOG_HLM_CLint  | explained_var       | 0.226706  |
|  5 | test       | LOG_HLM_CLint  | mean_absolute_error | 0.451496  |

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
| TabularCheMeleon                                |   0.494956 |

## `tdcommons/lipophilicity-astrazeneca`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.777911 |

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
| TabularCheMeleon                          |              0.777911 |

## `tdcommons/ppbr-az`

### Model Performance
|    | Test set   | Target label   | Metric              |   Score |
|---:|:-----------|:---------------|:--------------------|--------:|
|  0 | test       | Y              | mean_absolute_error | 9.95698 |

### Leaderboard Comparison
| Name                               |   mean_absolute_error |
|:-----------------------------------|----------------------:|
| 3B_e50_MPNN_LargeMix-and-Phenomics |               7.004   |
| 1B_MPNN_LargeMix-and-Phenomics     |               7.026   |
| TabPFNv2-rdkit-3D                  |               7.033   |
| TabPFNv2-rdkit                     |               7.117   |
| CheMeleon                          |               7.532   |
| TabularCheMeleon                   |               9.95698 |

## `tdcommons/clearance-hepatocyte-az`

### Model Performance
|    | Test set   | Target label   | Metric    |   Score |
|---:|:-----------|:---------------|:----------|--------:|
|  0 | test       | Y              | spearmanr | 0.23672 |

### Leaderboard Comparison
| Name                                |   spearmanr |
|:------------------------------------|------------:|
| 1B_MPNN_LargeMix-and-Phenomics      |     0.538   |
| 3B_e50_MPNN_LargeMix-and-Phenomics  |     0.525   |
| clearance-hepatocyte-az-GIRAFFE-wae |     0.429   |
| TabPFNv2-rdkit                      |     0.41    |
| TabPFNv2-rdkit-3D                   |     0.402   |
| CheMeleon                           |     0.387   |
| TabularCheMeleon                    |     0.23672 |

## `tdcommons/half-life-obach`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.432362 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| 1B_MPNN_LargeMix-and-Phenomics     |    0.625    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.573    |
| TabPFNv2-rdkit                     |    0.542    |
| TabPFNv2-rdkit-3D                  |    0.489    |
| TabularCheMeleon                   |    0.432362 |
| MolEncoder                         |    0.417    |
| CheMeleon                          |    0.36     |

## `tdcommons/clearance-microsome-az`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.498843 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.683    |
| 1B_MPNN_LargeMix-and-Phenomics     |    0.653    |
| TabPFNv2-rdkit-3D                  |    0.649    |
| TabPFNv2-rdkit                     |    0.64     |
| CheMeleon                          |    0.59     |
| TabularCheMeleon                   |    0.498843 |

## `tdcommons/vdss-lombardo`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.537123 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| vidraft-tabpfn-vdss                |    0.726    |
| TabPFNv2-rdkit                     |    0.7      |
| TabPFNv2-rdkit-3D                  |    0.69     |
| 1B_MPNN_LargeMix-and-Phenomics     |    0.637    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.6      |
| CheMeleon                          |    0.59     |
| TabularCheMeleon                   |    0.537123 |

## `tdcommons/caco2-wang`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.491573 |

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
| TabularCheMeleon                          |              0.491573 |

## `tdcommons/ld50-zhu`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.794611 |

### Leaderboard Comparison
| Name                               |   mean_absolute_error |
|:-----------------------------------|----------------------:|
| CheMeleon                          |              0.526    |
| TabPFNv2-rdkit                     |              0.6      |
| TabPFNv2-rdkit-3D                  |              0.605    |
| 1B_MPNN_LargeMix-and-Phenomics     |              0.614    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |              0.625    |
| TabularCheMeleon                   |              0.794611 |

# Summary

Average Rank of TabularCheMeleon across benchmarks with 4+ other entries (15): 8.20

results_dict = {
    "polaris/pkis2-ret-wt-reg-v2": {
        "mean_squared_error": 1496.3296031849375
    },
    "polaris/pkis2-kit-wt-reg-v2": {
        "mean_squared_error": 1204.1880602247647
    },
    "polaris/pkis2-egfr-wt-reg-v2": {
        "mean_squared_error": 794.1869662989852
    },
    "polaris/adme-fang-solu-1": {
        "pearsonr": 0.29702546600840307
    },
    "polaris/adme-fang-rppb-1": {
        "pearsonr": 0.7179627173175539
    },
    "polaris/adme-fang-hppb-1": {
        "pearsonr": 0.7999607806480777
    },
    "polaris/adme-fang-perm-1": {
        "pearsonr": 0.6179082566308102
    },
    "polaris/adme-fang-rclint-1": {
        "pearsonr": 0.5110856081013685
    },
    "polaris/adme-fang-hclint-1": {
        "pearsonr": 0.49495608406387676
    },
    "tdcommons/lipophilicity-astrazeneca": {
        "mean_absolute_error": 0.7779105678399403
    },
    "tdcommons/ppbr-az": {
        "mean_absolute_error": 9.956981549476255
    },
    "tdcommons/clearance-hepatocyte-az": {
        "spearmanr": 0.23671985817952404
    },
    "tdcommons/half-life-obach": {
        "spearmanr": 0.4323617352265313
    },
    "tdcommons/clearance-microsome-az": {
        "spearmanr": 0.49884320620051603
    },
    "tdcommons/vdss-lombardo": {
        "spearmanr": 0.5371231259086886
    },
    "tdcommons/caco2-wang": {
        "mean_absolute_error": 0.49157277791895626
    },
    "tdcommons/ld50-zhu": {
        "mean_absolute_error": 0.7946105123101779
    }
}
