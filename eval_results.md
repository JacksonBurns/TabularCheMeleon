# Eval Results

timestamp: 2026-10-04 18:31:41.583546
checkpoint: estes_logs/tabular_chemeleon_logs/inductive_tabular_chemeleon/version_3/checkpoints/best-epoch=291-val_loss=0.0000.ckpt

## `polaris/pkis2-ret-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |          Score |
|---:|:-----------|:---------------|:--------------------|---------------:|
|  0 | test       | RET            | explained_var       |    0.000379093 |
|  1 | test       | RET            | pearsonr            |    0.0220013   |
|  2 | test       | RET            | spearmanr           |    0.103578    |
|  3 | test       | RET            | mean_absolute_error |   31.5738      |
|  4 | test       | RET            | mean_squared_error  | 2162.94        |
|  5 | test       | RET            | r2                  |   -0.821569    |

### Leaderboard Comparison
| Name                                 |   mean_squared_error |
|:-------------------------------------|---------------------:|
| 1B_MolGPS-ens_LargeMix-and-Phenomics |              589.944 |
| 3B_e50_MPNN_LargeMix-and-Phenomics   |              609.399 |
| CheMeleon                            |              684.319 |
| CheMeleon                            |              724.347 |
| TabularCheMeleon                     |             2162.94  |

## `polaris/pkis2-kit-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |       Score |
|---:|:-----------|:---------------|:--------------------|------------:|
|  0 | test       | KIT            | explained_var       |    0.219179 |
|  1 | test       | KIT            | pearsonr            |    0.468197 |
|  2 | test       | KIT            | spearmanr           |    0.418719 |
|  3 | test       | KIT            | mean_absolute_error |   31.4364   |
|  4 | test       | KIT            | mean_squared_error  | 1697.07     |
|  5 | test       | KIT            | r2                  |   -0.406395 |

### Leaderboard Comparison
| Name             |   mean_squared_error |
|:-----------------|---------------------:|
| CheMeleon        |              849.611 |
| TabularCheMeleon |             1697.07  |

## `polaris/pkis2-egfr-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |       Score |
|---:|:-----------|:---------------|:--------------------|------------:|
|  0 | test       | EGFR           | explained_var       |   0.0299943 |
|  1 | test       | EGFR           | pearsonr            |   0.215586  |
|  2 | test       | EGFR           | spearmanr           |   0.134038  |
|  3 | test       | EGFR           | mean_absolute_error |  21.5575    |
|  4 | test       | EGFR           | mean_squared_error  | 984.474     |
|  5 | test       | EGFR           | r2                  |  -0.221782  |

### Leaderboard Comparison
| Name                                   |   mean_squared_error |
|:---------------------------------------|---------------------:|
| aether-pharmaos-pkis2-egfr-wt-ensemble |              430.821 |
| CheMeleon                              |              459.929 |
| TabularCheMeleon                       |              984.474 |

## `polaris/adme-fang-solu-1`

### Model Performance
|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | LOG_SOLUBILITY | explained_var       |  0.0235352 |
|  1 | test       | LOG_SOLUBILITY | pearsonr            |  0.240768  |
|  2 | test       | LOG_SOLUBILITY | spearmanr           |  0.328962  |
|  3 | test       | LOG_SOLUBILITY | mean_absolute_error |  0.523614  |
|  4 | test       | LOG_SOLUBILITY | mean_squared_error  |  0.781976  |
|  5 | test       | LOG_SOLUBILITY | r2                  | -0.442286  |

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
| TabularCheMeleon            |   0.240768 |

## `polaris/adme-fang-rppb-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_RPPB       | explained_var       | 0.55676  |
|  1 | test       | LOG_RPPB       | pearsonr            | 0.766917 |
|  2 | test       | LOG_RPPB       | spearmanr           | 0.82     |
|  3 | test       | LOG_RPPB       | mean_absolute_error | 0.525673 |
|  4 | test       | LOG_RPPB       | mean_squared_error  | 0.440899 |
|  5 | test       | LOG_RPPB       | r2                  | 0.503761 |

### Leaderboard Comparison
| Name                                          |   pearsonr |
|:----------------------------------------------|-----------:|
| 1B_MPNN_LargeMix-and-Phenomics                |   0.908    |
| 1B_MPNN_MolGPS-ens_LargeMix                   |   0.886    |
| nepare                                        |   0.863    |
| TabPFNv2-rdkit                                |   0.816    |
| nepare_chemprop                               |   0.78     |
| TabularCheMeleon                              |   0.766917 |
| chemma-2b-sft                                 |   0.747    |
| adme-fang-RPPB-1_desc2D_RandomForestRegressor |   0.722    |
| adme-fang-RPPB-1-GIRAFFE-wae                  |   0.68     |
| CheMeleon                                     |   0.662    |
| chemlactica-1b-sft                            |   0.614    |

## `polaris/adme-fang-hppb-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_HPPB       | explained_var       | 0.620655 |
|  1 | test       | LOG_HPPB       | pearsonr            | 0.819839 |
|  2 | test       | LOG_HPPB       | spearmanr           | 0.794713 |
|  3 | test       | LOG_HPPB       | mean_absolute_error | 0.391769 |
|  4 | test       | LOG_HPPB       | mean_squared_error  | 0.231752 |
|  5 | test       | LOG_HPPB       | r2                  | 0.617343 |

### Leaderboard Comparison
| Name                                            |   pearsonr |
|:------------------------------------------------|-----------:|
| 1B_MPNN_LargeMix-and-Phenomics                  |   0.888    |
| 1B_MPNN_MolGPS-ens_LargeMix                     |   0.884    |
| TabPFNv2-rdkit                                  |   0.827    |
| TabularCheMeleon                                |   0.819839 |
| adme-fang-HPPB-1-GIRAFFE-wae                    |   0.815    |
| agentomics-ml-adme-fang-hppb-1                  |   0.815    |
| nepare                                          |   0.809    |
| CheMeleon                                       |   0.793    |
| chemlactica-125m-sft                            |   0.774    |
| adme-fang-HPPB-1_atompair_RandomForestRegressor |   0.69     |
| chemma-2b-sft                                   |   0.636    |

## `polaris/adme-fang-perm-1`

### Model Performance
|    | Test set   | Target label     | Metric              |     Score |
|---:|:-----------|:-----------------|:--------------------|----------:|
|  0 | test       | LOG_MDR1-MDCK_ER | explained_var       |  0.286934 |
|  1 | test       | LOG_MDR1-MDCK_ER | pearsonr            |  0.633542 |
|  2 | test       | LOG_MDR1-MDCK_ER | spearmanr           |  0.658083 |
|  3 | test       | LOG_MDR1-MDCK_ER | mean_absolute_error |  0.644856 |
|  4 | test       | LOG_MDR1-MDCK_ER | mean_squared_error  |  0.750993 |
|  5 | test       | LOG_MDR1-MDCK_ER | r2                  | -0.515954 |

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
| TabularCheMeleon                              |   0.633542 |

## `polaris/adme-fang-rclint-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_RLM_CLint  | explained_var       | 0.267962 |
|  1 | test       | LOG_RLM_CLint  | pearsonr            | 0.523349 |
|  2 | test       | LOG_RLM_CLint  | spearmanr           | 0.519928 |
|  3 | test       | LOG_RLM_CLint  | mean_absolute_error | 0.52084  |
|  4 | test       | LOG_RLM_CLint  | mean_squared_error  | 0.414661 |
|  5 | test       | LOG_RLM_CLint  | r2                  | 0.265493 |

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
| TabularCheMeleon                                |   0.523349 |

## `polaris/adme-fang-hclint-1`

### Model Performance
|    | Test set   | Target label   | Metric              |     Score |
|---:|:-----------|:---------------|:--------------------|----------:|
|  0 | test       | LOG_HLM_CLint  | explained_var       |  0.255133 |
|  1 | test       | LOG_HLM_CLint  | pearsonr            |  0.520715 |
|  2 | test       | LOG_HLM_CLint  | spearmanr           |  0.530411 |
|  3 | test       | LOG_HLM_CLint  | mean_absolute_error |  0.535642 |
|  4 | test       | LOG_HLM_CLint  | mean_squared_error  |  0.504707 |
|  5 | test       | LOG_HLM_CLint  | r2                  | -0.299418 |

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
| TabularCheMeleon                                |   0.520715 |

## `tdcommons/lipophilicity-astrazeneca`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.821333 |

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
| TabularCheMeleon                          |              0.821333 |

## `tdcommons/ppbr-az`

### Model Performance
|    | Test set   | Target label   | Metric              |   Score |
|---:|:-----------|:---------------|:--------------------|--------:|
|  0 | test       | Y              | mean_absolute_error | 12.0787 |

### Leaderboard Comparison
| Name                               |   mean_absolute_error |
|:-----------------------------------|----------------------:|
| 3B_e50_MPNN_LargeMix-and-Phenomics |                7.004  |
| 1B_MPNN_LargeMix-and-Phenomics     |                7.026  |
| TabPFNv2-rdkit-3D                  |                7.033  |
| TabPFNv2-rdkit                     |                7.117  |
| CheMeleon                          |                7.532  |
| TabularCheMeleon                   |               12.0787 |

## `tdcommons/clearance-hepatocyte-az`

### Model Performance
|    | Test set   | Target label   | Metric    |     Score |
|---:|:-----------|:---------------|:----------|----------:|
|  0 | test       | Y              | spearmanr | 0.0685341 |

### Leaderboard Comparison
| Name                                |   spearmanr |
|:------------------------------------|------------:|
| 1B_MPNN_LargeMix-and-Phenomics      |   0.538     |
| 3B_e50_MPNN_LargeMix-and-Phenomics  |   0.525     |
| clearance-hepatocyte-az-GIRAFFE-wae |   0.429     |
| TabPFNv2-rdkit                      |   0.41      |
| TabPFNv2-rdkit-3D                   |   0.402     |
| CheMeleon                           |   0.387     |
| TabularCheMeleon                    |   0.0685341 |

## `tdcommons/half-life-obach`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.445803 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| 1B_MPNN_LargeMix-and-Phenomics     |    0.625    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.573    |
| TabPFNv2-rdkit                     |    0.542    |
| TabPFNv2-rdkit-3D                  |    0.489    |
| TabularCheMeleon                   |    0.445803 |
| MolEncoder                         |    0.417    |
| CheMeleon                          |    0.36     |

## `tdcommons/clearance-microsome-az`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.372066 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.683    |
| 1B_MPNN_LargeMix-and-Phenomics     |    0.653    |
| TabPFNv2-rdkit-3D                  |    0.649    |
| TabPFNv2-rdkit                     |    0.64     |
| CheMeleon                          |    0.59     |
| TabularCheMeleon                   |    0.372066 |

## `tdcommons/vdss-lombardo`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.402099 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| vidraft-tabpfn-vdss                |    0.726    |
| TabPFNv2-rdkit                     |    0.7      |
| TabPFNv2-rdkit-3D                  |    0.69     |
| 1B_MPNN_LargeMix-and-Phenomics     |    0.637    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.6      |
| CheMeleon                          |    0.59     |
| TabularCheMeleon                   |    0.402099 |

## `tdcommons/caco2-wang`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.494025 |

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
| TabularCheMeleon                          |              0.494025 |

## `tdcommons/ld50-zhu`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.976575 |

### Leaderboard Comparison
| Name                               |   mean_absolute_error |
|:-----------------------------------|----------------------:|
| CheMeleon                          |              0.526    |
| TabPFNv2-rdkit                     |              0.6      |
| TabPFNv2-rdkit-3D                  |              0.605    |
| 1B_MPNN_LargeMix-and-Phenomics     |              0.614    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |              0.625    |
| TabularCheMeleon                   |              0.976575 |

# Summary

Average Rank of TabularCheMeleon across benchmarks with 4+ other entries (15): 7.87

results_dict = {
    "polaris/pkis2-ret-wt-reg-v2": {
        "mean_squared_error": 2162.9399898730235
    },
    "polaris/pkis2-kit-wt-reg-v2": {
        "mean_squared_error": 1697.0745135529353
    },
    "polaris/pkis2-egfr-wt-reg-v2": {
        "mean_squared_error": 984.4735908275223
    },
    "polaris/adme-fang-solu-1": {
        "pearsonr": 0.24076762286591644
    },
    "polaris/adme-fang-rppb-1": {
        "pearsonr": 0.7669169212684463
    },
    "polaris/adme-fang-hppb-1": {
        "pearsonr": 0.8198394677566654
    },
    "polaris/adme-fang-perm-1": {
        "pearsonr": 0.6335422609233576
    },
    "polaris/adme-fang-rclint-1": {
        "pearsonr": 0.5233485109803212
    },
    "polaris/adme-fang-hclint-1": {
        "pearsonr": 0.5207149038560958
    },
    "tdcommons/lipophilicity-astrazeneca": {
        "mean_absolute_error": 0.8213333904913493
    },
    "tdcommons/ppbr-az": {
        "mean_absolute_error": 12.078676796573646
    },
    "tdcommons/clearance-hepatocyte-az": {
        "spearmanr": 0.06853406147182572
    },
    "tdcommons/half-life-obach": {
        "spearmanr": 0.4458034118236887
    },
    "tdcommons/clearance-microsome-az": {
        "spearmanr": 0.3720663352394473
    },
    "tdcommons/vdss-lombardo": {
        "spearmanr": 0.40209865879425905
    },
    "tdcommons/caco2-wang": {
        "mean_absolute_error": 0.494025012765369
    },
    "tdcommons/ld50-zhu": {
        "mean_absolute_error": 0.9765750817942844
    }
}
