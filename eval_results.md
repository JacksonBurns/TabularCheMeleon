# Eval Results

timestamp: 2026-10-06 18:36:37.511151
checkpoint: tabularchemeleonv3.pt

## `polaris/pkis2-ret-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | RET            | spearmanr           |   0.456117 |
|  1 | test       | RET            | r2                  |   0.170586 |
|  2 | test       | RET            | mean_absolute_error |  24.9269   |
|  3 | test       | RET            | explained_var       |   0.184838 |
|  4 | test       | RET            | pearsonr            |   0.540864 |
|  5 | test       | RET            | mean_squared_error  | 984.851    |

### Leaderboard Comparison
| Name                                 |   mean_squared_error |
|:-------------------------------------|---------------------:|
| 1B_MolGPS-ens_LargeMix-and-Phenomics |              589.944 |
| 3B_e50_MPNN_LargeMix-and-Phenomics   |              609.399 |
| CheMeleon                            |              684.319 |
| CheMeleon                            |              724.347 |
| TabularCheMeleon                     |              984.851 |

## `polaris/pkis2-kit-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |       Score |
|---:|:-----------|:---------------|:--------------------|------------:|
|  0 | test       | KIT            | spearmanr           |    0.372747 |
|  1 | test       | KIT            | r2                  |    0.156965 |
|  2 | test       | KIT            | mean_absolute_error |   27.1037   |
|  3 | test       | KIT            | explained_var       |    0.157848 |
|  4 | test       | KIT            | pearsonr            |    0.405649 |
|  5 | test       | KIT            | mean_squared_error  | 1017.28     |

### Leaderboard Comparison
| Name             |   mean_squared_error |
|:-----------------|---------------------:|
| CheMeleon        |              849.611 |
| TabularCheMeleon |             1017.28  |

## `polaris/pkis2-egfr-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |       Score |
|---:|:-----------|:---------------|:--------------------|------------:|
|  0 | test       | EGFR           | spearmanr           |   0.159321  |
|  1 | test       | EGFR           | r2                  |   0.0999442 |
|  2 | test       | EGFR           | mean_absolute_error |  21.636     |
|  3 | test       | EGFR           | explained_var       |   0.105006  |
|  4 | test       | EGFR           | pearsonr            |   0.325873  |
|  5 | test       | EGFR           | mean_squared_error  | 725.237     |

### Leaderboard Comparison
| Name                                   |   mean_squared_error |
|:---------------------------------------|---------------------:|
| aether-pharmaos-pkis2-egfr-wt-ensemble |              430.821 |
| CheMeleon                              |              459.929 |
| TabularCheMeleon                       |              725.237 |

## `polaris/adme-fang-solu-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_SOLUBILITY | spearmanr           | 0.436123 |
|  1 | test       | LOG_SOLUBILITY | r2                  | 0.235016 |
|  2 | test       | LOG_SOLUBILITY | mean_absolute_error | 0.435392 |
|  3 | test       | LOG_SOLUBILITY | explained_var       | 0.25477  |
|  4 | test       | LOG_SOLUBILITY | pearsonr            | 0.542335 |
|  5 | test       | LOG_SOLUBILITY | mean_squared_error  | 0.414758 |

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
| TabularCheMeleon            |   0.542335 |

## `polaris/adme-fang-rppb-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_RPPB       | spearmanr           | 0.831304 |
|  1 | test       | LOG_RPPB       | r2                  | 0.508305 |
|  2 | test       | LOG_RPPB       | mean_absolute_error | 0.49436  |
|  3 | test       | LOG_RPPB       | explained_var       | 0.508315 |
|  4 | test       | LOG_RPPB       | pearsonr            | 0.730109 |
|  5 | test       | LOG_RPPB       | mean_squared_error  | 0.436862 |

### Leaderboard Comparison
| Name                                          |   pearsonr |
|:----------------------------------------------|-----------:|
| 1B_MPNN_LargeMix-and-Phenomics                |   0.908    |
| 1B_MPNN_MolGPS-ens_LargeMix                   |   0.886    |
| nepare                                        |   0.863    |
| TabPFNv2-rdkit                                |   0.816    |
| nepare_chemprop                               |   0.78     |
| chemma-2b-sft                                 |   0.747    |
| TabularCheMeleon                              |   0.730109 |
| adme-fang-RPPB-1_desc2D_RandomForestRegressor |   0.722    |
| adme-fang-RPPB-1-GIRAFFE-wae                  |   0.68     |
| CheMeleon                                     |   0.662    |
| chemlactica-1b-sft                            |   0.614    |

## `polaris/adme-fang-hppb-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_HPPB       | spearmanr           | 0.787226 |
|  1 | test       | LOG_HPPB       | r2                  | 0.583733 |
|  2 | test       | LOG_HPPB       | mean_absolute_error | 0.3746   |
|  3 | test       | LOG_HPPB       | explained_var       | 0.63359  |
|  4 | test       | LOG_HPPB       | pearsonr            | 0.796005 |
|  5 | test       | LOG_HPPB       | mean_squared_error  | 0.252108 |

### Leaderboard Comparison
| Name                                            |   pearsonr |
|:------------------------------------------------|-----------:|
| 1B_MPNN_LargeMix-and-Phenomics                  |   0.888    |
| 1B_MPNN_MolGPS-ens_LargeMix                     |   0.884    |
| TabPFNv2-rdkit                                  |   0.827    |
| adme-fang-HPPB-1-GIRAFFE-wae                    |   0.815    |
| agentomics-ml-adme-fang-hppb-1                  |   0.815    |
| nepare                                          |   0.809    |
| TabularCheMeleon                                |   0.796005 |
| CheMeleon                                       |   0.793    |
| chemlactica-125m-sft                            |   0.774    |
| adme-fang-HPPB-1_atompair_RandomForestRegressor |   0.69     |
| chemma-2b-sft                                   |   0.636    |

## `polaris/adme-fang-perm-1`

### Model Performance
|    | Test set   | Target label     | Metric              |    Score |
|---:|:-----------|:-----------------|:--------------------|---------:|
|  0 | test       | LOG_MDR1-MDCK_ER | spearmanr           | 0.587483 |
|  1 | test       | LOG_MDR1-MDCK_ER | r2                  | 0.323351 |
|  2 | test       | LOG_MDR1-MDCK_ER | mean_absolute_error | 0.454934 |
|  3 | test       | LOG_MDR1-MDCK_ER | explained_var       | 0.334835 |
|  4 | test       | LOG_MDR1-MDCK_ER | pearsonr            | 0.596681 |
|  5 | test       | LOG_MDR1-MDCK_ER | mean_squared_error  | 0.335207 |

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
| TabularCheMeleon                              |   0.596681 |

## `polaris/adme-fang-rclint-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_RLM_CLint  | spearmanr           | 0.523662 |
|  1 | test       | LOG_RLM_CLint  | r2                  | 0.255537 |
|  2 | test       | LOG_RLM_CLint  | mean_absolute_error | 0.533613 |
|  3 | test       | LOG_RLM_CLint  | explained_var       | 0.256545 |
|  4 | test       | LOG_RLM_CLint  | pearsonr            | 0.511328 |
|  5 | test       | LOG_RLM_CLint  | mean_squared_error  | 0.420281 |

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
| TabularCheMeleon                                |   0.511328 |

## `polaris/adme-fang-hclint-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_HLM_CLint  | spearmanr           | 0.494846 |
|  1 | test       | LOG_HLM_CLint  | r2                  | 0.218967 |
|  2 | test       | LOG_HLM_CLint  | mean_absolute_error | 0.452001 |
|  3 | test       | LOG_HLM_CLint  | explained_var       | 0.226172 |
|  4 | test       | LOG_HLM_CLint  | pearsonr            | 0.486514 |
|  5 | test       | LOG_HLM_CLint  | mean_squared_error  | 0.303361 |

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
| TabularCheMeleon                                |   0.486514 |

## `tdcommons/lipophilicity-astrazeneca`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.827672 |

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
| TabularCheMeleon                          |              0.827672 |

## `tdcommons/ppbr-az`

### Model Performance
|    | Test set   | Target label   | Metric              |   Score |
|---:|:-----------|:---------------|:--------------------|--------:|
|  0 | test       | Y              | mean_absolute_error | 9.76204 |

### Leaderboard Comparison
| Name                               |   mean_absolute_error |
|:-----------------------------------|----------------------:|
| 3B_e50_MPNN_LargeMix-and-Phenomics |               7.004   |
| 1B_MPNN_LargeMix-and-Phenomics     |               7.026   |
| TabPFNv2-rdkit-3D                  |               7.033   |
| TabPFNv2-rdkit                     |               7.117   |
| CheMeleon                          |               7.532   |
| TabularCheMeleon                   |               9.76204 |

## `tdcommons/clearance-hepatocyte-az`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.269654 |

### Leaderboard Comparison
| Name                                |   spearmanr |
|:------------------------------------|------------:|
| 1B_MPNN_LargeMix-and-Phenomics      |    0.538    |
| 3B_e50_MPNN_LargeMix-and-Phenomics  |    0.525    |
| clearance-hepatocyte-az-GIRAFFE-wae |    0.429    |
| TabPFNv2-rdkit                      |    0.41     |
| TabPFNv2-rdkit-3D                   |    0.402    |
| CheMeleon                           |    0.387    |
| TabularCheMeleon                    |    0.269654 |

## `tdcommons/half-life-obach`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.131399 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| 1B_MPNN_LargeMix-and-Phenomics     |    0.625    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.573    |
| TabPFNv2-rdkit                     |    0.542    |
| TabPFNv2-rdkit-3D                  |    0.489    |
| MolEncoder                         |    0.417    |
| CheMeleon                          |    0.36     |
| TabularCheMeleon                   |    0.131399 |

## `tdcommons/clearance-microsome-az`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.429834 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.683    |
| 1B_MPNN_LargeMix-and-Phenomics     |    0.653    |
| TabPFNv2-rdkit-3D                  |    0.649    |
| TabPFNv2-rdkit                     |    0.64     |
| CheMeleon                          |    0.59     |
| TabularCheMeleon                   |    0.429834 |

## `tdcommons/vdss-lombardo`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.317454 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| vidraft-tabpfn-vdss                |    0.726    |
| TabPFNv2-rdkit                     |    0.7      |
| TabPFNv2-rdkit-3D                  |    0.69     |
| 1B_MPNN_LargeMix-and-Phenomics     |    0.637    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.6      |
| CheMeleon                          |    0.59     |
| TabularCheMeleon                   |    0.317454 |

## `tdcommons/caco2-wang`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.392176 |

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
| TabularCheMeleon                          |              0.392176 |

## `tdcommons/ld50-zhu`

### Model Performance
|    | Test set   | Target label   | Metric              |   Score |
|---:|:-----------|:---------------|:--------------------|--------:|
|  0 | test       | Y              | mean_absolute_error |  0.7336 |

### Leaderboard Comparison
| Name                               |   mean_absolute_error |
|:-----------------------------------|----------------------:|
| CheMeleon                          |                0.526  |
| TabPFNv2-rdkit                     |                0.6    |
| TabPFNv2-rdkit-3D                  |                0.605  |
| 1B_MPNN_LargeMix-and-Phenomics     |                0.614  |
| 3B_e50_MPNN_LargeMix-and-Phenomics |                0.625  |
| TabularCheMeleon                   |                0.7336 |

# Summary

Average Rank of TabularCheMeleon across benchmarks with 4+ other entries (15): 8.27

results_dict = {
    "polaris/pkis2-ret-wt-reg-v2": {
        "mean_squared_error": 984.8508528138325
    },
    "polaris/pkis2-kit-wt-reg-v2": {
        "mean_squared_error": 1017.2768585267203
    },
    "polaris/pkis2-egfr-wt-reg-v2": {
        "mean_squared_error": 725.2366257213008
    },
    "polaris/adme-fang-solu-1": {
        "pearsonr": 0.542335002013943
    },
    "polaris/adme-fang-rppb-1": {
        "pearsonr": 0.7301087810563714
    },
    "polaris/adme-fang-hppb-1": {
        "pearsonr": 0.7960050425912507
    },
    "polaris/adme-fang-perm-1": {
        "pearsonr": 0.596681022759891
    },
    "polaris/adme-fang-rclint-1": {
        "pearsonr": 0.5113284429372741
    },
    "polaris/adme-fang-hclint-1": {
        "pearsonr": 0.4865136284535299
    },
    "tdcommons/lipophilicity-astrazeneca": {
        "mean_absolute_error": 0.8276723220064527
    },
    "tdcommons/ppbr-az": {
        "mean_absolute_error": 9.762039288843253
    },
    "tdcommons/clearance-hepatocyte-az": {
        "spearmanr": 0.26965406434044464
    },
    "tdcommons/half-life-obach": {
        "spearmanr": 0.1313994847219221
    },
    "tdcommons/clearance-microsome-az": {
        "spearmanr": 0.42983390298042357
    },
    "tdcommons/vdss-lombardo": {
        "spearmanr": 0.3174543940721411
    },
    "tdcommons/caco2-wang": {
        "mean_absolute_error": 0.392175973895398
    },
    "tdcommons/ld50-zhu": {
        "mean_absolute_error": 0.7336004081869318
    }
}
