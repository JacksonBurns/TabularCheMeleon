# Eval Results

timestamp: 2026-10-05 18:13:24.705072
checkpoint: tabularchemeleonv2.pt

## `polaris/pkis2-ret-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |       Score |
|---:|:-----------|:---------------|:--------------------|------------:|
|  0 | test       | RET            | mean_squared_error  | 1038        |
|  1 | test       | RET            | r2                  |    0.125828 |
|  2 | test       | RET            | spearmanr           |    0.519291 |
|  3 | test       | RET            | explained_var       |    0.166035 |
|  4 | test       | RET            | pearsonr            |    0.486378 |
|  5 | test       | RET            | mean_absolute_error |   24.3885   |

### Leaderboard Comparison
| Name                                 |   mean_squared_error |
|:-------------------------------------|---------------------:|
| 1B_MolGPS-ens_LargeMix-and-Phenomics |              589.944 |
| 3B_e50_MPNN_LargeMix-and-Phenomics   |              609.399 |
| CheMeleon                            |              684.319 |
| CheMeleon                            |              724.347 |
| TabularCheMeleon                     |             1038     |

## `polaris/pkis2-kit-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |       Score |
|---:|:-----------|:---------------|:--------------------|------------:|
|  0 | test       | KIT            | mean_squared_error  | 1035.52     |
|  1 | test       | KIT            | r2                  |    0.141844 |
|  2 | test       | KIT            | spearmanr           |    0.394071 |
|  3 | test       | KIT            | explained_var       |    0.142978 |
|  4 | test       | KIT            | pearsonr            |    0.418554 |
|  5 | test       | KIT            | mean_absolute_error |   26.2674   |

### Leaderboard Comparison
| Name             |   mean_squared_error |
|:-----------------|---------------------:|
| CheMeleon        |              849.611 |
| TabularCheMeleon |             1035.52  |

## `polaris/pkis2-egfr-wt-reg-v2`

### Model Performance
|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | EGFR           | mean_squared_error  | 687.097    |
|  1 | test       | EGFR           | r2                  |   0.147278 |
|  2 | test       | EGFR           | spearmanr           |   0.259458 |
|  3 | test       | EGFR           | explained_var       |   0.147323 |
|  4 | test       | EGFR           | pearsonr            |   0.383996 |
|  5 | test       | EGFR           | mean_absolute_error |  20.7545   |

### Leaderboard Comparison
| Name                                   |   mean_squared_error |
|:---------------------------------------|---------------------:|
| aether-pharmaos-pkis2-egfr-wt-ensemble |              430.821 |
| CheMeleon                              |              459.929 |
| TabularCheMeleon                       |              687.097 |

## `polaris/adme-fang-solu-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_SOLUBILITY | mean_squared_error  | 0.380481 |
|  1 | test       | LOG_SOLUBILITY | r2                  | 0.298237 |
|  2 | test       | LOG_SOLUBILITY | spearmanr           | 0.463729 |
|  3 | test       | LOG_SOLUBILITY | explained_var       | 0.302055 |
|  4 | test       | LOG_SOLUBILITY | pearsonr            | 0.56994  |
|  5 | test       | LOG_SOLUBILITY | mean_absolute_error | 0.433076 |

### Leaderboard Comparison
| Name                        |   pearsonr |
|:----------------------------|-----------:|
| 1B_MPNN_MolGPS-ens_LargeMix |    0.77    |
| 1B_MPNN_LargeMix-Phenomics  |    0.764   |
| CheMeleonMOE                |    0.729   |
| CheMeleon                   |    0.682   |
| it-works-now                |    0.669   |
| ML4DD-team16                |    0.654   |
| ML4DD-team9                 |    0.651   |
| team9_submission_2          |    0.651   |
| ExactTanimotoGP             |    0.635   |
| ML4DD-team25                |    0.632   |
| TabularCheMeleon            |    0.56994 |

## `polaris/adme-fang-rppb-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_RPPB       | mean_squared_error  | 0.398001 |
|  1 | test       | LOG_RPPB       | r2                  | 0.552044 |
|  2 | test       | LOG_RPPB       | spearmanr           | 0.830435 |
|  3 | test       | LOG_RPPB       | explained_var       | 0.552585 |
|  4 | test       | LOG_RPPB       | pearsonr            | 0.767644 |
|  5 | test       | LOG_RPPB       | mean_absolute_error | 0.488705 |

### Leaderboard Comparison
| Name                                          |   pearsonr |
|:----------------------------------------------|-----------:|
| 1B_MPNN_LargeMix-and-Phenomics                |   0.908    |
| 1B_MPNN_MolGPS-ens_LargeMix                   |   0.886    |
| nepare                                        |   0.863    |
| TabPFNv2-rdkit                                |   0.816    |
| nepare_chemprop                               |   0.78     |
| TabularCheMeleon                              |   0.767644 |
| chemma-2b-sft                                 |   0.747    |
| adme-fang-RPPB-1_desc2D_RandomForestRegressor |   0.722    |
| adme-fang-RPPB-1-GIRAFFE-wae                  |   0.68     |
| CheMeleon                                     |   0.662    |
| chemlactica-1b-sft                            |   0.614    |

## `polaris/adme-fang-hppb-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_HPPB       | mean_squared_error  | 0.255678 |
|  1 | test       | LOG_HPPB       | r2                  | 0.577838 |
|  2 | test       | LOG_HPPB       | spearmanr           | 0.784628 |
|  3 | test       | LOG_HPPB       | explained_var       | 0.621175 |
|  4 | test       | LOG_HPPB       | pearsonr            | 0.788158 |
|  5 | test       | LOG_HPPB       | mean_absolute_error | 0.377935 |

### Leaderboard Comparison
| Name                                            |   pearsonr |
|:------------------------------------------------|-----------:|
| 1B_MPNN_LargeMix-and-Phenomics                  |   0.888    |
| 1B_MPNN_MolGPS-ens_LargeMix                     |   0.884    |
| TabPFNv2-rdkit                                  |   0.827    |
| adme-fang-HPPB-1-GIRAFFE-wae                    |   0.815    |
| agentomics-ml-adme-fang-hppb-1                  |   0.815    |
| nepare                                          |   0.809    |
| CheMeleon                                       |   0.793    |
| TabularCheMeleon                                |   0.788158 |
| chemlactica-125m-sft                            |   0.774    |
| adme-fang-HPPB-1_atompair_RandomForestRegressor |   0.69     |
| chemma-2b-sft                                   |   0.636    |

## `polaris/adme-fang-perm-1`

### Model Performance
|    | Test set   | Target label     | Metric              |    Score |
|---:|:-----------|:-----------------|:--------------------|---------:|
|  0 | test       | LOG_MDR1-MDCK_ER | mean_squared_error  | 0.297814 |
|  1 | test       | LOG_MDR1-MDCK_ER | r2                  | 0.398833 |
|  2 | test       | LOG_MDR1-MDCK_ER | spearmanr           | 0.643148 |
|  3 | test       | LOG_MDR1-MDCK_ER | explained_var       | 0.426322 |
|  4 | test       | LOG_MDR1-MDCK_ER | pearsonr            | 0.655597 |
|  5 | test       | LOG_MDR1-MDCK_ER | mean_absolute_error | 0.406575 |

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
| TabularCheMeleon                              |   0.655597 |

## `polaris/adme-fang-rclint-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_RLM_CLint  | mean_squared_error  | 0.396462 |
|  1 | test       | LOG_RLM_CLint  | r2                  | 0.29773  |
|  2 | test       | LOG_RLM_CLint  | spearmanr           | 0.550171 |
|  3 | test       | LOG_RLM_CLint  | explained_var       | 0.302899 |
|  4 | test       | LOG_RLM_CLint  | pearsonr            | 0.551986 |
|  5 | test       | LOG_RLM_CLint  | mean_absolute_error | 0.511034 |

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
| TabularCheMeleon                                |   0.551986 |
| adme-fang-RCLint-1_desc2D_FCModel               |   0.544    |

## `polaris/adme-fang-hclint-1`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_HLM_CLint  | mean_squared_error  | 0.281841 |
|  1 | test       | LOG_HLM_CLint  | r2                  | 0.274373 |
|  2 | test       | LOG_HLM_CLint  | spearmanr           | 0.551216 |
|  3 | test       | LOG_HLM_CLint  | explained_var       | 0.280386 |
|  4 | test       | LOG_HLM_CLint  | pearsonr            | 0.529797 |
|  5 | test       | LOG_HLM_CLint  | mean_absolute_error | 0.422363 |

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
| TabularCheMeleon                                |   0.529797 |

## `tdcommons/lipophilicity-astrazeneca`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.854417 |

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
| TabularCheMeleon                          |              0.854417 |

## `tdcommons/ppbr-az`

### Model Performance
|    | Test set   | Target label   | Metric              |   Score |
|---:|:-----------|:---------------|:--------------------|--------:|
|  0 | test       | Y              | mean_absolute_error | 9.54081 |

### Leaderboard Comparison
| Name                               |   mean_absolute_error |
|:-----------------------------------|----------------------:|
| 3B_e50_MPNN_LargeMix-and-Phenomics |               7.004   |
| 1B_MPNN_LargeMix-and-Phenomics     |               7.026   |
| TabPFNv2-rdkit-3D                  |               7.033   |
| TabPFNv2-rdkit                     |               7.117   |
| CheMeleon                          |               7.532   |
| TabularCheMeleon                   |               9.54081 |

## `tdcommons/clearance-hepatocyte-az`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.271247 |

### Leaderboard Comparison
| Name                                |   spearmanr |
|:------------------------------------|------------:|
| 1B_MPNN_LargeMix-and-Phenomics      |    0.538    |
| 3B_e50_MPNN_LargeMix-and-Phenomics  |    0.525    |
| clearance-hepatocyte-az-GIRAFFE-wae |    0.429    |
| TabPFNv2-rdkit                      |    0.41     |
| TabPFNv2-rdkit-3D                   |    0.402    |
| CheMeleon                           |    0.387    |
| TabularCheMeleon                    |    0.271247 |

## `tdcommons/half-life-obach`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.275632 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| 1B_MPNN_LargeMix-and-Phenomics     |    0.625    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.573    |
| TabPFNv2-rdkit                     |    0.542    |
| TabPFNv2-rdkit-3D                  |    0.489    |
| MolEncoder                         |    0.417    |
| CheMeleon                          |    0.36     |
| TabularCheMeleon                   |    0.275632 |

## `tdcommons/clearance-microsome-az`

### Model Performance
|    | Test set   | Target label   | Metric    |   Score |
|---:|:-----------|:---------------|:----------|--------:|
|  0 | test       | Y              | spearmanr | 0.48273 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| 3B_e50_MPNN_LargeMix-and-Phenomics |     0.683   |
| 1B_MPNN_LargeMix-and-Phenomics     |     0.653   |
| TabPFNv2-rdkit-3D                  |     0.649   |
| TabPFNv2-rdkit                     |     0.64    |
| CheMeleon                          |     0.59    |
| TabularCheMeleon                   |     0.48273 |

## `tdcommons/vdss-lombardo`

### Model Performance
|    | Test set   | Target label   | Metric    |    Score |
|---:|:-----------|:---------------|:----------|---------:|
|  0 | test       | Y              | spearmanr | 0.475422 |

### Leaderboard Comparison
| Name                               |   spearmanr |
|:-----------------------------------|------------:|
| vidraft-tabpfn-vdss                |    0.726    |
| TabPFNv2-rdkit                     |    0.7      |
| TabPFNv2-rdkit-3D                  |    0.69     |
| 1B_MPNN_LargeMix-and-Phenomics     |    0.637    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |    0.6      |
| CheMeleon                          |    0.59     |
| TabularCheMeleon                   |    0.475422 |

## `tdcommons/caco2-wang`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.522187 |

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
| TabularCheMeleon                          |              0.522187 |

## `tdcommons/ld50-zhu`

### Model Performance
|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | Y              | mean_absolute_error | 0.772467 |

### Leaderboard Comparison
| Name                               |   mean_absolute_error |
|:-----------------------------------|----------------------:|
| CheMeleon                          |              0.526    |
| TabPFNv2-rdkit                     |              0.6      |
| TabPFNv2-rdkit-3D                  |              0.605    |
| 1B_MPNN_LargeMix-and-Phenomics     |              0.614    |
| 3B_e50_MPNN_LargeMix-and-Phenomics |              0.625    |
| TabularCheMeleon                   |              0.772467 |

# Summary

Average Rank of TabularCheMeleon across benchmarks with 4+ other entries (15): 8.20

results_dict = {
    "polaris/pkis2-ret-wt-reg-v2": {
        "mean_squared_error": 1037.996127514081
    },
    "polaris/pkis2-kit-wt-reg-v2": {
        "mean_squared_error": 1035.523705201853
    },
    "polaris/pkis2-egfr-wt-reg-v2": {
        "mean_squared_error": 687.096800249474
    },
    "polaris/adme-fang-solu-1": {
        "pearsonr": 0.5699402211404956
    },
    "polaris/adme-fang-rppb-1": {
        "pearsonr": 0.767643594185686
    },
    "polaris/adme-fang-hppb-1": {
        "pearsonr": 0.788158262970134
    },
    "polaris/adme-fang-perm-1": {
        "pearsonr": 0.6555968176409784
    },
    "polaris/adme-fang-rclint-1": {
        "pearsonr": 0.5519860796616008
    },
    "polaris/adme-fang-hclint-1": {
        "pearsonr": 0.5297974682148052
    },
    "tdcommons/lipophilicity-astrazeneca": {
        "mean_absolute_error": 0.8544173751501809
    },
    "tdcommons/ppbr-az": {
        "mean_absolute_error": 9.540807343348195
    },
    "tdcommons/clearance-hepatocyte-az": {
        "spearmanr": 0.27124671122435706
    },
    "tdcommons/half-life-obach": {
        "spearmanr": 0.2756324202600745
    },
    "tdcommons/clearance-microsome-az": {
        "spearmanr": 0.4827299383920377
    },
    "tdcommons/vdss-lombardo": {
        "spearmanr": 0.47542233531403916
    },
    "tdcommons/caco2-wang": {
        "mean_absolute_error": 0.5221869749885644
    },
    "tdcommons/ld50-zhu": {
        "mean_absolute_error": 0.7724665914865889
    }
}
