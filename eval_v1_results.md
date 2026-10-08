
## `polaris/pkis2-ret-wt-reg-v2`

|    | Test set   | Target label   | Metric              |        Score |
|---:|:-----------|:---------------|:--------------------|-------------:|
|  0 | test       | RET            | explained_var       |   -0.0137751 |
|  1 | test       | RET            | spearmanr           |    0.0158202 |
|  2 | test       | RET            | pearsonr            |   -0.0623289 |
|  3 | test       | RET            | mean_absolute_error |   27.382     |
|  4 | test       | RET            | r2                  |   -0.0472511 |
|  5 | test       | RET            | mean_squared_error  | 1243.51      |

| Name                                 |   mean_squared_error |
|:-------------------------------------|---------------------:|
| 1B_MolGPS-ens_LargeMix-and-Phenomics |              589.944 |
| 3B_e50_MPNN_LargeMix-and-Phenomics   |              609.399 |
| CheMeleon                            |              684.319 |
| CheMeleon                            |              724.347 |
| ScratchICL-v1                        |             1243.51  |

## `polaris/pkis2-kit-wt-reg-v2`

|    | Test set   | Target label   | Metric              |        Score |
|---:|:-----------|:---------------|:--------------------|-------------:|
|  0 | test       | KIT            | explained_var       |    0.0219922 |
|  1 | test       | KIT            | spearmanr           |    0.0987601 |
|  2 | test       | KIT            | pearsonr            |    0.157898  |
|  3 | test       | KIT            | mean_absolute_error |   32.3422    |
|  4 | test       | KIT            | r2                  |   -0.05765   |
|  5 | test       | KIT            | mean_squared_error  | 1276.25      |

| Name          |   mean_squared_error |
|:--------------|---------------------:|
| CheMeleon     |              849.611 |
| ScratchICL-v1 |             1276.25  |

## `polaris/pkis2-egfr-wt-reg-v2`

|    | Test set   | Target label   | Metric              |        Score |
|---:|:-----------|:---------------|:--------------------|-------------:|
|  0 | test       | EGFR           | explained_var       |   0.00992549 |
|  1 | test       | EGFR           | spearmanr           |   0.0881397  |
|  2 | test       | EGFR           | pearsonr            |   0.10015    |
|  3 | test       | EGFR           | mean_absolute_error |  20.9862     |
|  4 | test       | EGFR           | r2                  |   0.0098637  |
|  5 | test       | EGFR           | mean_squared_error  | 797.821      |

| Name                                   |   mean_squared_error |
|:---------------------------------------|---------------------:|
| aether-pharmaos-pkis2-egfr-wt-ensemble |              430.821 |
| CheMeleon                              |              459.929 |
| ScratchICL-v1                          |              797.821 |

## `polaris/adme-fang-solu-1`

|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | LOG_SOLUBILITY | explained_var       | -0.016109  |
|  1 | test       | LOG_SOLUBILITY | spearmanr           | -0.0171379 |
|  2 | test       | LOG_SOLUBILITY | pearsonr            | -0.091491  |
|  3 | test       | LOG_SOLUBILITY | mean_absolute_error |  0.485542  |
|  4 | test       | LOG_SOLUBILITY | r2                  | -0.0865944 |
|  5 | test       | LOG_SOLUBILITY | mean_squared_error  |  0.589128  |

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
| ScratchICL-v1               |  -0.091491 |

## `polaris/adme-fang-rppb-1`

|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | LOG_RPPB       | explained_var       | -0.0184415 |
|  1 | test       | LOG_RPPB       | spearmanr           | -0.265217  |
|  2 | test       | LOG_RPPB       | pearsonr            | -0.174154  |
|  3 | test       | LOG_RPPB       | mean_absolute_error |  0.821273  |
|  4 | test       | LOG_RPPB       | r2                  | -0.0343923 |
|  5 | test       | LOG_RPPB       | mean_squared_error  |  0.919039  |

| Name                                          |   pearsonr |
|:----------------------------------------------|-----------:|
| 1B_MPNN_LargeMix-and-Phenomics                |   0.908    |
| 1B_MPNN_MolGPS-ens_LargeMix                   |   0.886    |
| nepare                                        |   0.863    |
| TabPFNv2-rdkit                                |   0.816    |
| nepare_chemprop                               |   0.78     |
| chemma-2b-sft                                 |   0.747    |
| adme-fang-RPPB-1_desc2D_RandomForestRegressor |   0.722    |
| adme-fang-RPPB-1-GIRAFFE-wae                  |   0.68     |
| CheMeleon                                     |   0.662    |
| chemlactica-1b-sft                            |   0.614    |
| ScratchICL-v1                                 |  -0.174154 |

## `polaris/adme-fang-hppb-1`

|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | LOG_HPPB       | explained_var       |  0.0428315 |
|  1 | test       | LOG_HPPB       | spearmanr           |  0.267553  |
|  2 | test       | LOG_HPPB       | pearsonr            |  0.362679  |
|  3 | test       | LOG_HPPB       | mean_absolute_error |  0.699342  |
|  4 | test       | LOG_HPPB       | r2                  | -0.0399483 |
|  5 | test       | LOG_HPPB       | mean_squared_error  |  0.629833  |

| Name                                            |   pearsonr |
|:------------------------------------------------|-----------:|
| 1B_MPNN_LargeMix-and-Phenomics                  |   0.888    |
| 1B_MPNN_MolGPS-ens_LargeMix                     |   0.884    |
| TabPFNv2-rdkit                                  |   0.827    |
| adme-fang-HPPB-1-GIRAFFE-wae                    |   0.815    |
| agentomics-ml-adme-fang-hppb-1                  |   0.815    |
| nepare                                          |   0.809    |
| CheMeleon                                       |   0.793    |
| chemlactica-125m-sft                            |   0.774    |
| adme-fang-HPPB-1_atompair_RandomForestRegressor |   0.69     |
| chemma-2b-sft                                   |   0.636    |
| ScratchICL-v1                                   |   0.362679 |

## `polaris/adme-fang-perm-1`

|    | Test set   | Target label     | Metric              |       Score |
|---:|:-----------|:-----------------|:--------------------|------------:|
|  0 | test       | LOG_MDR1-MDCK_ER | explained_var       | -0.00350646 |
|  1 | test       | LOG_MDR1-MDCK_ER | spearmanr           |  0.0463428  |
|  2 | test       | LOG_MDR1-MDCK_ER | pearsonr            |  0.0265523  |
|  3 | test       | LOG_MDR1-MDCK_ER | mean_absolute_error |  0.589611   |
|  4 | test       | LOG_MDR1-MDCK_ER | r2                  | -0.00402104 |
|  5 | test       | LOG_MDR1-MDCK_ER | mean_squared_error  |  0.497385   |

| Name                                          |   pearsonr |
|:----------------------------------------------|-----------:|
| 1B_MPNN_MolGPS-ens_LargeMix                   |  0.879     |
| 1B_MPNN_LargeMix-and-Phenomics                |  0.86      |
| CheMeleon                                     |  0.822     |
| MolEncoder                                    |  0.804     |
| TabPFNv2-rdkit                                |  0.798     |
| adme-fang-PERM-1-GIRAFFE-wae_s                |  0.772     |
| chemlactica-1b-sft                            |  0.762     |
| optimized-random-forest-rdkit-descriptors     |  0.727     |
| adme-fang-PERM-1_desc2D_RandomForestRegressor |  0.716     |
| chemlactica-125m-sft                          |  0.714     |
| ScratchICL-v1                                 |  0.0265523 |

## `polaris/adme-fang-rclint-1`

|    | Test set   | Target label   | Metric              |       Score |
|---:|:-----------|:---------------|:--------------------|------------:|
|  0 | test       | LOG_RLM_CLint  | explained_var       | -0.00392814 |
|  1 | test       | LOG_RLM_CLint  | spearmanr           | -0.00989619 |
|  2 | test       | LOG_RLM_CLint  | pearsonr            | -0.0103459  |
|  3 | test       | LOG_RLM_CLint  | mean_absolute_error |  0.636244   |
|  4 | test       | LOG_RLM_CLint  | r2                  | -0.0062736  |
|  5 | test       | LOG_RLM_CLint  | mean_squared_error  |  0.568085   |

| Name                                            |   pearsonr |
|:------------------------------------------------|-----------:|
| 1B_MPNN_MolGPS-ens_LargeMix                     |  0.798     |
| 1B_MPNN_LargeMix-and-Phenomics                  |  0.784     |
| CheMeleon                                       |  0.757     |
| chemlactica-125m-sft                            |  0.714     |
| chemlactica-1b-sft                              |  0.698     |
| TabPFNv2-rdkit                                  |  0.694     |
| chemma-2b-sft                                   |  0.66      |
| adme-fang-RCLint-1_desc2D_RandomForestRegressor |  0.64      |
| adme-fang-RCLint-1_desc2D_RandomForestRegressor |  0.631     |
| adme-fang-RCLint-1_desc2D_FCModel               |  0.544     |
| ScratchICL-v1                                   | -0.0103459 |

## `polaris/adme-fang-hclint-1`

|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | LOG_HLM_CLint  | explained_var       | -0.0303225 |
|  1 | test       | LOG_HLM_CLint  | spearmanr           | -0.0965985 |
|  2 | test       | LOG_HLM_CLint  | pearsonr            | -0.0936099 |
|  3 | test       | LOG_HLM_CLint  | mean_absolute_error |  0.535297  |
|  4 | test       | LOG_HLM_CLint  | r2                  | -0.0318663 |
|  5 | test       | LOG_HLM_CLint  | mean_squared_error  |  0.400787  |

| Name                                            |   pearsonr |
|:------------------------------------------------|-----------:|
| seqera-gradient                                 |  0.796     |
| 1B_MPNN_LargeMix-and-Phenomics                  |  0.778     |
| chemlactica-1b-sft                              |  0.72      |
| CheMeleon                                       |  0.72      |
| chemlactica-125m-sft                            |  0.717     |
| MolEncoder                                      |  0.714     |
| agentomics-ml-adme-fang-hclint-1                |  0.697     |
| chemma-2b-sft                                   |  0.674     |
| TabPFNv2-rdkit                                  |  0.662     |
| adme-fang-HCLint-1_desc2D_RandomForestRegressor |  0.639     |
| ScratchICL-v1                                   | -0.0936099 |

# Summary

Average rank (7): 10.14
