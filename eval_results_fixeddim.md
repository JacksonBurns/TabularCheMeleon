# FixedDim-ICL Eval Results

timestamp: 2026-10-08 00:37:44.715225
checkpoint: icl_logs/fixeddim_icl.pt


## `polaris/pkis2-ret-wt-reg-v2`

|    | Test set   | Target label   | Metric              |        Score |
|---:|:-----------|:---------------|:--------------------|-------------:|
|  0 | test       | RET            | spearmanr           |    0.241656  |
|  1 | test       | RET            | explained_var       |    0.0133615 |
|  2 | test       | RET            | mean_squared_error  | 1206.55      |
|  3 | test       | RET            | pearsonr            |    0.117624  |
|  4 | test       | RET            | r2                  |   -0.0161196 |
|  5 | test       | RET            | mean_absolute_error |   26.9585    |

| Name                                 |   mean_squared_error |
|:-------------------------------------|---------------------:|
| 1B_MolGPS-ens_LargeMix-and-Phenomics |              589.944 |
| 3B_e50_MPNN_LargeMix-and-Phenomics   |              609.399 |
| CheMeleon                            |              684.319 |
| CheMeleon                            |              724.347 |
| FixedDimICL                          |             1206.55  |

## `polaris/pkis2-kit-wt-reg-v2`

|    | Test set   | Target label   | Metric              |          Score |
|---:|:-----------|:---------------|:--------------------|---------------:|
|  0 | test       | KIT            | spearmanr           |    0.0660042   |
|  1 | test       | KIT            | explained_var       |    0.000758247 |
|  2 | test       | KIT            | mean_squared_error  | 1262.55        |
|  3 | test       | KIT            | pearsonr            |    0.0462222   |
|  4 | test       | KIT            | r2                  |   -0.0463      |
|  5 | test       | KIT            | mean_absolute_error |   32.142       |

| Name        |   mean_squared_error |
|:------------|---------------------:|
| CheMeleon   |              849.611 |
| FixedDimICL |             1262.55  |

## `polaris/pkis2-egfr-wt-reg-v2`

|    | Test set   | Target label   | Metric              |       Score |
|---:|:-----------|:---------------|:--------------------|------------:|
|  0 | test       | EGFR           | spearmanr           |  -0.0996799 |
|  1 | test       | EGFR           | explained_var       |  -0.0224388 |
|  2 | test       | EGFR           | mean_squared_error  | 832.121     |
|  3 | test       | EGFR           | pearsonr            |  -0.104022  |
|  4 | test       | EGFR           | r2                  |  -0.0327055 |
|  5 | test       | EGFR           | mean_absolute_error |  22.262     |

| Name                                   |   mean_squared_error |
|:---------------------------------------|---------------------:|
| aether-pharmaos-pkis2-egfr-wt-ensemble |              430.821 |
| CheMeleon                              |              459.929 |
| FixedDimICL                            |              832.121 |

## `polaris/adme-fang-solu-1`

|    | Test set   | Target label   | Metric              |       Score |
|---:|:-----------|:---------------|:--------------------|------------:|
|  0 | test       | LOG_SOLUBILITY | spearmanr           |  0.0910165  |
|  1 | test       | LOG_SOLUBILITY | explained_var       |  0.00394964 |
|  2 | test       | LOG_SOLUBILITY | mean_squared_error  |  0.555315   |
|  3 | test       | LOG_SOLUBILITY | pearsonr            |  0.066994   |
|  4 | test       | LOG_SOLUBILITY | r2                  | -0.0242295  |
|  5 | test       | LOG_SOLUBILITY | mean_absolute_error |  0.506961   |

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
| FixedDimICL                 |   0.066994 |

## `polaris/adme-fang-rppb-1`

|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | LOG_RPPB       | spearmanr           | -0.265217  |
|  1 | test       | LOG_RPPB       | explained_var       | -0.0184414 |
|  2 | test       | LOG_RPPB       | mean_squared_error  |  0.919039  |
|  3 | test       | LOG_RPPB       | pearsonr            | -0.174153  |
|  4 | test       | LOG_RPPB       | r2                  | -0.0343922 |
|  5 | test       | LOG_RPPB       | mean_absolute_error |  0.821273  |

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
| FixedDimICL                                   |  -0.174153 |

## `polaris/adme-fang-hppb-1`

|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | LOG_HPPB       | spearmanr           |  0.267553  |
|  1 | test       | LOG_HPPB       | explained_var       |  0.0428316 |
|  2 | test       | LOG_HPPB       | mean_squared_error  |  0.629833  |
|  3 | test       | LOG_HPPB       | pearsonr            |  0.36268   |
|  4 | test       | LOG_HPPB       | r2                  | -0.0399483 |
|  5 | test       | LOG_HPPB       | mean_absolute_error |  0.699342  |

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
| adme-fang-HPPB-1_atompair_RandomForestRegressor |    0.69    |
| chemma-2b-sft                                   |    0.636   |
| FixedDimICL                                     |    0.36268 |

## `polaris/adme-fang-perm-1`

|    | Test set   | Target label     | Metric              |       Score |
|---:|:-----------|:-----------------|:--------------------|------------:|
|  0 | test       | LOG_MDR1-MDCK_ER | spearmanr           |  0.0601156  |
|  1 | test       | LOG_MDR1-MDCK_ER | explained_var       |  0.00267369 |
|  2 | test       | LOG_MDR1-MDCK_ER | mean_squared_error  |  0.508808   |
|  3 | test       | LOG_MDR1-MDCK_ER | pearsonr            |  0.0596856  |
|  4 | test       | LOG_MDR1-MDCK_ER | r2                  | -0.0270809  |
|  5 | test       | LOG_MDR1-MDCK_ER | mean_absolute_error |  0.622509   |

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
| FixedDimICL                                   |  0.0596856 |
