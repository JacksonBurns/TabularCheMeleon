# TabICL-v2 (CheMeleon embeddings) Baseline

timestamp: 2026-10-08 05:12:58.561092


## `polaris/pkis2-ret-wt-reg-v2`

|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | RET            | mean_squared_error  | 656.012    |
|  1 | test       | RET            | spearmanr           |   0.653887 |
|  2 | test       | RET            | r2                  |   0.447525 |
|  3 | test       | RET            | explained_var       |   0.462052 |
|  4 | test       | RET            | pearsonr            |   0.683805 |
|  5 | test       | RET            | mean_absolute_error |  19.246    |

| Name                                 |   mean_squared_error |
|:-------------------------------------|---------------------:|
| 1B_MolGPS-ens_LargeMix-and-Phenomics |              589.944 |
| 3B_e50_MPNN_LargeMix-and-Phenomics   |              609.399 |
| TabICL-CheMeleon                     |              656.012 |
| CheMeleon                            |              684.319 |
| CheMeleon                            |              724.347 |

## `polaris/pkis2-kit-wt-reg-v2`

|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | KIT            | mean_squared_error  | 863.643    |
|  1 | test       | KIT            | spearmanr           |   0.483716 |
|  2 | test       | KIT            | r2                  |   0.284284 |
|  3 | test       | KIT            | explained_var       |   0.292597 |
|  4 | test       | KIT            | pearsonr            |   0.555904 |
|  5 | test       | KIT            | mean_absolute_error |  22.5676   |

| Name             |   mean_squared_error |
|:-----------------|---------------------:|
| CheMeleon        |              849.611 |
| TabICL-CheMeleon |              863.643 |

## `polaris/pkis2-egfr-wt-reg-v2`

|    | Test set   | Target label   | Metric              |      Score |
|---:|:-----------|:---------------|:--------------------|-----------:|
|  0 | test       | EGFR           | mean_squared_error  | 398.603    |
|  1 | test       | EGFR           | spearmanr           |   0.46981  |
|  2 | test       | EGFR           | r2                  |   0.505314 |
|  3 | test       | EGFR           | explained_var       |   0.505373 |
|  4 | test       | EGFR           | pearsonr            |   0.728127 |
|  5 | test       | EGFR           | mean_absolute_error |  15.4974   |

| Name                                   |   mean_squared_error |
|:---------------------------------------|---------------------:|
| TabICL-CheMeleon                       |              398.603 |
| aether-pharmaos-pkis2-egfr-wt-ensemble |              430.821 |
| CheMeleon                              |              459.929 |

## `polaris/adme-fang-solu-1`

|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_SOLUBILITY | mean_squared_error  | 0.359544 |
|  1 | test       | LOG_SOLUBILITY | spearmanr           | 0.524196 |
|  2 | test       | LOG_SOLUBILITY | r2                  | 0.336852 |
|  3 | test       | LOG_SOLUBILITY | explained_var       | 0.347307 |
|  4 | test       | LOG_SOLUBILITY | pearsonr            | 0.603288 |
|  5 | test       | LOG_SOLUBILITY | mean_absolute_error | 0.403286 |

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
| TabICL-CheMeleon            |   0.603288 |

## `polaris/adme-fang-rppb-1`

|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_RPPB       | mean_squared_error  | 0.375959 |
|  1 | test       | LOG_RPPB       | spearmanr           | 0.797391 |
|  2 | test       | LOG_RPPB       | r2                  | 0.576853 |
|  3 | test       | LOG_RPPB       | explained_var       | 0.576864 |
|  4 | test       | LOG_RPPB       | pearsonr            | 0.778875 |
|  5 | test       | LOG_RPPB       | mean_absolute_error | 0.467318 |

| Name                                          |   pearsonr |
|:----------------------------------------------|-----------:|
| 1B_MPNN_LargeMix-and-Phenomics                |   0.908    |
| 1B_MPNN_MolGPS-ens_LargeMix                   |   0.886    |
| nepare                                        |   0.863    |
| TabPFNv2-rdkit                                |   0.816    |
| nepare_chemprop                               |   0.78     |
| TabICL-CheMeleon                              |   0.778875 |
| chemma-2b-sft                                 |   0.747    |
| adme-fang-RPPB-1_desc2D_RandomForestRegressor |   0.722    |
| adme-fang-RPPB-1-GIRAFFE-wae                  |   0.68     |
| CheMeleon                                     |   0.662    |
| chemlactica-1b-sft                            |   0.614    |

## `polaris/adme-fang-hppb-1`

|    | Test set   | Target label   | Metric              |    Score |
|---:|:-----------|:---------------|:--------------------|---------:|
|  0 | test       | LOG_HPPB       | mean_squared_error  | 0.302388 |
|  1 | test       | LOG_HPPB       | spearmanr           | 0.685461 |
|  2 | test       | LOG_HPPB       | r2                  | 0.500713 |
|  3 | test       | LOG_HPPB       | explained_var       | 0.547019 |
|  4 | test       | LOG_HPPB       | pearsonr            | 0.740368 |
|  5 | test       | LOG_HPPB       | mean_absolute_error | 0.447057 |

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
| TabICL-CheMeleon                                |   0.740368 |
| adme-fang-HPPB-1_atompair_RandomForestRegressor |   0.69     |
| chemma-2b-sft                                   |   0.636    |

## `polaris/adme-fang-perm-1`

|    | Test set   | Target label     | Metric              |    Score |
|---:|:-----------|:-----------------|:--------------------|---------:|
|  0 | test       | LOG_MDR1-MDCK_ER | mean_squared_error  | 0.22543  |
|  1 | test       | LOG_MDR1-MDCK_ER | spearmanr           | 0.718242 |
|  2 | test       | LOG_MDR1-MDCK_ER | r2                  | 0.544947 |
|  3 | test       | LOG_MDR1-MDCK_ER | explained_var       | 0.545557 |
|  4 | test       | LOG_MDR1-MDCK_ER | pearsonr            | 0.744333 |
|  5 | test       | LOG_MDR1-MDCK_ER | mean_absolute_error | 0.357777 |

| Name                                          |   pearsonr |
|:----------------------------------------------|-----------:|
| 1B_MPNN_MolGPS-ens_LargeMix                   |   0.879    |
| 1B_MPNN_LargeMix-and-Phenomics                |   0.86     |
| CheMeleon                                     |   0.822    |
| MolEncoder                                    |   0.804    |
| TabPFNv2-rdkit                                |   0.798    |
| adme-fang-PERM-1-GIRAFFE-wae_s                |   0.772    |
| chemlactica-1b-sft                            |   0.762    |
| TabICL-CheMeleon                              |   0.744333 |
| optimized-random-forest-rdkit-descriptors     |   0.727    |
| adme-fang-PERM-1_desc2D_RandomForestRegressor |   0.716    |
| chemlactica-125m-sft                          |   0.714    |
