# Reimplementation Self-Supervised Vision Transformers with DINO v2
---
* Pytorch implementation and pretrained models for DINO v2 in remote sensing.
* See Official Paper and Github for information in detail.
[[`arXiv #1`]](https://arxiv.org/abs/2304.07193)
[[`arXiv #2`]](https://arxiv.org/abs/2309.16588)
[[`Github`]](https://github.com/facebookresearch/dinov2)

---
## Training

This project use the deepspeed interface for multi gpu training
```
deepspeed --include localhost:0,1,2,3... train_v2.py
```
---
## Training Dataset for Remote Sensing

| Dataset name | # of corpus | Dataset Paper |
| :-: | :-: | :-: |
| Million-AID | 990,666 | [Link](https://arxiv.org/abs/2006.12485) |
| SkyScript | 5,181,068 | [Link](https://arxiv.org/abs/2312.12856) |
| Total | 6,171,734 | |

---
## Pretrained Model on Huggingface
| Model | Epoch | Total Params | Student Backbone Params | Student DINO Head Params | Student iBOT Head Params | Weight & Config | Logs |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ViT-S/16 | 25 | 132M | 21M | 22M | 22M | [Link](https://huggingface.co/KevinCha/dinov2-vit-small-remote-sensing) | [logs](https://huggingface.co/KevinCha/dinov2-vit-small-remote-sensing/tensorboard) |

---

## Evaluation

The evaluation methods for DINOv2 are k-nn clustering and linear probing. 90% of the data is randomly selected as the training set while the 10% is selected as test set. The `k=20` is selected for evaluation with K-NN. The evaluation datasets are including below table. The splited data is stored in [linprob_data_lists](/linprob_data_lists).

| Dataset Name | Dataset Paper |
| :-: | :-: |
| `RESISC` | [Remote Sensing Image Scene Classification: Benchmark and State of the Art](https://arxiv.org/abs/1703.00121) |
|`Optimal 31` | [Scene Classification With Recurrent Attention of VHR Remote Sensing Images](https://ieeexplore.ieee.org/document/8454883) |
| `MLRSNet`| [MLRSNet: A Multi-label High Spatial Resolution Remote Sensing Dataset for Semantic Scene Understanding](https://arxiv.org/abs/2010.00243) |
| `WHU-RS19` |  |
| `EuroSAT` | [EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://arxiv.org/abs/1709.00029) |
| `UC Merced` | [Bag-of-visual-words and spatial extensions for land-use classification](https://dl.acm.org/doi/10.1145/1869790.1869829) |
| `Cv-BrCT` | [AiRound and CV-BrCT: Novel Multi-View Datasets for Scene Classification](https://arxiv.org/abs/2008.01133) |
| `AiRound`| [AiRound and CV-BrCT: Novel Multi-View Datasets for Scene Classification](https://arxiv.org/abs/2008.01133) |
|`RSI-CB128` | [RSI-CB: A Large Scale Remote Sensing Image Classification Benchmark via Crowdsource Data](https://arxiv.org/abs/1705.10450) |

### Linear Probing Evaluation

```
python3 evaluation/linprob.py --model-path {model_registry} \
                              --data-root {data_root} \
                              --train-text {train_textfile} \
                              --test-text {test_textfile}
```

Example for RESISC
```
python3 evaluation/linprob.py --model-path KevinCha/dinov2-vit-small-remote-sensing \
                              --data-root {YOUR_RESISC_ROOT} \
                              --train-text linprob_data_lists/RESISC/train.txt \
                              --test-text linprob_data_lists/RESISC/test.txt
```

| Model | RESISC | Optimal 31 | MLRSNet | WHU-RS19 | EuroSAT | UC Merced | Cv-BrCT | AiRound |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ViT-Small/16 | | | | | | |

### KNN Evaluation

```
python3 evaluation/knn.py --model-path {model_registry} \
                              --data-root {data_root} \
                              --train-text {train_textfile} \
                              --test-text {test_textfile}
```

Example for RESISC
```
python3 evaluation/knn.py --model-path KevinCha/dinov2-vit-small-remote-sensing \
                              --data-root {YOUR_RESISC_ROOT} \
                              --train-text linprob_data_lists/RESISC/train.txt \
                              --test-text linprob_data_lists/RESISC/test.txt
```

| Model | RESISC | Optimal 31 | MLRSNet | WHU-RS19 | EuroSAT | UC Merced | Cv-BrCT | AiRound |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ViT-Small/16 | | | | | | |

---

## Property Analysis

* Feature Mapping - [feature_mapping.ipynb](/notebook/feature_mapping.ipynb)
* Sparse Feature Matching - [vit-feature-matching.ipynb](/notebook/vit-feature-matching.ipynb)
