# Reimplementation Self-Supervised Vision Transformers for DINO v2 with Huggingface 🤗
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
deepspeed --include localhost:0,1,2,3... vit_train.py
```
```
deepspeed --include localhost:0,1,2,3... convvit_train.py
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
| ViT-S/16-e25 | 25 | 132M | 21M | 22M | 22M | [Link](https://huggingface.co/KevinCha/dinov2-vit-small-remote-sensing) | [logs](https://huggingface.co/KevinCha/dinov2-vit-small-remote-sensing/tensorboard) |
| ViT-S/16-e100 | 25 |  |  |  |  | | |
| ViT-B/16-e25 | 25 | 264M | 88M | 21M | 21M | [Link](https://huggingface.co/KevinCha/dinov2-vit-base-remote-sensing) | [logs](https://huggingface.co/KevinCha/dinov2-vit-base-remote-sensing/tensorboard) |
| ConvViT-S-e25(DINOv1) | 25 | 88.5M | 22.2M | 22M | x | [Link](https://huggingface.co/KevinCha/dinov2-conv-vit-small-remote-sensing) | [logs](https://huggingface.co/KevinCha/dinov2-conv-vit-small-remote-sensing/tensorboard) |

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
# train_textfile = linprob_data_lists/RESISC/train.txt
# test_textfile = linprob_data_lists/RESISC/test.txt

python3 evaluation/linprob.py --model-path {model_registry} \
                              --data-root {data_root} \
                              --train-text {train_textfile} \
                              --test-text {test_textfile}
```

| Model | RESISC | Optimal 31 | MLRSNet | WHU-RS19 | EuroSAT | UC Merced | Cv-BrCT | AiRound | RSI-CB128 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ViT-S/16-e25 | 94.381 | 96.237 | 96.642 | 99.811 | 98.037 | 99.048 | 77.613 | 78.644 | 99.593 |
| ViT-B/16-e25 | 95.460 | 98.925 | 97.301 | 100.00 | 97.889 | 98.571 | 79.058 | 80.339 | 99.675 |
| ConvViT-S-e25(DINOv1) | 94.476 | 93.548 | 95.919 | 99.065 | 96.778 | 98.095 | 77.695 | 81.949 | 99.295 |

### KNN Evaluation

```
# train_textfile = linprob_data_lists/RESISC/train.txt
# test_textfile = linprob_data_lists/RESISC/test.txt

python3 evaluation/knn.py --model-path {model_registry} \
                              --data-root {data_root} \
                              --train-text {train_textfile} \
                              --test-text {test_textfile}
```

| Model | RESISC | Optimal 31 | MLRSNet | WHU-RS19 | EuroSAT | UC Merced | Cv-BrCT | AiRound | RSI-CB128 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| ViT-S/16-e25 | 93.365 | 89.785 | 96.981 | 97.196 | 95.741 | 87.143 | 76.208 | 77.881 | 98.943 |
| ViT-B/16-e25 | 94.286 | 90.323 | 97.328 | 100.00 | 95.704 | 87.143 | 76.456 | 77.373 | 99.106 |
| ConvViT-S-e25(DINOv1) | 92.508 | 91.935 | 95.947 | 98.131 | 94.074 | 90.000 | 75.630 | 76.271 | 98.374 |

---

## Property Analysis

* Feature Mapping - [feature_mapping.ipynb](/notebook/feature_mapping.ipynb)
![feature mapping1](/assets/feature_vis_1.png)
![feature mapping2](/assets/feature_vis_2.png)
* Sparse Feature Matching - [vit-feature-matching.ipynb](/notebook/vit-feature-matching.ipynb)
![sparse matching](/assets/sparse_matching.png)
* Image Retrieval - [index_search.ipynb](/notebook/index_search.ipynb)
![index search1](/assets/1.png)
![index search2](/assets/2.png)
![index search3](/assets/3.png)
![index search4](/assets/4.png)
![index search5](/assets/5.png)
![index search6](/assets/6.png)
![index search7](/assets/7.png)
