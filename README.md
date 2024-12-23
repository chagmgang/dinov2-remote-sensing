# Reimplementation Self-Supervised Vision Transformers with DINO v2
---
* Pytorch implementation and pretrained models for DINO v2 in remote sensing.
* See Official Paper and Github for information in detail.
[[`arXiv #1`]](https://arxiv.org/abs/2304.07193)
[[`arXiv #2`]](https://arxiv.org/abs/2309.16588)
[[`Github`]](https://github.com/facebookresearch/dinov2)

## Training

* Using Deepspeed for Training Interface
    ```
    deepspeed --include localhost:0,1,2,3... train_v2.py
    ```
