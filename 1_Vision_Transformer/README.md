# FlashBias for Vision Transformer
FlashBias can be used to accelerate the computation of the [Swin Transformer](https://github.com/microsoft/Swin-Transformer), which encodes relative position information using optimized model parameters to guide attention learning.

<u>**Implementation**:</u> This work is based on the **SVD version of FlashBias**, where we compute the SVD of the relative position matrix and load the decomposed factor tensors as $\mathbf{q}_{\text{bias}}$ and $\mathbf{k}_{\text{bias}}$ in FlashBias.

<p align="center">
<img src="..\pic\low_rank_swin.png" height = "220" alt="" align=center />
<br><br>
<b>Figure 1.</b> Visualization of bias matrix in SwinV2-B. R can maintain 99.5% energy of the bias matrix..
</p>

## Get Started

This part focuses on the [SwinV2-B](https://github.com/microsoft/Swin-Transformer/blob/main/configs/swinv2/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.yaml).

1. Pre-compute the SVD decomposition.

- Since SwinV2 adopts a learnable projection to generate bias, we need to infer the model once to obtain the relative position bias matrices and save them to the `./swin_v2_bias_save` folder. **Or download from [here](https://drive.google.com/drive/folders/1JQjyKf5QDDPmgh9KF_xl6ujwgrMVywf1?usp=sharing).**
- Decompose the bias matrix, where the factor tensors should be saved under `./decomposed_bias`. **Or download from [here](https://drive.google.com/drive/folders/1JQjyKf5QDDPmgh9KF_xl6ujwgrMVywf1?usp=sharing).**

2. Load factor tensors for fast inference.

- Move `swin_transformer_v2.py` and `flash_bias_triton.py` to the `./model` folder of the [official code](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py)  of Swin Transformer.
- Execute the following script for evaluation

```bash
export CUDA_VISIBLE_DEVICES=0
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12346 main.py --eval \
--cfg configs/swinv2/swinv2_base_patch4_window12to24_192to384_22kto1k_ft.yaml --resume swinv2_base_patch4_window12to24_192to384_22kto1k_ft.pth \
--data-path /data/imagenet/ \ # place to your own data path
--batch-size 64
```

## Efficiency Comparison

FlashBias can significantly speed up the inference without loss of accuracy.

| A100 Inference                                               | s / batch | Top 1 Acc | Top 5 Acc |
| ------------------------------------------------------------ | --------- | --------- | --------- |
| [Official Code](https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py) | 0.473     | 87.144%   | 98.232%   |
| FlashBias                                                    | **0.190** | 87.186%   | 98.220%   |

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{wu2025flashbias,
  title={FlashBias: Fast Computation of Attention with Bias},
  author={Haixu Wu and Minghao Guo and Yuezhou Ma and Yuanxu Sun and Jianmin Wang and Wojciech Matusik and Mingsheng Long},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Contact

If you have any questions or want to use the code, please contact wuhaixu98@gmail.com
