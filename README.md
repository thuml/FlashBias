# FlashBias (NeurIPS 2025)
FlashBias: Fast Computation of Attention with Bias [[Paper]](https://arxiv.org/pdf/2505.12044)

Attention with bias widely exists, such as relative position encoding in [Swin Transformer](https://github.com/microsoft/Swin-Transformer), scientific prior encoding in [AlphaFold (Nature 2024)](https://www.nature.com/articles/s41586-024-07487-w) and [Pangu-Weather (Nature 2023)](https://www.nature.com/articles/s41586-023-06185-3). Surprisingly, despite the common use of attention with bias, its targeted efficiency optimization remains absent.

This paper presents [FlashBias](https://arxiv.org/pdf/2505.12044) based on the low-rank compressed sensing theory, which can provide fast-exact computation for many widely used attention biases, enabling 1.5× speedup for AlphaFold, and over 2× speedup for attention with bias in vision and language models without loss of accuracy.

<p align="center">
<img src=".\pic\flashbias.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of FlashBias.
</p>

## Get Started

**Check [./flash_bias](https://github.com/thuml/FlashBias/tree/main/flash_bias) for Triton kernel.**

The following are some representative applications of FlashBias.

1. **Large language model** with ALiBi bias: See [./0_Language_Model](https://github.com/thuml/FlashBias/tree/main/0_Language_Model)
2. **Swin Transformer V2** with relative position bias: See [./1_Vision_Transformer](https://github.com/thuml/FlashBias/tree/main/1_Vision_Transformer)
3. **Transformer PDE Solver** with spatial distance bias: See [./2_Neual_Solver](https://github.com/thuml/FlashBias/tree/main/2_Neural_Solver)
4. **AlphaFold 3** with pair representation bias: Coming soon (this week).

## Overall Comparison

A significant memory and running time reduction compared to vanilla FlashAttention.

<p align="center">
<img src=".\pic\efficiency.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Efficiency comparison with vanilla FlashAttention.
</p>
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

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code base or datasets:

https://github.com/Dao-AILab/flash-attention

https://github.com/meta-pytorch/attention-gym

https://github.com/bytedance/Protenix

https://github.com/microsoft/Swin-Transformer
