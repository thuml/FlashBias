# FlashBias for Language Model

Large language models rely on **relative position encoding**. With FlashBias, we can easily speed up the computation of attention with [ALiBi](https://github.com/jaketae/alibi) bias.

<u>**Implementation**:</u> ALiBi bias can be exactly decomposed as two basis functions:

<p align="center">
<img src="..\pic\low_rank_alibi.png" height = "90" alt="" align=center />
<br><br>
<b>Figure 1.</b> Exact decomposition of ALiBi bias.
</p>

## Get Started

We consider the [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) style model with 50 heads and 1,600 hidden channels in attention.

1. Implementation

It is quite easy to implement ALiBi with FlashBias. Please see `./attention_with_alibi_bias.py` for details.

2. Efficiency comparison

For [../flash_bias/config.py](https://github.com/thuml/FlashBias/blob/main/flash_bias/config.py), we change the configuration of the attention benchmark as follows:

```python
BATCH, N_HEADS, HEAD_DIM, RANK_DIM, CAUSAL = 1, 50, 32, 2, True
```

Execute the following command for efficiency comparison:

```bash
cd ../flash_bias
python benchmark.py
```

## Efficiency Comparison

Note that vanilla flash_attn_triton is unstable during the backward pass phase when the sequence length is 8192. Our FlashBias and all the other kernels are correct across all input lengths.

<p align="center">
<img src=".\attention-comparison-batch1-head50-d32-r2-causalTrue-fwd.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Efficiency comparison for forward pass on GPT-2 size attention with ALiBi Bias.
</p>

<p align="center">
<img src=".\attention-comparison-batch1-head50-d32-r2-causalTrue-bwd.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Efficiency comparison for backward propagation on GPT-2 size attention with ALiBi Bias.
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
