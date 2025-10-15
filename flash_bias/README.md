# Triton Implementation of FlashBias
This is a Triton implementation of FlashBias, which supports both forward and backward attention computation with decomposed bias terms in the following formalization:
$$
o=\text{softmax}(\frac{\mathbf{q}\mathbf{k}^T}{\sqrt{C}}+\mathbf{b})\mathbf{v}=\text{softmax}(\frac{\mathbf{q}\mathbf{k}^T}{\sqrt{C}}+\mathbf{q}_{\text{bias}}\mathbf{k}_{\text{bias}}^\top)\mathbf{v}
$$

## Benchmark

1. Install `python=3.9, torch>=2.0.0, triton>=2.0.0, xformers`.
2. Excute the following command for benchmarking.

```python
python benchmark.py
```

3. Check efficiency benchmark among

- FlashBias (**Ours**)
- Triton implementation of FlashAttention with Bias ([Opensource Implementation](https://github.com/pengzhangzhi/Flash-Attention-with-Bias-Triton))
- Pytorch implementation of vanilla attention with Bias
- Pytorch implementation of vanilla attention with Bias with Compile
- xFormers Implementation of FlashAttention with Bias ([xFormers](https://github.com/facebookresearch/xformers))
- Pytorch Implementation of FlashAttention with Bias ([SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html))

## Efficiency Comparison

<p><div style="display: flex; justify-content: space-between; gap: 20px;">
  <div style="width: 48%; text-align: center;">
    <img src="attention-comparison-batch2-head4-d32-r8-fwd.png" alt="Forward" style="width: 100%;">
    <p>(a) Forward Efficiency.</p>
  </div>
  <div style="width: 48%; text-align: center;">
    <img src="attention-comparison-batch2-head4-d32-r8-bwd.png" alt="Backward" style="width: 100%;">
    <p>(b) Backward Efficiency.</p>
  </div>
</div></p>
## Quick Start of FlashBias

See `flash_bias_triton.py` for more details.

```python
from flash_bias_triton import flash_bias_func
output_flash = flash_bias_func(q, k, v, q_bias, k_bias, None, False, 1 / np.sqrt(headdim))
```

## API Configuration

```python
flash_bias_func(q, k, v, q_bias=None, k_bias=None, mask=None, causal=False, softmax_scale=None)
"""Arguments:
q: (batch_size, seqlen_q, nheads, headdim)
k, v: (batch_size, seqlen_k, nheads, headdim)
q_bias: (batch_size, seqlen_q, nheads, rankdim) or (1, seqlen_q, nheads, rankdim) or (1, seqlen_q, 1, rankdim) or (batch_size, seqlen_q, 1, rankdim)
k_bias: (batch_size, seqlen_k, nheads, rankdim) or (1, seqlen_k, nheads, rankdim) or (1, seqlen_q, 1, rankdim) or (batch_size, seqlen_q, 1, rankdim)
mask: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
softmax_scale: should be set as 1 / sqrt(headdim), otherwise directly multiply to the q vector. If without any input, it will be set as 1 / sqrt(headdim)
"""
```

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
