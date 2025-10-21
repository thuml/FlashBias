# Triton Implementation of FlashBias
This is a Triton implementation of FlashBias, which supports both forward and backward attention computation with decomposed bias terms in the following formalization:

$$
\mathbf{o}=\text{softmax}(\frac{\mathbf{q}\mathbf{k}^T}{\sqrt{C}}+\mathbf{b})\mathbf{v}=\text{softmax}(\frac{\mathbf{q}\mathbf{k}^T}{\sqrt{C}}+\mathbf{q}_{\text{bias}}\mathbf{k}_{\text{bias}}^T)\mathbf{v}
$$

## Benchmark

1. Install `python=3.9, torch>=2.0.0, triton>=2.0.0, xformers`.

**Important notice !!! ** (1) Since the backward pass of Triton kernel may cause a race condition due to the Triton compiler, we suggest you **configure `triton=3.0.0`**, which has been tested as correct. (2) Usually, reduce the `num_warps` can avoid the race condition.

2. Execute the following command for benchmarking.

```bash
python benchmark_alibi_gpt2.py
```

3. Check the efficiency benchmark among

- Triton implementation of FlashBias (**Ours**)
- PyTorch-SDPA based implementation of FlashBias (**Ours**)
- Triton implementation of FlashAttention with Bias ([Opensource Implementation](https://github.com/pengzhangzhi/Flash-Attention-with-Bias-Triton))
- Pytorch implementation of vanilla attention with Bias
- Pytorch implementation of vanilla attention with Bias with Compile
- xFormers Implementation of FlashAttention with Bias ([xFormers](https://github.com/facebookresearch/xformers))
- Pytorch Implementation of FlashAttention with Bias ([SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html))

## Efficiency Comparison

We suggest to use [Triton-based FlashBias](https://github.com/thuml/FlashBias/blob/main/flash_bias/flash_bias_triton.py) in inference and [PyTorch-SDPA-based FlashBias](https://github.com/thuml/FlashBias/blob/main/flash_bias/attention_func.py) for training.

<p align="center">
<img src=".\attention-comparison-batch2-head4-d32-r8-causalFalse-fwd.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 1.</b> Efficiency comparison for forward pass.
</p>

<p align="center">
<img src=".\attention-comparison-batch2-head4-d32-r8-causalFalse-bwd.png" height = "250" alt="" align=center />
<br><br>
<b>Figure 2.</b> Efficiency comparison for backward propagation.
</p>


## Quick Start of FlashBias

```python
## Triton-based FlashBias, See `flash_bias_triton.py` for more details.
from flash_bias_triton import flash_bias_func
output_flash_triton = flash_bias_func(q, k, v, q_bias, k_bias, None, False, 1 / np.sqrt(headdim))

## PyTorch-SDPA-based FlashBias
import torch
# Notably, the dimension of concat[q, q_bias] should be divided evenly by 8; otherwise, you cannot activate flashattention in the backend
output_flash_sdpa = torch.nn.functional.scaled_dot_product_attention(
    query=torch.concat([q * softmax_scale, q_bias], dim=-1),
    key=torch.concat([k, k_bias], dim=-1),
    value=v,
    attn_mask=None,
    dropout_p=0.0,
    scale=1,
    is_causal=causal,
)
```

## Triton Kernel API Configuration

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
