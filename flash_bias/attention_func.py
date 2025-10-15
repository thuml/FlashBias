"""
Attention functions for flashbias triton, torch torch-compile, xformers, and sdpa

Refactor from the original project from [https://github.com/pengzhangzhi/Flash-Attention-with-Bias-Triton]
"""
import torch
import xformers.ops as xops

from flash_attn_triton import FlashAttnFunc
from flash_bias_triton import FlashBiasFunc

attention_triton = FlashAttnFunc.apply
flashbias_triton = FlashBiasFunc.apply


def attention_torch(q, k, v, softmax_scale, bias=None):
    """
    Implements standard scaled dot-product attention using PyTorch.

    Args:
        q: Query tensor of shape (batch_size, seqlen_q, nheads, headdim)
        k: Key tensor of shape (batch_size, seqlen_k, nheads, headdim)
        v: Value tensor of shape (batch_size, seqlen_k, nheads, headdim)
        softmax_scale: Scalar for softmax normalization (default: 1/sqrt(headdim))
        bias: Optional bias tensor, shape (batch, nheads, seqlen_q, seqlen_k)

    Returns:
        Output tensor of shape (batch_size, seqlen_q, nheads, headdim)
    """
    # Compute scaled dot-product attention scores
    attn_scores = torch.einsum("bqnh,bknh->bqnk", q, k)  # (batch, seqlen_q, nheads, seqlen_k)
    # Apply softmax scaling
    attn_scores *= softmax_scale

    # Apply bias if provided
    if bias is not None:
        attn_scores += bias.permute(0, 2, 1, 3)

    # Apply softmax along key dimension
    attn_probs = torch.softmax(attn_scores, dim=-1)

    # Compute output by weighted sum of values
    output = torch.einsum("bqnk,bknh->bqnh", attn_probs, v)

    return output


@torch.compile
def attention_torch_compile(q, k, v, softmax_scale, bias=None):
    """
    Implements torch compile scaled dot-product attention using PyTorch.

    Author:
        Xuan Chen (Apr 15, 2025), referenced from: attention_torch()

    Args:
        q: Query tensor of shape (batch_size, seqlen_q, nheads, headdim)
        k: Key tensor of shape (batch_size, seqlen_k, nheads, headdim)
        v: Value tensor of shape (batch_size, seqlen_k, nheads, headdim)
        softmax_scale: Scalar for softmax normalization (default: 1/sqrt(headdim))
        bias: Optional bias tensor, shape (batch, nheads, seqlen_q, seqlen_k)

    Returns:
        Output tensor of shape (batch_size, seqlen_q, nheads, headdim)
    """
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 3, 1)
    v = v.permute(0, 2, 1, 3)

    # Compute scaled dot-product attention scores
    attn_scores = torch.matmul(q, k)
    # Apply softmax scaling
    attn_scores *= softmax_scale

    # Apply bias if provided
    if bias is not None:
        attn_scores += bias

    # Apply softmax along key dimension
    attn_probs = torch.softmax(attn_scores, dim=-1)

    # Compute output by weighted sum of values
    output = torch.matmul(attn_probs, v)
    output = output.permute(0, 2, 1, 3)

    return output


def attention_xformers(q, k, v, softmax_scale, bias=None):
    """
    Implements torch compile scaled dot-product attention using xFormers.

    Args:
        q: Query tensor of shape (batch_size, seqlen_q, nheads, headdim)
        k: Key tensor of shape (batch_size, seqlen_k, nheads, headdim)
        v: Value tensor of shape (batch_size, seqlen_k, nheads, headdim)
        softmax_scale: Scalar for softmax normalization (default: 1/sqrt(headdim))
        bias: Optional bias tensor, shape (batch, nheads, seqlen_q, seqlen_k)

    Returns:
        Output tensor of shape (batch_size, seqlen_q, nheads, headdim)
    """
    return xops.memory_efficient_attention(
        query=q,
        key=k,
        value=v,
        attn_bias=bias,
        scale=softmax_scale
    )


def attention_sdpa(q, k, v, softmax_scale, bias=None):
    """
    Implements scaled dot-product attention using PyTorch's F.scaled_dot_product_attention.

    Args:
        q: Query tensor of shape (batch_size, seqlen_q, nheads, headdim)
        k: Key tensor of shape (batch_size, seqlen_k, nheads, headdim)
        v: Value tensor of shape (batch_size, seqlen_k, nheads, headdim)
        softmax_scale: Scalar for softmax normalization (default: 1/sqrt(headdim))
        bias: Optional bias tensor, shape (batch, nheads, seqlen_q, seqlen_k)

    Returns:
        Output tensor of shape (batch_size, seqlen_q, nheads, headdim)
    """
    # SDPA expects input of shape (B, Nh, L, D)
    # Our inputs are already in this format after transposing
    q = q.transpose(1, 2)  # (batch, nheads, seqlen_q, headdim)
    k = k.transpose(1, 2)  # (batch, nheads, seqlen_k, headdim)
    v = v.transpose(1, 2)  # (batch, nheads, seqlen_k, headdim)

    # Prepare attention mask from bias
    attn_mask = bias if bias is not None else None

    # Call SDPA with our inputs
    output = torch.nn.functional.scaled_dot_product_attention(
        query=q,
        key=k,
        value=v,
        attn_mask=attn_mask,
        dropout_p=0.0,
        scale=softmax_scale,
        is_causal=False
    )

    # Return to original format (B, Lq, Nh, D)
    return output.transpose(1, 2)
