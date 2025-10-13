import torch
import torch.nn.functional as F
import numpy as np
from flash_bias_triton import flash_bias_func


def test_flash_attention_bias_forward():
    print("\n\nTesting FlashAttention with Bias Forward")

    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = 16
    seqlen_q = 1024
    seqlen_k = 1024
    nheads = 4
    headdim = 64
    rank = 16

    q = torch.randn(batch_size, seqlen_q, nheads, headdim, requires_grad=True, device='cuda').half()
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, requires_grad=True, device='cuda').half()
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, requires_grad=True, device='cuda').half()

    q_bias = torch.randn(batch_size, seqlen_q, nheads, rank, requires_grad=True, device='cuda').half()
    k_bias = torch.randn(batch_size, seqlen_k, nheads, rank, requires_grad=True, device='cuda').half()

    print(f"Input shapes:")
    print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
    print(f"q_bias: {q_bias.shape}, k_bias: {k_bias.shape}")

    print("\n1. Testing forward pass...")

    output_flash = flash_bias_func(q, k, v, q_bias, k_bias, None, False, 1 / np.sqrt(headdim))

    with torch.no_grad():
        q_reshaped = q.transpose(1, 2)  # [batch, nheads, seqlen_q, headdim]
        k_reshaped = k.transpose(1, 2)  # [batch, nheads, seqlen_k, headdim]
        v_reshaped = v.transpose(1, 2)  # [batch, nheads, seqlen_k, headdim]

        q_bias_reshaped = q_bias.transpose(1, 2)  # [1, nheads, seqlen_q, rank]
        k_bias_reshaped = k_bias.transpose(1, 2)  # [1, nheads, seqlen_k, rank]

        attn_scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / np.sqrt(headdim)
        bias_scores = torch.matmul(q_bias_reshaped, k_bias_reshaped.transpose(-2, -1))
        attn_scores = attn_scores + bias_scores

        attn_weights = F.softmax(attn_scores, dim=-1)

        output_standard = torch.matmul(attn_weights, v_reshaped).transpose(1, 2)

    forward_diff = torch.abs(output_flash - output_standard).max().item()
    print(f"Forward Pass Maximum Difference: {forward_diff:.6f}")

    if forward_diff < 0.05:
        print("✓ Forward pass test PASSED")
    else:
        print("✗ Forward pass test FAILED")
        return False


def test_flash_attention_bias_backward():
    print("\n\nTesting FlashAttention with Bias Backward")
    torch.manual_seed(42)
    np.random.seed(42)

    batch_size = 16
    seqlen_q = 1024
    seqlen_k = 1024
    nheads = 4
    headdim = 64
    rank = 16

    q = torch.randn(batch_size, seqlen_q, nheads, headdim, requires_grad=True, device='cuda').half()
    k = torch.randn(batch_size, seqlen_k, nheads, headdim, requires_grad=True, device='cuda').half()
    v = torch.randn(batch_size, seqlen_k, nheads, headdim, requires_grad=True, device='cuda').half()

    q_bias = torch.randn(batch_size, seqlen_q, nheads, rank, requires_grad=True, device='cuda').half()
    k_bias = torch.randn(batch_size, seqlen_k, nheads, rank, requires_grad=True, device='cuda').half()

    print(f"Input shapes:")
    print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}")
    print(f"q_bias: {q_bias.shape}, k_bias: {k_bias.shape}")

    print("\n2. Testing backward pass...")

    output_flash = flash_bias_func(q, k, v, q_bias, k_bias, None, False, 1 / np.sqrt(headdim))
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    q_bias.retain_grad()
    k_bias.retain_grad()

    target = torch.randn_like(output_flash)
    loss_flash = 0.5 * torch.sum((output_flash - target) ** 2)
    loss_flash.backward()

    dq_flash = q.grad.clone()
    dk_flash = k.grad.clone()
    dv_flash = v.grad.clone()
    dq_bias_flash = q_bias.grad.clone()
    dk_bias_flash = k_bias.grad.clone()

    q.grad = None
    k.grad = None
    v.grad = None
    q_bias.grad = None
    k_bias.grad = None

    q_reshaped = (q.transpose(1, 2))  # [batch, nheads, seqlen_q, headdim]
    k_reshaped = (k.transpose(1, 2))  # [batch, nheads, seqlen_k, headdim]
    v_reshaped = (v.transpose(1, 2))  # [batch, nheads, seqlen_k, headdim]
    q_bias_reshaped = q_bias.transpose(1, 2)  # [1, nheads, seqlen_q, rank]
    k_bias_reshaped = k_bias.transpose(1, 2)  # [1, nheads, seqlen_k, rank]
    q_reshaped.retain_grad()
    k_reshaped.retain_grad()
    v_reshaped.retain_grad()
    q_bias_reshaped.retain_grad()
    k_bias_reshaped.retain_grad()

    attn_scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / np.sqrt(headdim)
    bias_scores = torch.matmul(q_bias_reshaped, k_bias_reshaped.transpose(-2, -1))
    attn_scores = attn_scores + bias_scores

    attn_weights = F.softmax(attn_scores, dim=-1)

    output_standard = torch.matmul(attn_weights, v_reshaped).transpose(1, 2)

    loss_standard = 0.5 * torch.sum((output_standard - target) ** 2)
    loss_standard.backward()

    dq_standard = q_reshaped.grad.transpose(1, 2)
    dk_standard = k_reshaped.grad.transpose(1, 2)
    dv_standard = v_reshaped.grad.transpose(1, 2)
    dq_bias_standard = q_bias_reshaped.grad.transpose(1, 2)
    dk_bias_standard = k_bias_reshaped.grad.transpose(1, 2)

    dq_diff = torch.abs(dq_flash - dq_standard).max().item()
    dk_diff = torch.abs(dk_flash - dk_standard).max().item()
    dv_diff = torch.abs(dv_flash - dv_standard).max().item()
    dq_bias_diff = torch.abs(dq_bias_flash - dq_bias_standard).max().item()
    dk_bias_diff = torch.abs(dk_bias_flash - dk_bias_standard).max().item()

    print(f"Maximum Gradient differences:")
    print(f"  dQ: {dq_diff:.6f}")
    print(f"  dK: {dk_diff:.6f}")
    print(f"  dV: {dv_diff:.6f}")
    print(f"  dQ_bias: {dq_bias_diff:.6f}")
    print(f"  dK_bias: {dk_bias_diff:.6f}")

    max_grad_diff = max(dq_diff, dk_diff, dv_diff, dq_bias_diff, dk_bias_diff)
    if max_grad_diff < 1.0:
        print("✓ Backward pass test PASSED")
    else:
        print("✗ Backward pass test FAILED")
        return False


if __name__ == "__main__":
    if torch.cuda.is_available():
        success = test_flash_attention_bias_forward()
        success = test_flash_attention_bias_backward()
    else:
        print("CUDA is not available. Please run on a GPU-enabled machine.")
