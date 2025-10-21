"""
Configs for triton

Refactor from the original project from [https://github.com/pengzhangzhi]

@author: Zhangzhi Peng, Xuan Chen
"""

import triton

# Benchmarking configuration
BATCH, N_HEADS, HEAD_DIM, RANK_DIM, CAUSAL = 2, 4, 32, 8, False  # Aligned with test_attention_correctness

benchmark_configs = [
    triton.testing.Benchmark(
        x_names=["len"],
        x_vals=list(2 ** i for i in range(8, 15)),
        line_arg="provider",
        line_vals=[
            "triton-flash-bias",
            "sdpa-flash-bias",
            "triton-attn-bias",
            "torch-attn-bias",
            "torch-compile-attn-bias",
            "xformers-attn-bias",
            "torch-sdpa"
        ],
        line_names=[
            "Triton-flash-bias",
            "PyTorch-SDPA-flash-bias",
            "Triton-Attn-Bias",
            "PyTorch-Attn-Bias",
            "PyTorch-Compile-Attn-Bias",
            "xFormers-Attn-Bias",
            "PyTorch-SDPA"
        ],
        styles=[
            ("red", "-"),
            ("red", "--"),
            ("pink", "-"),
            ("blue", "--"),
            ("cyan", "-."),
            ("orange", "-."),
            ("green", ":")
        ],
        ylabel="Speed (TFLOPS/s)",
        xlabel="Sequence Length",
        plot_name=f"attention-comparison-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-r{RANK_DIM}-causal{CAUSAL}-{mode}",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "HEAD_DIM": HEAD_DIM,
            "RANK_DIM": RANK_DIM,
            "CAUSAL": CAUSAL,
            "mode": mode,
        },
    )
    for mode in ["fwd", "bwd"]
]
