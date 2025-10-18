import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from flash_bias_triton import flash_bias_func


def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(0)
        .unsqueeze(0)
        .unsqueeze(-1)
    )


class Attention_with_ALiBi_Bias(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.scale = dim_head ** -0.5
        # Separate projection layers for query, key, and value
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.register_buffer("slope", get_alibi_slope(self.heads))  # 1, 1, head, 1

    def forward(self, x):
        # x shape: [batch_size, seq_len, dim]
        batch_size, seq_len, _ = x.shape

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = rearrange(q, 'b n (h d) -> b n h d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b n h d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b n h d', h=self.heads)

        pos_index = torch.arange(seq_len).float().to(x.device)[None, :, None, None].repeat(1, 1, self.heads, 1)

        q_bias = torch.concat([torch.ones(1, seq_len, self.heads, 1).to(x.device), pos_index], dim=-1) * self.slope
        k_bias = torch.concat([-pos_index, torch.ones(1, seq_len, self.heads, 1).to(x.device)], dim=-1)

        # causal attention with alibi bias
        attn_output = flash_bias_func(
            q.half(), k.half(), v.half(),
            q_bias.half(),
            k_bias.half(),
            None,
            True,
            self.scale
        )
        out = rearrange(attn_output.float(), 'b n h d -> b n (h d)')
        return self.to_out(out)


if __name__ == "__main__":
    x = torch.randn(1, 1024, 1600).cuda()
    attn = Attention_with_ALiBi_Bias(dim=1600, heads=50, dim_head=32, dropout=0.1).cuda()
    output = attn(x)
    print(output.shape)
