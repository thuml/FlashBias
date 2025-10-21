# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partialmethod, partial
from typing import Optional, List
import numpy as np
import torch
import torch.nn as nn
from protenix.metrics.rmsd import self_aligned_rmsd
from protenix.openfold_local.model.primitives import Linear, LayerNorm, Attention
from protenix.openfold_local.utils.chunk_utils import chunk_layer
from protenix.openfold_local.utils.tensor_utils import (
    permute_final_dims,
)


class TriangleAttention(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, starting=True, inf=1e9):
        """
        Args:
            c_in:
                Input channel dimension
            c_hidden:
                Overall hidden channel dimension (not per-head)
            no_heads:
                Number of attention heads
        """
        super(TriangleAttention, self).__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.starting = starting
        self.inf = inf

        self.layer_norm = LayerNorm(self.c_in)
        self.linear = Linear(c_in, self.no_heads, bias=False, init="normal")

        self.Rank = 96
        self.linear_q_flashbias = nn.Sequential(
            Linear(c_in + 449, 256, bias=False, init="normal"),
            nn.Tanh(),
            Linear(256, 256, bias=False, init="normal"),
            nn.Tanh(),
            Linear(256, self.no_heads * self.Rank, bias=False, init="normal")
        )
        self.linear_k_flashbias = nn.Sequential(
            Linear(c_in + 449, 256, bias=False, init="normal"),
            nn.Tanh(),
            Linear(256, 256, bias=False, init="normal"),
            nn.Tanh(),
            Linear(256, self.no_heads * self.Rank, bias=False, init="normal")
        )

        self.mha = Attention(
            self.c_in, self.c_in, self.c_in, self.c_hidden, self.no_heads
        )

    @torch.jit.ignore
    def _chunk(
            self,
            x: torch.Tensor,
            biases: List[torch.Tensor],
            chunk_size: int,
            q_bias: torch.Tensor = None,
            k_bias: torch.Tensor = None,
            use_memory_efficient_kernel: bool = False,
            use_deepspeed_evo_attention: bool = False,
            use_lma: bool = False,
            inplace_safe: bool = False,
    ) -> torch.Tensor:
        "triangle! triangle!"
        if q_bias is not None and k_bias is not None and biases is not None:
            mha_inputs = {
                "q_x": x,
                "kv_x": x,
                "biases": biases,
                'q_bias': q_bias,
                'k_bias': k_bias,
            }
        elif q_bias is not None and k_bias is not None:
            mha_inputs = {
                "q_x": x,
                "kv_x": x,
                'q_bias': q_bias,
                'k_bias': k_bias,
            }
        elif biases is not None:
            mha_inputs = {
                "q_x": x,
                "kv_x": x,
                "biases": biases,
            }
        else:
            mha_inputs = {
                "q_x": x,
                "kv_x": x,
            }

        return chunk_layer(
            partial(
                self.mha,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
            ),
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(x.shape[:-2]),
            _out=x if inplace_safe else None,
        )

    def forward(
            self,
            x: torch.Tensor,
            s_inputs: torch.Tensor = None,
            id: int = 10000,
            mask: Optional[torch.Tensor] = None,
            chunk_size: Optional[int] = None,
            use_memory_efficient_kernel: bool = False,
            use_deepspeed_evo_attention: bool = False,
            use_lma: bool = False,
            inplace_safe: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x:
                [*, I, J, C_in] input tensor (e.g. the pair representation)
        Returns:
            [*, I, J, C_in] output tensor
        """
        # [*, I, J]

        if not self.starting:
            x = x.transpose(-2, -3)

        # [*, I, J, C_in]
        x = self.layer_norm(x)
        N = x.shape[0]
        C = x.shape[-1]

        ###### for flashbias ######
        if self.training:
            # [*, H, I, J]
            # ###### for finetune
            # # x N N 128
            triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))
            # [*, 1, H, I, J]
            triangle_bias = triangle_bias.unsqueeze(-4)
            if s_inputs is not None:
                q_bias = self.linear_q_flashbias(torch.concat([x.mean(1), s_inputs], dim=-1)) \
                    .reshape(N, self.no_heads, self.Rank).permute(1, 0, 2).unsqueeze(0)  # N 128
                k_bias = self.linear_k_flashbias(torch.concat([x.mean(0), s_inputs], dim=-1)) \
                    .reshape(N, self.no_heads, self.Rank).permute(1, 0, 2).unsqueeze(0)  # N 128
                biases = None
                triangle_bias_low_rank = q_bias @ k_bias.transpose(-1, -2)
                finetune_loss = torch.mean((triangle_bias_low_rank - triangle_bias) ** 2)
            else:
                # x N N 128
                triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))
                # [*, 1, H, I, J]
                triangle_bias = triangle_bias.unsqueeze(-4)
                finetune_loss = 0
                biases = [triangle_bias]
                q_bias = None
                k_bias = None
        else:
            if s_inputs is not None and id < 16: # adopt flashbias for the first 16 layers
                ###### for flashbias attention
                q_bias = self.linear_q_flashbias(torch.concat([x.mean(1), s_inputs], dim=-1)) \
                    .reshape(N, self.no_heads, self.Rank).permute(1, 0, 2).unsqueeze(0)  # N 128
                k_bias = self.linear_k_flashbias(torch.concat([x.mean(0), s_inputs], dim=-1)) \
                    .reshape(N, self.no_heads, self.Rank).permute(1, 0, 2).unsqueeze(0)  # N 128
                finetune_loss = 0
                biases = None
            else:
                # [*, H, I, J]
                ###### for original attention
                triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1))
                # [*, 1, H, I, J]
                triangle_bias = triangle_bias.unsqueeze(-4)
                finetune_loss = 0
                q_bias = None
                k_bias = None
                biases = [triangle_bias]

        if chunk_size is not None:
            x = self._chunk(
                x,
                biases,
                chunk_size,
                q_bias=q_bias,
                k_bias=k_bias,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
            )
        else:
            x = self.mha(
                q_x=x,
                kv_x=x,
                q_bias=q_bias,
                k_bias=k_bias,
                biases=biases,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
            )

        if not self.starting:
            x = x.transpose(-2, -3)

        return x, finetune_loss


# Implements Algorithm 13
TriangleAttentionStartingNode = TriangleAttention


class TriangleAttentionEndingNode(TriangleAttention):
    """
    Implements Algorithm 14.
    """

    __init__ = partialmethod(TriangleAttention.__init__, starting=False)
