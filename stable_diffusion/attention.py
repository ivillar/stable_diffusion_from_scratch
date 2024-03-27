import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(
        self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True
    ):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: batch_num, seq_len, dim
        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape
        interim_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # batch_num, seq_len, dim -> batch_num, seq_length, 3*dim ->
        # 3 tensors of size batch_num, seq_len, dim
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # batch_num, seq_len, dim -> batch_num, seq_len, h, dim/h
        # -> batch_num, h, seq_len, dim/h
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # batch_num, h, seq_len, seq_len
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # batch_num, h, seq_len, seq_len @ batch_num, h, seq_len, dim/h
        # -> batch_num, h, seq_len, dim/h
        output = weight @ v

        # batch_num, h, seq_len, dim/h -> batch_num, seq_len, h, dim/h
        output = output.transpose(1, 2)

        # batch_size, seq_len, dim
        output = output.reshape(input_shape)

        output = self.out_proj(output)

        return output
