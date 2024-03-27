import torch
from torch import nn
from torch.nn import functional as F
from .attention import SelfAttention


class CLIPEmbedding(nn.Module):

    def __init__(self, n_vocab: int, n_embed: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embed))

    def forward(self, tokens):
        # tokens: batch_num, seq_len -> batch_num, seq_len, dim
        x = self.token_embedding(tokens)
        x += self.position_embedding[: tokens.size(1), :]
        return x


class CLIPLayer(nn.Module):

    def __init__(self, n_head: int, n_embed: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embed)
        self.attention = SelfAttention(n_head, n_embed)
        self.layernorm_2 = nn.LayerNorm(n_embed)
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # batch_num_ seq_len, dim

        residue = x

        # Self-attention

        x = self.layernorm_1(x)

        x = self.attention(x, causal_mask=True)

        x += residue

        # Feedforward layer

        residue = x

        x - self.layernorm_2(x)

        # batch_num_ seq_len, 4 * dim

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x)  # QuickGELU activation function

        # batch_num_ seq_len, dim

        x = self.linear_2(x)

        x += residue

        return x


class CLIP(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = CLIPEmbedding(49408, 768, 77)

        self.layers = nn.ModuleList([CLIPLayer(12, 768) for i in range(12)])

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        # batch_num, seq_len
        tokens = tokens.type(torch.long)

        # batch_num, seq_len, dim
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # batch_num, seq_len, dim
        output = self.layernorm(state)

        return output
