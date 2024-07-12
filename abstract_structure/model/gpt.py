import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from copy import deepcopy

import numpy as np
import json
import glob
import os
import sentencepiece as spm

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.notebook import trange

import math

from typing import Dict, Tuple

import torch.optim as optim
from torch.optim import AdamW as Adam
from torch.nn import LayerNorm

from abstract_structure.config.train_config import Config

# ================ Attention ================
class BaseAttention(nn.Module):
    """
       Tensor          Type            Shape
       ===========================================================================
       q               float           (bs, query_len, dims)   or (..., heads, query_len, dims)
       k               float           (bs, kv_len, dims)      or (..., heads, kv_len, dims)
       v               float           (bs, kv_len, dims)      or (..., heads, kv_len, dims)
       mask            bool            (bs, query_len, kv_len) or (..., heads, query_len, dims)
       ---------------------------------------------------------------------------
       output          float           (bs, query_len, dims)   or (..., heads, query_len, dims)
       ===========================================================================
       """
    def __init__(self, dropout:float=0.1):
        super().__init__()
        self.dropout=nn.Dropout(dropout)

    def forward(self,
                q:torch.Tensor,
                k:torch.Tensor,
                v:torch.Tensor,
                mask=None) -> torch.Tensor:
        x = torch.matmul(q,k.transpose(-2,-1)) / math.sqrt(k.size(-1))

        if mask is not None:
            x += mask.type_as(x) * x.new_tensor(-1e4)

        x = self.dropout(x.softmax(dim=-1))

        return torch.matmul(x, v)

class MultiHeadAttention(nn.Module):
    """
        Tensor          Type            Shape
        ===========================================================================
        q               float           (..., query_len, dims)
        k               float           (..., kv_len, dims)
        v               float           (..., kv_len, dims)
        mask            bool            (..., query_len, kv_len)
        ---------------------------------------------------------------------------
        output          float           (..., query_len, dims)
        ===========================================================================
    """
    def __init__(self, heads:int, dropout:float=0.1):
        super().__init__()
        self.dropout=dropout
        self.heads = heads

    def forward(self,
                q:torch.Tensor,
                k:torch.Tensor,
                v:torch.Tensor,
                mask=None) -> torch.Tensor:
        # Split the tensor to multi-heads
        q_dim = q.size(-1)
        k_dim = k.size(-1)
        v_dim = v.size(-1)

        q = q.view(q.size()[:-1] + (self.heads, q_dim // self.heads))
        k = k.view(k.size()[:-1] + (self.heads, k_dim // self.heads))
        v = v.view(v.size()[:-1] + (self.heads,v_dim // self.heads))

        q = q.transpose(-3,-2)
        k = k.transpose(-3,-2)
        v = v.transpose(-3,-2)

        if mask is not None:
            mask=mask.unsqueeze(-3)

        # Calculate multi-headed attentions and merge them into one
        #   shape of q:
        #   shape of k:
        #   shape of v:
        out_MSA = super().forward(q, k, v, mask)
        # Shape of out_MSA:

        out_MSA = out_MSA.view(q.size()[:-3] + (q.size(-2), v.size(-1)  * self.heads))
        # Shape of out_MSA:
        return out_MSA

class AttentionLayer(nn.Module):
    """
        Tensor          Type            Shape
        ===========================================================================
        q               float           (..., query_len, dims)
        k               float           (..., kv_len, dims)
        v               float           (..., kv_len, dims)
        past (*)        float           (..., past_len, dims)
        mask            bool            (..., query_len, past_len + kv_len)
        ---------------------------------------------------------------------------
        output 1        float           (..., query_len, dims)
        output 2 (*)    float           (..., past_len + kv_len, dims)
        ===========================================================================
    """
    def __init__(self, heads:int, dims:int, dropout:float=0.1):
        super().__init__()
        self.attn=MultiHeadAttention(heads, dropout)
        self.proj_q = nn.Linear(dims, dims)
        self.proj_k = nn.Linear(dims, dims)
        self.proj_v = nn.Linear(dims, dims)
        self.linear = nn.Linear(dims, dims)

    def forward(self,
                q:torch.Tensor,
                k:torch.Tensor,
                v:torch.Tensor,
                past=None,
                mask=None):
        q, k, v =self.proj_q(q), self.proj_k(k),self.proj_v(v)

        # Reuse attention keys and values by concatenating to the current ones.
        if past is not None:
            k = torch.cat((past[0],k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)

        x = self.linear(self.attn(q, k, v, mask))
        return x, (k,v) # (k,v) means --->
# ============= FFNN =============
class PositionwiseFeedForward(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           float           (..., dims)
    ---------------------------------------------------------------------------
    output          float           (..., dims)
    ===========================================================================
    """
    def __init__(self, dims: int, rate: int = 4, dropout: float = 0.1):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(dims, dims * rate),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dims * rate, dims)
        )

    def forward(self, x):
        return self.linear(x)
class Swish(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           float           (..., dims)
    ---------------------------------------------------------------------------
    output          float           (..., dims)
    ===========================================================================
    """
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)

# =============== Masking ================
class PadMasking(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len), [this is a token]
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, seq_len + offset)
    ===========================================================================
    """
    def __init__(self, pad_idx: int):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        is_pad = (x == self.pad_idx).unsqueeze(-2)
        #
        shifted = torch.zeros(x.size()[:-1] + (1, offset,),
                              dtype=torch.bool, device=x.device)

        mask = torch.cat((shifted, is_pad), dim=-1)
        return mask.expand(x.shape + mask.shape[-1:])

class FutureMasking(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, seq_len + offset)
    ===========================================================================
    """
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        seq_len = x.size(-1)

        # Create shifted upper triangular matrix.
        future = torch.ones((seq_len, seq_len + offset),
                            dtype=torch.bool, device=x.device)
        future = future.triu(offset + 1)

        mask = future.view((1,) * (x.ndim - 1) + future.size())
        return mask.expand(x.shape + mask.shape[-1:])
# ================ Embedding ====================
class PositionalEmbedding(nn.Embedding):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long            (..., seq_len)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, embedding_dim)
    ===========================================================================
    """
    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)

    def _load_from_state_dict(self,
                              state_dict: Dict[str, torch.Tensor],
                              prefix: str,
                              *args,
                              **kwargs):
        weight = state_dict[f'{prefix}weight']

        # Reduce or expand the positional embedding matrix to increase or
        # decrease the total sequence length.
        if weight.size(0) < self.num_embeddings:
            weight = torch.cat((weight, self.weight[weight.size(0):]), dim=0)
        elif weight.size(0) > self.num_embeddings:
            weight = weight[:self.num_embeddings]

        state_dict[f'{prefix}weight'] = weight
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        position = torch.arange(offset, offset + x.size(-1), dtype=torch.long, device=x.device)
        position = position.view((1,) * (x.ndim - 1) + (-1,)).expand_as(x)

        return super().forward(position)

class TokenEmbedding(nn.Embedding):
    """
    Tensor          Type            Shape
    ===========================================================================
    input           long or float  (..., seq_len)
                                    or (..., seq_len, embedding_dim)
    ---------------------------------------------------------------------------
    output          float           (..., seq_len, embedding_dim)
                                    or (..., seq_len, num_embeddings)
    ===========================================================================
    """
    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)

    def forward(self,
                x: torch.Tensor,
                transposed: bool = False) -> torch.Tensor:
        if transposed:
            return torch.matmul(x, self.weight.transpose(0, 1))
        else:
            return super().forward(x)

# ====================== GPT ======================
class TransformerLayer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               float           (..., seq_len, dims)
    past (*)        float           (..., past_len, dims)
    mask            bool            (..., seq_len, past_len + seq_len)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dims)
    output 2 (*)    float           (..., past_len + seq_len, dims)
    ===========================================================================
    """
    def __init__(self,
                 heads: int,
                 dims: int,
                 rate: int,
                 training: bool = True,
                 dropout: float = 0.1):
        super().__init__()

        self.attn = AttentionLayer(heads, dims, dropout)
        self.ff = PositionwiseFeedForward(dims, rate, dropout)
        #
        self.ln_attn = LayerNorm(dims)
        self.ln_ff = LayerNorm(dims)
        #
        self.training = training

    def forward(self,
                x: torch.Tensor,
                past = None,
                mask = None,
                ):
        # Layer normalizations are performed before the layers respectively.
        a = self.ln_attn(x)
        a, past = self.attn(a, a, a, past, mask)

        x = x + a
        x = x + self.ff(self.ln_ff(x))

        return x if self.training else (x, past)

class Transformer(nn.Module):
    """
    Tensor          Type            Shape
    ===========================================================================
    x               long            (..., seq_len)
    past (**)       float           (..., past_len, dims)
    ---------------------------------------------------------------------------
    output 1        float           (..., seq_len, dims)
    output 2 (**)   float           (..., past_len + seq_len, dims)
    ===========================================================================
    """
    def __init__(self, config, words:int, dropout:float=0.1, training:bool=True, bidirectional:bool=True):
        super().__init__()

        from abstract_structure.config.generation_config import get_generation_config
        self.cfg = Config(config)
        self.training = training

        self.bidirectional = bidirectional
        self.pad_masking = PadMasking(self.cfg['pad_idx'])
        self.future_masking = FutureMasking()

        self.positional_embedding = PositionalEmbedding(self.cfg['n_seq'], self.cfg['d_hidn'])
        self.token_embedding = TokenEmbedding(words + self.cfg["n_special_char"], self.cfg['d_hidn'])
        self.dropout_embedding = nn.Dropout(dropout)

        self.transformers = nn.ModuleList([
            TransformerLayer(self.cfg['n_head'], self.cfg['d_hidn'], self.cfg['rate'], training, dropout)
            for _ in range(self.cfg['n_layer'])])
        self.ln_head = LayerNorm(self.cfg['d_hidn'])

    def forward(self,
                x: torch.Tensor,
                past = None
                ):

        offset = past[0][0].size(-2) if past is not None else 0

        # Create masking tensor.
        mask = self.pad_masking(x, offset)
        if not self.bidirectional:
            mask = mask + self.future_masking(x, offset)
            mask = mask.to(x.to(self.device))

        # Use token embedding and positional embedding layers.
        x = self.token_embedding(x) + self.positional_embedding(x, offset)
        x = self.dropout_embedding(x)

        # Apply transformer layers sequentially.
        present = []
        for i, transformer in enumerate(self.transformers):
            x = transformer(x, past[i] if past is not None else None, mask)

            if not self.training:
                present.append(x[1])  # (k, v) ���� ��. ==> ���� �������� past �� ��.
                x = x[0]

        x = self.ln_head(x)
        x = self.token_embedding(x, transposed=True)

        return x if self.training else (x, present)