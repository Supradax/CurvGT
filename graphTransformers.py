import copy
import torch
import math
import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SelfAttention(nn.Module):
    def __init__(self, input_dim, heads):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = input_dim // heads
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)
        
    def forward(self, value, key, query, mask=None):
        batch_size = query.shape[0]
        
        Q = self.query(query).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch_size, heads, seq_len, seq_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(energy, dim=-1)  # (batch_size, heads, seq_len, seq_len)
        
        out = torch.matmul(attention, V)  # (batch_size, heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.head_dim)

        out = self.fc_out(out)  # (batch_size, seq_len, input_dim)
        
        return out

def linear_attention(query, key, value,
                     mask=None, dropout=None,
                     attention_type='galerkin'):
    '''
    Adapted from lucidrains' implementaion
    https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    to https://nlp.seas.harvard.edu/2018/04/03/attention.html template
    linear_attn function
    Compute the Scaled Dot Product Attention globally
    '''

    seq_len = query.size(-2)
    if attention_type in ['linear', 'global']:
        query = query.softmax(dim=-1)
        key = key.softmax(dim=-2)
    scores = torch.matmul(key.transpose(-2, -1), value)

    if mask is not None:
        raise RuntimeError("linear attention does not support casual mask.")

    p_attn = scores / seq_len

    if dropout is not None:
        p_attn = F.dropout(p_attn)

    out = torch.matmul(query, p_attn)
    return out, p_attn

def causal_linear_attn(query, key, value, kv_mask = None, dropout = None, eps = 1e-7):
    '''
    Modified from https://github.com/lucidrains/linear-attention-transformer
    '''
    bsz, n_head, seq_len, d_k, dtype = *query.shape, query.dtype

    key /= seq_len

    if kv_mask is not None:
        mask = kv_mask[:, None, :, None]
        key = key.masked_fill_(~mask, 0.)
        value = value.masked_fill_(~mask, 0.)
        del mask
    
    b_q, b_k, b_v = [x.reshape(bsz, n_head, -1, 1, d_k) for x in (query, key, value)]

    b_k_sum = b_k.sum(dim=-2)
    b_k_cumsum = b_k_sum.cumsum(dim = -2).type(dtype)

    p_attn = torch.einsum('bhund,bhune->bhude', b_k, b_v)
    p_attn = p_attn.cumsum(dim = -3).type(dtype)
    if dropout is not None:
        p_attn = F.dropout(p_attn)

    D_inv = 1. / torch.einsum('bhud,bhund->bhun', b_k_cumsum + eps, b_q)
    attn = torch.einsum('bhund,bhude,bhun->bhune', b_q, p_attn, D_inv)
    return attn.reshape(*query.shape), p_attn

def attention(query, key, value,
              mask=None, dropout=None, weight=None,
              attention_type='softmax'):
    '''
    Simplified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    Compute the Scaled Dot Product Attention
    '''

    d_k = query.size(-1)

    if attention_type == 'cosine':
        p_attn = F.cosine_similarity(query, key.transpose(-2, -1)) \
            / math.sqrt(d_k)
    else:
        scores = torch.matmul(query, key.transpose(-2, -1)) \
            / math.sqrt(d_k)
        seq_len = scores.size(-1)

        if attention_type == 'softmax':
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            p_attn = F.softmax(scores, dim=-1)
        elif attention_type in ['fourier', 'integral', 'local']:
            if mask is not None:
                scores = scores.masked_fill(mask == 0, 0)
            p_attn = scores / seq_len

    if dropout is not None:
        p_attn = F.dropout(p_attn)

    out = torch.matmul(p_attn, value)

    return out, p_attn

class SimpleAttention(nn.Module):
    '''
    The attention is using a vanilla (QK^T)V or Q(K^T V) with no softmax
    For an encoder layer, the tensor size is slighly different from the official pytorch implementation

    attn_types: 
        - fourier: integral, local
        - galerkin: global
        - linear: standard linearization
        - softmax: classic softmax attention

    In this implementation, output is (N, L, E).
    batch_first will be added in the next version of PyTorch: https://github.com/pytorch/pytorch/pull/55285

    Reference: code base modified from
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
    - added xavier init gain
    - added layer norm <-> attn norm switch
    - added diagonal init

    In https://github.com/lucidrains/linear-attention-transformer/blob/master/linear_attention_transformer/linear_attention_transformer.py
    the linear attention in each head is implemented as an Einstein sum
    attn_matrix = torch.einsum('bhnd,bhne->bhde', k, v)
    attn = torch.einsum('bhnd,bhde->bhne', q, attn_matrix)
    return attn.reshape(*q.shape)
    here in our implementation this is achieved by a slower transpose+matmul
    but can conform with the template Harvard NLP gave
    '''

    def __init__(self, n_head, d_model,
                 pos_dim: int = 1,
                 attention_type='fourier',
                 dropout=0.1,
                 xavier_init=1e-4,
                 diagonal_weight=1e-2,
                 symmetric_init=False,
                 norm=False,
                 norm_type='layer',
                 eps=1e-5,
                 debug=False):
        super(SimpleAttention, self).__init__()
        assert d_model % n_head == 0
        self.attention_type = attention_type
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.pos_dim = pos_dim
        self.linears = nn.ModuleList(
            [copy.deepcopy(nn.Linear(d_model, d_model)) for _ in range(3)])
        self.xavier_init = xavier_init
        self.diagonal_weight = diagonal_weight
        self.symmetric_init = symmetric_init
        # if self.xavier_init > 0:
            # self._reset_parameters()
        self.add_norm = norm
        self.norm_type = norm_type
        if norm:
            self._get_norm(eps=eps)

        if pos_dim > 0:
            self.fc = nn.Linear(d_model + n_head*pos_dim, d_model)

        self.attn_weight = None
        self.dropout = nn.Dropout(dropout)
        self.debug = debug

    def forward(self, query, key, value, pos=None, mask=None, weight=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        bsz = query.size(0)
        if weight is not None:
            query, key = weight*query, weight*key

        query, key, value = \
            [layer(x).view(bsz, -1, self.n_head, self.d_k).transpose(1, 2)
             for layer, x in zip(self.linears, (query, key, value))]

        if self.add_norm:
            if self.attention_type in ['linear', 'galerkin', 'global']:
                if self.norm_type == 'instance':
                    key, value = key.transpose(-2, -1), value.transpose(-2, -1)

                key = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], dim=1)
                value = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_V, (value[:, i, ...] for i in range(self.n_head)))], dim=1)

                if self.norm_type == 'instance':
                    key, value = key.transpose(-2, -1), value.transpose(-2, -1)
            else:
                if self.norm_type == 'instance':
                    key, query = key.transpose(-2, -1), query.transpose(-2, -1)

                key = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_K, (key[:, i, ...] for i in range(self.n_head)))], dim=1)
                query = torch.stack(
                    [norm(x) for norm, x in
                     zip(self.norm_Q, (query[:, i, ...] for i in range(self.n_head)))], dim=1)

                if self.norm_type == 'instance':
                    key, query = key.transpose(-2, -1), value.transpose(-2, -1)

        if pos is not None and self.pos_dim > 0:
            assert pos.size(-1) == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.n_head, 1, 1])
            query, key, value = [torch.cat([pos, x], dim=-1)
                                 for x in (query, key, value)]

        if self.attention_type in ['linear', 'galerkin', 'global']:
            x, self.attn_weight = linear_attention(query, key, value,
                                                   mask=mask,
                                                   attention_type=self.attention_type,
                                                   dropout=self.dropout)
        elif self.attention_type == 'causal':
            assert mask is not None
            x, self.attn_weight = causal_linear_attn(query, key, value,
                                                   mask=mask,
                                                   dropout=self.dropout)
        else:
            x, self.attn_weight = attention(query, key, value,
                                            mask=mask,
                                            attention_type=self.attention_type,
                                            dropout=self.dropout)

        out_dim = self.n_head * self.d_k if pos is None else self.n_head * \
            (self.d_k + self.pos_dim)
        att_output = x.transpose(1, 2).contiguous().view(bsz, -1, out_dim)

        if pos is not None and self.pos_dim > 0:
            att_output = self.fc(att_output)

        return att_output

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class TransformerLayer_GT(nn.Module):
    def __init__(self, input_dim, heads, hidden_dim, output_dim, dropout=0.1):
        super(TransformerLayer_GT, self).__init__()
        self.attention = SelfAttention(input_dim, heads)
        self.fourierAttention = SimpleAttention(d_model=input_dim,n_head=heads,attention_type='fourier')
        self.galerkinAttention = SimpleAttention(d_model=input_dim,n_head=heads,attention_type='galerkin')

        self.feed_forward = FeedForward(input_dim, hidden_dim, output_dim)
        
        self.layernorm1 = nn.LayerNorm([input_dim])
        self.layernorm2 = nn.LayerNorm([input_dim])
        
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x, mask=None):
        attention_out = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout(attention_out).view(x.shape[0],-1))  # Add & Norm
        ff_out = self.feed_forward(x)
        x = self.layernorm2(x + self.dropout(ff_out))  # Add & Norm
        return x

class TransformerLayer_Fourier(nn.Module):
    def __init__(self, input_dim, heads, hidden_dim, output_dim, dropout=0.1):
        super(TransformerLayer_Fourier, self).__init__()
        self.attention = SelfAttention(input_dim, heads)
        self.fourierAttention = SimpleAttention(d_model=input_dim,n_head=heads,attention_type='fourier')
        self.galerkinAttention = SimpleAttention(d_model=input_dim,n_head=heads,attention_type='galerkin')

        self.feed_forward = FeedForward(input_dim, hidden_dim, output_dim)
        
        self.layernorm1 = nn.LayerNorm([input_dim])
        self.layernorm2 = nn.LayerNorm([input_dim])
        
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x, mask=None):
        attention_out = self.fourierAttention(x, x, x)

        x = self.layernorm1(x + self.dropout(attention_out).view(x.shape[0],-1))  # Add & Norm
        ff_out = self.feed_forward(x)
        x = self.layernorm2(x + self.dropout(ff_out))  # Add & Norm
        return x
    
class TransformerLayer_Galerkin(nn.Module):
    def __init__(self, input_dim, heads, hidden_dim, output_dim, dropout=0.1):
        super(TransformerLayer_Galerkin, self).__init__()
        self.attention = SelfAttention(input_dim, heads)
        self.fourierAttention = SimpleAttention(d_model=input_dim,n_head=heads,attention_type='fourier')
        self.galerkinAttention = SimpleAttention(d_model=input_dim,n_head=heads,attention_type='galerkin')

        self.feed_forward = FeedForward(input_dim, hidden_dim, output_dim)
        
        self.layernorm1 = nn.LayerNorm([input_dim])
        self.layernorm2 = nn.LayerNorm([input_dim])
        
        self.dropout = nn.Dropout(dropout)
        

    def forward(self, x, mask=None):
        attention_out = self.galerkinAttention(x, x, x)

        x = self.layernorm1(x + self.dropout(attention_out).view(x.shape[0],-1))  # Add & Norm
        ff_out = self.feed_forward(x)
        x = self.layernorm2(x + self.dropout(ff_out))  # Add & Norm
        return x
    