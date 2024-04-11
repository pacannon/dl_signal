import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class CMultiheadAttention(nn.Module):
    """CMulti-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = torch.nn.Parameter(torch.empty(3 * embed_dim, embed_dim, dtype=torch.complex64))

        if bias:
            self.in_proj_bias = torch.nn.Parameter(torch.empty((3 * embed_dim), dtype=torch.complex64))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, dtype=torch.complex64)

        if add_bias_kv:
            self.bias_k = torch.nn.Parameter(torch.empty((1, 1, embed_dim), dtype=torch.complex64))
            self.bias_v = torch.nn.Parameter(torch.empty((1, 1, embed_dim), dtype=torch.complex64))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, x, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        tgt_len, bsz, embed_dim = x.size()
        assert embed_dim == self.embed_dim

        # projecting q, k, v 
        q, k, v = self.in_proj(x)

        # scaling q 
        q = q * self.scaling

        # extending k, v by one time step at the end, with self.bias_k and self.bias_v 
        if self.bias_k is not None:
            assert self.bias_v is not None

            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])

            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        # extending k, v by another time step at the end, (bsz * num_heads, 1, head_dim) of zeros 
        if self.add_zero_attn:

            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)

            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights *= attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False
                
        attn_weights = (attn_weights.real + attn_weights.imag)
        attn_weights = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min())

        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn_weights = torch.complex(attn_weights, torch.zeros_like(attn_weights))

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads

        return attn, attn_weights

    def in_proj(self, x):
        # Perform the combined projection for Q, K, and V in one step
        combined_projection = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        
        # Split the combined projection into Q, K, and V
        q, k, v = combined_projection.split(self.embed_dim, dim=-1)
        return q, k, v

