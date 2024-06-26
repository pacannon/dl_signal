import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from modules.position_embedding import SinusoidalPositionalEmbedding
from modules.multihead_attention import MultiheadAttention
from modules.c_multihead_attention import CMultiheadAttention
from models import *
import math


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_dim (int): embedding dimension of input signal
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
        complex_mha (bool): whether to use the reformulated complex multiheaded attention
        conj_attn (bool): whether to use the complex conjugate of the Key projections in the attention mechanism
        pre_ln (bool): whether to position encoder block Layer Norms before Attention and FF
        softmax (bool): whether to use softmax in the attention scoring
        rescale (bool): whether to rescale the embeddings by an additional factor after applying softmax
        squared_norm (bool): whether to use squared norm on the real and imaginary parts during attention score softmax
        re (bool): whether to use the real part of the complex number during attention score softmax
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            layers,
            attn_dropout,
            relu_dropout,
            res_dropout,
            attn_mask,
            complex_mha,
            conj_attn,
            pre_ln,
            softmax,
            rescale,
            squared_norm,
            minus_im,
            re,
        ):
        super().__init__()
        self.dropout = 0.3      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = 1
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)        
        self.attn_mask = attn_mask
        self.complex_mha = complex_mha
        self.conj_attn = conj_attn
        self.pre_ln = pre_ln
        self.softmax = softmax
        self.squared_norm = squared_norm
        self.minus_im = minus_im
        self.rescale = rescale
        self.re = re
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                relu_dropout=relu_dropout,
                res_dropout=res_dropout,
                attn_mask=attn_mask,
                complex_mha=complex_mha,
                conj_attn=conj_attn,
                pre_ln=pre_ln,
                softmax=softmax,
                rescale=rescale,
                squared_norm=squared_norm,
                minus_im=minus_im,
                re=re,
            )
            for _ in range(layers)
        ])

        self.variance_stats = False
        self.layer_stats = torch.zeros((layers, 2, 3), dtype=torch.float)

        self.register_buffer('version', torch.Tensor([2]))

    def forward(self, input_A, input_B):
        """
        Args:
            input_A: real part of input signal.
            input_B: imaginary part of input signal.
        """
        input_A = self.scale_embed_position_dropout(input_A)
        input_B = self.scale_embed_position_dropout(input_B)
        # For each transformer encoder layer:
        i = 0
        for layer in self.layers:
            input_A, input_B = layer(input_A, input_B)
                                 
            if self.variance_stats:
                self.update_stats(input_A, 0, i)
                self.update_stats(input_B, 1, i)
                i += 1
            
        return input_A, input_B
    
    def update_stats(self, vector, stats_index, layer_index):
        vector_flat = vector.view(-1).detach()
        n = vector_flat.numel()

        current_count = self.layer_stats[layer_index][stats_index, 0]
        new_count = current_count + n

        current_sum = self.layer_stats[layer_index][stats_index, 1]
        new_sum = current_sum + vector_flat.sum()

        current_sum_squares = self.layer_stats[layer_index][stats_index, 2]
        new_sum_squares = current_sum_squares + (vector_flat ** 2).sum()

        self.layer_stats[layer_index][stats_index, 0] = new_count
        self.layer_stats[layer_index][stats_index, 1] = new_sum
        self.layer_stats[layer_index][stats_index, 2] = new_sum_squares


    def get_and_reset_stats(self):
        stats_output = torch.zeros((self.layer_stats.shape[0], 2, 3), dtype=torch.float)

        for layer_index in range(self.layer_stats.shape[0]):
            for stats_index in range(2): 
                count = self.layer_stats[layer_index][stats_index, 0]
                total_sum = self.layer_stats[layer_index][stats_index, 1]
                sum_squares = self.layer_stats[layer_index][stats_index, 2]

                mean = total_sum / count if count != 0 else 0
                variance = (sum_squares - (total_sum ** 2) / count) / count if count != 0 else 0

                stats_output[layer_index, stats_index, 0] = count
                stats_output[layer_index, stats_index, 1] = mean
                stats_output[layer_index, stats_index, 2] = variance

        # Reset statistics
        self.layer_stats.zero_()
        return stats_output


    
    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(
            self,
            embed_dim,
            num_heads=4,
            attn_dropout=0.1,
            relu_dropout=0.1,
            res_dropout=0.1,
            attn_mask=False,
            complex_mha=False,
            conj_attn=False,
            pre_ln=False,
            softmax=False,
            rescale=1,
            squared_norm=True,
            minus_im=False,
            re=False,
        ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.complex_mha = complex_mha
        self.conj_attn = conj_attn
        self.pre_ln = pre_ln
        self.softmax = softmax
        self.rescale = rescale
        self.squared_norm = squared_norm
        self.minus_im = minus_im
        self.re = re
        self.self_attn = CMultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout,
            bias=True,
            add_bias_kv=True, 
            add_zero_attn=True,
            softmax=self.softmax,
            rescale=self.rescale,
            squared_norm=self.squared_norm,
            minus_im=self.minus_im,
            re=self.re,
        ) if self.complex_mha else MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout,
            bias=True,
            add_bias_kv=True, 
            add_zero_attn=True,
            softmax=self.softmax,
            rescale=self.rescale,
            squared_norm=self.squared_norm,
        )
        self.attn_mask = attn_mask
        self.crossmodal = True
        self.normalize = True

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = ComplexLinear(self.embed_dim, self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = ComplexLinear(self.embed_dim, self.embed_dim)   # The "Add & Norm" part in the paper

        self.layer_norms_A = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])
        self.layer_norms_B = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x_A, x_B):
        """
        Args:
            input_A: real part of input signal.
            input_B: imaginary part of input signal.
        """
        ## Attention Part
        # Residual and Layer Norm
        residual_A = x_A
        residual_B = x_B

        if self.pre_ln:
            x_A = self.layer_norms_A[0](x_A)
            x_B = self.layer_norms_B[0](x_B)

        if self.complex_mha:
            x_stacked = torch.stack((x_A, x_B), dim=-1)
            x = torch.view_as_complex(x_stacked)
            result = self.complex_attention_block(x)

            x_A = result.real
            x_B = result.imag
        else:
            # Multihead Attention
            x_aaa = self.attention_block(x_A, x_A, x_A)
            x_aab = self.attention_block(x_A, x_A, x_B)
            x_aba = self.attention_block(x_A, x_B, x_A)
            x_baa = self.attention_block(x_B, x_A, x_A)
            x_abb = self.attention_block(x_A, x_B, x_B)
            x_bab = self.attention_block(x_B, x_A, x_B)
            x_bba = self.attention_block(x_B, x_B, x_A)
            x_bbb = self.attention_block(x_B, x_B, x_B)


            if self.conj_attn:
                x_A = x_aaa + x_abb - x_bab + x_bba
                x_B = x_bbb + x_baa - x_aba + x_aab
            else:
                x_A = x_aaa - x_abb - x_bab - x_bba
                x_B = -x_bbb + x_baa + x_aba + x_aab
         
        if not self.pre_ln:
            x_A = self.layer_norms_A[0](x_A)
            x_B = self.layer_norms_B[0](x_B)

        # Dropout and Residual
        x_A = F.dropout(x_A, p=self.res_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.res_dropout, training=self.training)

        x_A = residual_A + x_A
        x_B = residual_B + x_B
    
        
        # ##FC Part
        residual_A = x_A
        residual_B = x_B

        if self.pre_ln:
            x_A = self.layer_norms_A[1](x_A)
            x_B = self.layer_norms_B[1](x_B)
        
        # FC1
        x_A, x_B = self.fc1(x_A, x_B)
        x_A = F.relu(x_A)
        x_B = F.relu(x_B)
        x_A = F.dropout(x_A, p=self.relu_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.relu_dropout, training=self.training)
        
        # FC2
        x_A, x_B = self.fc2(x_A, x_B)


        if not self.pre_ln:
            x_A = self.layer_norms_A[1](x_A)
            x_B = self.layer_norms_B[1](x_B)

        x_A = F.dropout(x_A, p=self.res_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.res_dropout, training=self.training)
        
        x_A = residual_A + x_A
        x_B = residual_B + x_B

        return x_A, x_B

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def attention_block(self, x, x_k, x_v):

        mask = None
        x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        return x

    def complex_attention_block(self, x):

        mask = None
        x, _ = self.self_attn(x, attn_mask=mask, conj_attn=self.conj_attn)
        return x

class TransformerDecoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_dim (int): embedding dimension of input signal
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, src_attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0, tgt_attn_dropout=0.0):
        super().__init__()
        self.dropout = 0.3      # Embedding dropout
        # self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = 1
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    src_attn_dropout=src_attn_dropout,
                                    relu_dropout=relu_dropout,
                                    res_dropout=res_dropout,
                                    tgt_attn_dropout=tgt_attn_dropout
                                    )
            for _ in range(layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))

    def forward(self, input_A, input_B, enc_A, enc_B):
        """
        Args:
            input_A: real part of input signal.
            input_B: imaginary part of input signal.
            enc_A: real part of encoder output.
            enc_B: imaginary part of encoder output.
        """
        input_A = self.scale_embed_position_dropout(input_A)
        input_B = self.scale_embed_position_dropout(input_B)
        
        # For each transformer encoder layer:
        for layer in self.layers:
            input_A, input_B = layer(input_A, input_B, enc_A, enc_B)
        return input_A, input_B

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class TransformerDecoderLayer(nn.Module): 
    def __init__(self, embed_dim, num_heads=4, src_attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, tgt_attn_dropout=0.1, src_mask=True, tgt_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=src_attn_dropout,
            bias=True,
            add_bias_kv=True,
            add_zero_attn=True, 
        )
        self.src_mask = src_mask   # used as last arg in forward function call 
        self.tgt_mask = tgt_mask   # used as last arg in forward function call 

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout

        self.attn = MultiheadAttention(
            embed_dim=self.embed_dim, 
            num_heads=self.num_heads, 
            attn_dropout=tgt_attn_dropout, 
            bias=True,
            add_bias_kv=True, 
            add_zero_attn=True, 
        )

        self.fc1 = ComplexLinear(self.embed_dim, self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = ComplexLinear(self.embed_dim, self.embed_dim)

        self.layer_norms_A = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(3)])
        self.layer_norms_B = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(3)])

    def forward(self, x_A, x_B, enc_A, enc_B):
        """
        Args:
            input_A: real part of input signal.
            input_B: imaginary part of input signal.
            enc_A: real part of encoder output.
            enc_B: imaginary part of encoder output.
        """
        ## Attention Part
        # Residual and Layer Norm
        residual_A = x_A
        residual_B = x_B
        
        # Self Attention
        if self.src_mask: 
            assert x_A.shape[0] == x_B.shape[0]
            mask = buffered_future_mask(x_A)
        else: 
            mask = None

        x_aaa, _ = self.self_attn(x_A, x_A, x_A, attn_mask=mask)
        x_aab, _ = self.self_attn(x_A, x_A, x_B, attn_mask=mask)
        x_aba, _ = self.self_attn(x_A, x_B, x_A, attn_mask=mask)
        x_baa, _ = self.self_attn(x_B, x_A, x_A, attn_mask=mask)
        x_abb, _ = self.self_attn(x_A, x_B, x_B, attn_mask=mask)
        x_bab, _ = self.self_attn(x_B, x_A, x_B, attn_mask=mask)
        x_bba, _ = self.self_attn(x_B, x_B, x_A, attn_mask=mask)
        x_bbb, _ = self.self_attn(x_B, x_B, x_B, attn_mask=mask)

        x_A = x_aaa - x_abb - x_bab - x_bba
        x_B = -x_bbb + x_baa + x_aba + x_aab
        
        # Layer Norm, Dropout and Residual;
        x_A = F.dropout(x_A, p=self.res_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.res_dropout, training=self.training)
        x_A += residual_A 
        x_B += residual_B
        x_A = self.layer_norms_A[0](x_A)
        x_B = self.layer_norms_B[0](x_B)
        
        residual_A = x_A
        residual_B = x_B
        
        # Attention between encoder and decoder 
        x_acc, _ = self.attn(x_A, enc_A, enc_A) 
        x_add, _ = self.attn(x_A, enc_B, enc_B) 
        x_bcd, _ = self.attn(x_B, enc_A, enc_B) 
        x_bdc, _ = self.attn(x_B, enc_B, enc_A) 
        x_acd, _ = self.attn(x_A, enc_A, enc_B)
        x_adc, _ = self.attn(x_A, enc_B, enc_A)
        x_bcc, _ = self.attn(x_B, enc_A, enc_A)
        x_bdd, _ = self.attn(x_B, enc_B, enc_B) 

        x_A = x_acc - x_add - x_bcd - x_bdc 
        x_B = x_acd + x_adc + x_bcc - x_bdd 

        # Layer Norm, Dropout and Residual;
        x_A = F.dropout(x_A, p=self.res_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.res_dropout, training=self.training)
        x_A += residual_A 
        x_B += residual_B
        x_A = self.layer_norms_A[1](x_A)
        x_B = self.layer_norms_B[1](x_B)
        
        residual_A = x_A 
        residual_B = x_B

        # FC1
        x_A, x_B = self.fc1(x_A, x_B)        
        x_A = F.relu(x_A)
        x_B = F.relu(x_B)
        x_A = F.dropout(x_A, p=self.relu_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.relu_dropout, training=self.training)

        # FC2
        x_A, x_B = self.fc2(x_A, x_B)
        x_A = F.dropout(x_A, p=self.res_dropout, training=self.training)
        x_B = F.dropout(x_B, p=self.res_dropout, training=self.training)
        
        x_A += residual_A 
        x_B += residual_B 
        x_A = self.layer_norms_A[2](x_A)
        x_B = self.layer_norms_B[2](x_B)
        #print("Attention here: ", x_A.mean().item(), x_B.mean().item())

        return x_A, x_B

class TransformerConcatEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_dim (int): embedding dimension of input signal
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, attn_dropout, relu_dropout, res_dropout, attn_mask=False):
        super().__init__()
        self.dropout = 0.3      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = 1
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerConcatEncoderLayer(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    attn_dropout=attn_dropout,
                                    relu_dropout=relu_dropout,
                                    res_dropout=res_dropout,
                                    attn_mask=attn_mask)
            for _ in range(layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))

    def forward(self, x):
        x = self.scale_embed_position_dropout(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class TransformerConcatEncoderLayer(nn.Module):
    """Encoder layer block.
    Args:
        embed_dim: Embedding dimension
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout,
            bias=True,
            add_bias_kv=True, 
            add_zero_attn=True
        )
        self.attn_mask = attn_mask
        self.normalize = True

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.embed_dim)

        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x):
        ## Attention Part
        # Residual and Layer Norm
        residual = x
        # Multihead Attention
        x = self.attention_block(x,x,x)

        x = self.layer_norms[0](x)
        # Dropout and Residual
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        
        # ##FC Part
        residual = x
        
        # FC1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        
        x = self.fc2(x)
        x = self.layer_norms[1](x)

        x = F.dropout(x, p=self.res_dropout, training=self.training)
        
        x = residual + x

        return x

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def attention_block(self, x, x_k, x_v):

        mask = None
        x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        return x

class TransformerConcatDecoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_dim (int): embedding dimension of input signal
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, src_attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0, tgt_attn_dropout=0.0):
        super().__init__()
        self.dropout = 0.3
        self.embed_dim = embed_dim
        self.embed_scale = 1
        self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)
        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerConcatDecoderLayer(embed_dim=embed_dim,
                                    num_heads=num_heads,
                                    src_attn_dropout=src_attn_dropout,
                                    relu_dropout=relu_dropout,
                                    res_dropout=res_dropout,
                                    tgt_attn_dropout=tgt_attn_dropout
                                    )
            for _ in range(layers)
        ])
        self.register_buffer('version', torch.Tensor([2]))
    
    def forward(self, input, enc):
        input = self.scale_embed_position_dropout(input)
        for layer in self.layers:
            input = layer(input, enc)
        return input

    def scale_embed_position_dropout(self, x_in):
        x = self.embed_scale * x_in
        if self.embed_positions is not None:
            y = self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            x += self.embed_positions(x_in.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        x = F.dropout(x, p=self.dropout, training=self.training) # may change
        return x


class TransformerConcatDecoderLayer(nn.Module): 
    def __init__(self, embed_dim, num_heads=4, src_attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, tgt_attn_dropout=0.1, src_mask=True, tgt_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=src_attn_dropout,
            bias=True,
            add_bias_kv=True,
            add_zero_attn=True, 
        )
        self.src_mask = src_mask   # used as last arg in forward function call 
        self.tgt_mask = tgt_mask   # used as last arg in forward function call 

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout

        self.attn = MultiheadAttention(
            embed_dim=self.embed_dim, 
            num_heads=self.num_heads, 
            attn_dropout=tgt_attn_dropout, 
            bias=True,
            add_bias_kv=True, 
            add_zero_attn=True, 
        )

        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.fc2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(3)])

    def forward(self, x, enc):
        residual = x 
        # Self Attention
        if self.src_mask: 
            mask = buffered_future_mask(x) 
        else: 
            mask = None
        x, _ = self.self_attn(x, x, x)
        # Layer Norm, Dropout and Residual;
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x += residual
        x = self.layer_norms[0](x)
        
        residual = x
        
        # Attention between encoder and decoder 
        x, _ = self.attn(x, enc, enc) 

        # Layer Norm, Dropout and Residual;
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x += residual
        x = self.layer_norms[1](x)
        
        residual = x

        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        
        x += residual
        x = self.layer_norms[2](x)

        return x

def fill_with_neg_inf(t):
    return t.float().fill_(float('-inf')).type_as(t)

def fill_with_one(t): 
    return t.float().fill_(float(1)).type_as(t)

def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.tril(fill_with_one(torch.ones(dim1, dim2)), 0)
    if tensor.is_cuda:
        future_mask = future_mask.cuda()
    return future_mask[:dim1, :dim2]


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim, eps=1e-20)
    return m





if __name__ == '__main__':
    encoder = TransformerEncoder(300, 4, 2)
    x = torch.tensor(torch.rand(20, 2, 300))
    print(encoder(x).shape)

