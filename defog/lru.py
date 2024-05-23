import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import copy

class LRUModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.bert_hidden_units
        self.layers = args.bert_num_blocks

        self.lru_blocks = nn.ModuleList([LRUBlock(self.args) for _ in range(self.layers)])

    def forward(self, x, h=None, mask=None):
        # left padding to the power of 2
        if mask is None:
            mask = torch.ones_like(x[..., 0]).bool()
        if h is None:
            h = [None] * self.layers
        if h[0] is not None:
            assert x.size(1) == 1, "h is not None, but x.size(1) != 1"
        seq_len = x.size(1)
        log2_L = int(np.ceil(np.log2(seq_len)))
        x = F.pad(x, (0, 0, 2 ** log2_L - x.size(1), 0, 0, 0))
        mask_ = F.pad(mask, (2 ** log2_L - mask.size(1), 0, 0, 0))

        # LRU blocks with pffn
        h_new = []
        for i, lru_block in enumerate(self.lru_blocks):
            x, h_new_i = lru_block.forward(x, h[i], mask_)
            h_new.append(h_new_i)
        x = x[:, -seq_len:]  # B x L x D (64)
        
        return x, h_new

class LRUBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        hidden_size = args.bert_hidden_units
        self.lru_layer = LRULayer(
            d_model=hidden_size, dropout=args.bert_attn_dropout)
        if args.gating:
            self.feed_forward = PositionwiseGatedForward(
                d_model=hidden_size, d_ff=hidden_size*4, dropout=args.bert_dropout)
        else:
            self.feed_forward = PositionwiseFeedForward(
                d_model=hidden_size, d_ff=hidden_size*4, dropout=args.bert_dropout)
    
    def forward(self, x, h, mask):
        z, h_new = self.lru_layer(x, h, mask)
        z = self.feed_forward(z, x)
        return z, h_new
    

class LRULayer(nn.Module):
    def __init__(self,
                 d_model,
                 dropout=0.1,
                 use_bias=True,
                 r_min=0.8,
                 r_max=0.99):
        super().__init__()
        self.embed_size = d_model
        self.hidden_size = 2 * d_model
        self.use_bias = use_bias

        # init nu, theta, gamma
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)
        nu_log = torch.log(-0.5 * torch.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta_log = torch.log(u2 * torch.tensor(np.pi) * 2)
        diag_lambda = torch.exp(torch.complex(-torch.exp(nu_log), torch.exp(theta_log)))
        gamma_log = torch.log(torch.sqrt(1 - torch.abs(diag_lambda) ** 2))
        self.params_log = nn.Parameter(torch.vstack((nu_log, theta_log, gamma_log)))

        # Init B, C, D
        self.in_proj = nn.Linear(self.embed_size, self.hidden_size, bias=use_bias).to(torch.cfloat)
        self.out_proj = nn.Linear(self.hidden_size, self.embed_size, bias=use_bias).to(torch.cfloat)
        # self.out_vector = nn.Parameter(torch.rand(self.embed_size))
        self.out_vector = nn.Identity()
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.embed_size)

    def lru_parallel(self, i, h, lamb, mask, B, L, D):
        # Parallel algorithm, see: https://kexue.fm/archives/9554#%E5%B9%B6%E8%A1%8C%E5%8C%96
        # The original implementation is slightly slower and does not consider 0 padding
        l = 2 ** i
        h = h.reshape(B * L // l, l, D)  # (B, L, D) -> (B * L // 2, 2, D)
        mask_ = mask.reshape(B * L // l, l)  # (B, L) -> (B * L // 2, 2)
        h1, h2 = h[:, :l // 2], h[:, l // 2:]  # Divide data in half

        if i > 1: lamb = torch.cat((lamb, lamb * lamb[-1]), 0)
        h2 = h2 + lamb * h1[:, -1:] * mask_[:, l // 2 - 1:l // 2].unsqueeze(-1)
        h = torch.cat([h1, h2], axis=1)
        return h, lamb

    def forward(self, x, prev, mask):
        # compute bu and lambda
        nu, theta, gamma = torch.exp(self.params_log).split((1, 1, 1))
        lamb = torch.exp(torch.complex(-nu, theta))
        h = self.in_proj(x.to(torch.cfloat)) * gamma  # bu
        
        # compute h in parallel
        log2_L = int(np.ceil(np.log2(h.size(1))))
        B, L, D = h.size(0), h.size(1), h.size(2)
        if prev is not None:
            assert L == 1, "prev is not None, but L != 1"
            h = lamb * prev + h
            now = h.clone()
        elif L == 1:
            now = h.clone()
        else:
            now = None
            for i in range(log2_L):
                h, lamb = self.lru_parallel(i + 1, h, lamb, mask, B, L, D)
        x = self.dropout(self.out_proj(h).real) + self.out_vector(x)
        return self.layer_norm(x), now  # residual connection introduced above 
    

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, *args, **kwargs):
        x_ = self.dropout(self.activation(self.w_1(x)))
        return self.layer_norm(self.dropout(self.w_2(x_)) + x)
    
class PositionwiseGatedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.gate1 = GRUGatingUnit(d_model)
        self.gate2 = GRUGatingUnit(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, intputs):
        o1 = self.gate1(intputs, self.activation(x))
        x2 = self.layer_norm(o1)
        m2 = self.mlp(x2)
        o2 = self.gate2(x2, m2)
        return o2
        
class GRUGatingUnit(torch.nn.Module):
    """
    Overview:
        The GRUGatingUnit module implements the GRU gating mechanism used in the GTrXL model.
    """

    def __init__(self, input_dim: int, bg: float = 2.):
        """
        Arguments:
            - input_dim (:obj:`int`): The dimensionality of the input.
            - bg (:obj:`bg`): The gate bias. By setting bg > 0 we can explicitly initialize the gating mechanism to \
                be close to the identity map. This can greatly improve the learning speed and stability since it \
                initializes the agent close to a Markovian policy (ignore attention at the beginning).
        """
        super(GRUGatingUnit, self).__init__()
        self.Wr = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ur = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Uz = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Wg = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.Ug = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.bg = nn.Parameter(torch.full([input_dim], bg))  # bias
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Overview:
            Compute the output value using the GRU gating mechanism.
        Arguments:
            - x: (:obj:`torch.Tensor`): The first input tensor.
            - y: (:obj:`torch.Tensor`): The second input tensor. \
                x and y should have the same shape and their last dimension should match the input_dim.
        Returns:
            - g: (:obj:`torch.Tensor`): The output of the GRU gating mechanism. \
                The shape of g matches the shapes of x and y.
        """
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))  # element wise multiplication
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g  # x.shape == y.shape == g.shape