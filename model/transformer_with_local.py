import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import copy


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class EncoderLayer(nn.Module):

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)



def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention_withlocal(nn.Module):
    def __init__(self, h, d_model, local_kernel_size, dropout=0.1):
        super(MultiHeadedAttention_withlocal, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.local_kernel_size = local_kernel_size
        self.local_conv_layer = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.local_kernel_size,
                                          padding=int((self.local_kernel_size-1)/2))

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]


        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        value_conv = value.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k).transpose(1,2)
        local_conv_feat = self.local_conv_layer(value_conv).permute(0,2,1)

        return self.linears[-1](x+local_conv_feat)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.selu = nn.SELU()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.selu(self.w_1(x))))

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Transformer_local(nn.Module):
    def __init__(self, hidden_size, head_num, initial_embedding, encoder_layers, local_kernel_size, dropout_rate=0.2):
        super(Transformer_local, self).__init__()
        self.model_embeddings = initial_embedding
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.head_num  = head_num
        self.encoder_layers = encoder_layers
        self.local_kernel_size =local_kernel_size
        c = copy.deepcopy
        attn = MultiHeadedAttention_withlocal(self.head_num, self.hidden_size, self.local_kernel_size, self.dropout_rate)
        ff = PositionwiseFeedForward(self.hidden_size, self.hidden_size * 4, self.dropout_rate)
        self.position = PositionalEncoding(self.hidden_size, self.dropout_rate)
        self.encoder = Encoder(EncoderLayer(self.hidden_size, c(attn), c(ff), dropout_rate), self.encoder_layers)
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, mode, *input):
        if mode == 'calc_text_and_visit':
            return self.calc_text_and_visit(*input)

    def calc_text_and_visit(self, input, mask):
        input = self.position(input)
        contexts = self.encoder(input, mask)
        return contexts