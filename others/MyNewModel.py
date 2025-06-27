import torch.nn as nn
import torch
from torch.nn.init import *
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# 定义Embeddings类来实现文本嵌入层
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 将x传给self.lut并与根号下self.d_model相乘作为结果返回，缩放
        return self.lut(x) * math.sqrt(self.d_model)

# 位置编码类
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    # d_model: 词嵌入维度, dropout: 置0比率, max_len: 每个句子的最大长度
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化一个位置编码矩阵, 它是一个0阵，矩阵的大小是max_len x d_model
        pe = torch.zeros(max_len, d_model)
        # 初始化一个绝对位置矩阵,词汇的绝对位置就是用它的索引去表示，使向量变成一个max_len x 1 的矩阵
        position = torch.arange(0, max_len).unsqueeze(1)

        # 长为d_model的离散余弦变换（DCT）系数序列
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)# 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)#奇数位置
        pe.requires_grad = False
        # 拓展维度，使其成为一个max_len x d_model的矩阵
        # pe = pe.unsqueeze(0)
        # 把pe位置编码矩阵注册成模型的buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x = x + self.pe[:x.size(0), :]#Variable(self.pe[:x.size(0), :], requires_grad=False)
        x = x + self.pe[:x.size(1), :].unsqueeze(0).repeat(x.size(0), 1, 1)  ## modified by Bing to adapt to batch
        return self.dropout(x)


# 注意力机制的实现, 输入分别是query, key, value, mask: 掩码张量, dropout是nn.Dropout层的实例化对象, 默认为None
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    # 缩放点积注意力计算，得到注意力得分张量scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# 多头注意力机制的处理
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

# 实现前馈全连接层
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# 实现规范化层的类
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# 实现子层连接结构的类
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

# 实现编码器层
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

# 实现编码器的类
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


'''
def make_model(src_vocab, tgt_vocab, N=6,
               d_model=200, d_ff=800, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)  # 多头attention
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # FFN
    position = PositionalEncoding(d_model, dropout)  # 位置向量
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

'''

class Transformer(nn.Module):
    def __init__(self, hidden_dim, N, H):
        super(Transformer, self).__init__()
        # self. pos_encoding = PositionalEncoding(hidden_dim, 0.1)
        self.model = Encoder(
            EncoderLayer(hidden_dim, MultiHeadedAttention(H, hidden_dim),
                         PositionwiseFeedForward(hidden_dim, hidden_dim * 4),0.1),N)

    def forward(self, x, mask=None):
        # x = self.pos_encoding(x)
        return self.model(x, mask)

class TransformerM(torch.nn.Module):
    def __init__(self, args):
        super(TransformerM, self).__init__()
        self.args = args
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.transformer = Transformer(90, args.hlayers, 9)

        if args.vlayers == 0:
            self.v_transformer = None
            self.dense = torch.nn.Linear(90, 11)
        else:
            self.v_transformer = Transformer(200, args.vlayers, 20)
            self.linear = torch.nn.Linear(200, 11)
            self.dense = torch.nn.Linear(90, 11)

        #self.linear = torch.nn.Linear(2000, args.com_dim)
        self.cls = torch.nn.Parameter(torch.zeros([1, 1, 90], dtype=torch.float, requires_grad=True))
        self.sep = torch.nn.Parameter(torch.zeros([1, 1, 90], dtype=torch.float, requires_grad=True))
        torch.nn.init.xavier_uniform_(self.cls, gain=1)
        torch.nn.init.xavier_uniform_(self.sep, gain=1)

    # 实现两个输入张量的融合操作
    def fusion(self, x, y):
        y = self.softmax(self.linear(y))
        x = self.softmax(self.dense(x))
        predict = x + y
        return predict

    def forward(self, data):
        d1 = data.size(dim=0)
        d3 = data.size(2)
        x = data.unsqueeze(-2)
        x = data.view(d1, -1, self.args.sample, d3)
        x = torch.sum(x, dim=-2).squeeze(-2)
        x = torch.div(x, self.args.sample)
        #x = torch.cat((self.cls.repeat(d1, 1, 1), x), dim=1)
        dx = x.size(1)
        x = self.transformer(x)
        x = torch.div(torch.sum(x, dim=1).squeeze(dim=1), dx)
        #x = x[:, 0, :]
        if self.v_transformer is not None:
            y = data.view(-1, 200, 3, 30)
            y = torch.sum(y, dim=-2).squeeze(-2)
            y = y.transpose(-1, -2)
            d2 = y.size(1)
            y = self.v_transformer(y)
            dy = y.size(1)*3
            y = torch.div(torch.sum(y, dim=1).squeeze(dim=1), dy)
            predict = self.fusion(x, y)
        else:
            predict = self.softmax(self.dense(x))

        return predict
