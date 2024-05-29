import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as func
from typing import Tuple, Optional
from torch import Tensor

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, norm=None):
        super(TransformerEncoder, self).__init__()
        #两层transformer_encoder为同一种类
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1)
        self.encoder_layer2 = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1)
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None)-> Tuple[Tensor, Tensor]:
        #顺序通过
        out1 = self.encoder_layer1(src, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        out2 = self.encoder_layer2(out1, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        if self.norm is not None:
            out2 = self.norm(out2)
        return out2

class PositionalEncoding(nn.Module):
    '''
    实现positional encoding, dropout max_seq_len 两个超参数设置参考 Personalized Transformer for Explainable Recommendation. ACL'21.
    代码实现参照，Attention is all you need.
    '''
    def __init__(self, d_model, dropout=0.1, max_seq_len=5000):
        '''
        :param d_model:word embedding size
        :param max_seq_len:max length of the sequence
        '''
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)#创建position enconding
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, d_model) -> (1, max_len, d_model) -> (max_len, 1, d_model)
        self.register_buffer('pe', pe)#pe是position encoding 只与位置有关，无需optim.step更新，故注册buffer

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MLP(nn.Module):
    def __init__(self, emsize=512):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(emsize, emsize)
        self.linear2 = nn.Linear(emsize, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        self.linear1.weight.data.uniform_(-0.1, 0.1)
        self.linear2.weight.data.uniform_(-0.1, 0.1)
        self.linear1.bias.data.zero_()
        self.linear2.bias.data.zero_()

    def forward(self, hidden):  # (batch_size, emsize)
        temp1 = self.linear1(hidden) #(batch_size, emsize)
        temp2 = self.sigmoid(temp1)  # (batch_size, emsize)
        rating = torch.squeeze(self.linear2(temp2))  # (batch_size,)
        return rating


def get_attn_mask(src_len, tgt_len):
    total_len = src_len + tgt_len
    mask = torch.tril(torch.ones(total_len, total_len))  # 矩阵的下半部分为1，其余部分为0
    mask[0, 1] = False  #user和item间的mask取消，使其能互相看见
    return mask


def get_keypadding_mask(text, pad_idx, batchsize, device):
    '''
    生成keypadding_mask, user, item no mask, text padding部分mask
    '''
    ui_mask = torch.zeros(batchsize, 2).bool().to(device)#user item 无需 mask
    text_mask = torch.tensor(text.t() == pad_idx).to(device)   #仅mask padding
    return torch.cat([ui_mask, text_mask], 1)


class PETER(nn.Module):
    def __init__(self, peter_mask, src_len, tgt_len, pad_idx, user_size, item_size, token_size, emsize, nhead, nhid, nlayers, dropout=0.5):
        '''
        主模型， 将(user, item, src) 经过两层transformer后， 利用hidden[1]完成context prediction；利用hidden[0]，经过MLP完成推荐任务
        利用hidden[2:]完成训练阶段的sentence prediction, 利用hidden[-1]完成测试阶段的explanation generation
        :param pad_idx: padding index
        :param user_size: user embedding 的大小
        :param item_size: item embedding 的大小
        :param token_size: token embedding 的大小
        :param emsize: word embedding 的大小
        :param nhead: transformer 参数， 多头注意力的个数
        :param nhid: transformer 参数， dimension of feedforward
        '''
        super(PETER, self).__init__()
        self.user_embeddings = nn.Embedding(user_size, emsize)
        self.item_embeddings = nn.Embedding(item_size, emsize)
        self.word_embeddings = nn.Embedding(token_size, emsize)
        self.hidden2token = nn.Linear(emsize, token_size)
        self.recommender = MLP(emsize)
        self.position_encoder = PositionalEncoding(emsize, dropout)
        self.transformer_encoder = TransformerEncoder(emsize, nhead, nhid)

        self.src_len = src_len
        self.pad_idx = pad_idx
        self.emsize = emsize
        self.attn_mask = get_attn_mask(src_len, tgt_len)
        self.initialize()

    def initialize(self):
        self.user_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.item_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.word_embeddings.weight.data.uniform_(-0.1, 0.1)
        self.hidden2token.weight.data.uniform_(-0.1, 0.1)
        self.hidden2token.bias.data.zero_()

    def context_predict(self, hidden):
        #context prediction 利用hidden[1] 经过log softmax
        context_prob = self.hidden2token(hidden[1])  # (batch_size, ntoken)
        context_pred = func.log_softmax(context_prob, dim=-1)
        return context_pred

    def recommend(self, hidden):
        #recommend任务， 使用MLP
        rating = self.recommender(hidden[0])  # (batch_size,)
        return rating

    def sequence_predict(self, hidden):
        #sequence prediction 任务， 训练时生成tgt_len 的log probablity
        word_prob = self.hidden2token(hidden[self.src_len:])  # (tgt_len, batch_size, ntoken)
        word_pred= func.log_softmax(word_prob, dim=-1)
        return word_pred

    def generate_token(self, hidden):
        #generate token任务， 用于预测时的explanation generation，迭代生成下一个单词
        word_prob = self.hidden2token(hidden[-1])  # (batch_size, ntoken)
        word_pred = func.log_softmax(word_prob, dim=-1)
        return word_pred

    def forward(self, user, item, text, seq_prediction=True, context_prediction=True, recommend=True):
        '''
        :param user: (batch_size,)
        :param item: (batch_size,)
        :param text: (total_len - 2, batch_size)
        :param seq_prediction: bool 是否进行sequence prediction 任务, False 执行 token generation 任务
        :param context_prediction: bool 是否进行context_prediction 任务
        :param recommend: bool  是否进行 recommend 任务
        :return word_pred: (tgt_len, batch_size, ntoken) if seq_prediction=True; (batch_size, ntoken) otherwise.
        :return log_context_dis: (batch_size, ntoken) if context_prediction=True; None otherwise.
        :return rating: (batch_size,) if rating_prediction=True; None otherwise.
        '''
        device = user.device
        batch_size = user.size(0)
        total_len = 2 + text.size(0)
        attn_mask = self.attn_mask[:total_len, :total_len].to(device)  # (total_len, total_len)
        key_padding_mask = get_keypadding_mask(text, self.pad_idx, batch_size,device) # (batch_size, total_len)

        #得到embedding
        user_src = self.user_embeddings(user.unsqueeze(0))  # (1, batch_size, emsize)
        item_src = self.item_embeddings(item.unsqueeze(0))  # (1, batch_size, emsize)
        word_src = self.word_embeddings(text)  # (total_len - 2, batch_size, emsize)
        src = torch.cat([user_src, item_src, word_src], 0)  # (total_len, batch_size, emsize)
        src = self.position_encoder(src)
        hidden = self.transformer_encoder(src, attn_mask, key_padding_mask)  # (total_len, batch_size, emsize)
        if recommend:
            rating = self.recommend(hidden)  # (batch_size,)
        else:
            rating = None
        if context_prediction:
            context_pred = self.context_predict(hidden)  # (batch_size, ntoken)
        else:
            context_pred = None
        if seq_prediction:  #seq_prediction 说明是否处于训练阶段，若是，则将srclen之后的所有解释text放入hidden2token中进行log likelyhood计算
            word_pred = self.sequence_predict(hidden)  # (tgt_len, batch_size, ntoken)
        else:
            word_pred = self.generate_token(hidden)  # (batch_size, ntoken)#若不是，则只将最后一个放入hidden2token中计算，便于生成后续解释
        return word_pred, context_pred, rating

