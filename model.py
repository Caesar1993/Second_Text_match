# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

"""
建立网络模型结构
"""


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        # hidden_size = config["hidden_size"]
        # vocab_size = config["vocab_size"] + 1
        # max_length = config["max_length"]
        # lstm
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        # linear
        # self.layer = nn.Linear(hidden_size, hidden_size)#bert什么的都可以用
        pretrain_model_path = config["pretrain_model_path"]
        self.bert_encoder = BertModel.from_pretrained(pretrain_model_path)
        # hidden_size = self.bert_encoder.pooler.dense.out_features

        # 不清楚dropout是否管用，所以先加上
        # self.dropout = nn.Dropout(0.5)

    # 输入为问题字符编码
    def forward(self, x):
        # sentence_length = torch.sum(x.gt(0), dim=-1)
        # x = self.embedding(x)
        # 使用lstm可以使用以下部分
        # x = pack_padded_sequence(x, sentence_length, batch_first=True, enforce_sorted=False)
        # x, _ = self.layer(x)
        # x, _ = pad_packed_sequence(x, batch_first=True)
        # 使用线性层
        # x = self.layer(x)
        # 使用bert
        x = self.bert_encoder(x)[0]
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()  # squeeze是把单维度的删除
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)  # 定义一个编码器，bert在内的所有编码器都可以用
        self.loss = nn.CosineEmbeddingLoss()  # 此loss需要输入三个值

    # 计算余弦距离
    # 0.5 * (1 + cosine)的目的是把-1到1的余弦值转化到0-1，这样可以直接作为得分
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)  # A除以A的模长
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)  # B除以B的模长
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 0.5 * (1 + cosine)

    # def cosine_triplet_loss(self, a, p, n, margin=None):#这部分没用用到，需要重新组织输入
    #     ap = self.cosine_distance(a, p)
    #     an = self.cosine_distance(a, n)
    #     if margin is None:
    #         diff = an - ap + 0.1
    #     else:
    #         diff = an - ap + margin.squeeze()
    #     return torch.mean(diff[diff.gt(0)])#gt0是大于0的部分，即选取大于0的部分

    def forward(self, sentence1, sentence2=None, target=None):
        # 同时传入两个句子，形成ppt中的图示
        if sentence2 is not None:
            vector1 = self.sentence_encoder(sentence1)
            vector2 = self.sentence_encoder(sentence2)
            # 如果有标签，则计算loss
            if target is not None:
                return self.loss(vector1, vector2, target.squeeze())
            # 如果无标签，计算余弦距离
            else:
                return self.cosine_distance(vector1, vector2)
        # 单独传入一个句子时，认为正在使用向量化能力
        else:
            return self.sentence_encoder(sentence1)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
