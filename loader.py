# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # self.vocab = load_vocab(config["vocab_path"])
        # self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])

        self.tokenizer = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.tokenizer.vocab)
        # self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        # self.train_data_size = config["epoch_data_size"]  # 由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None  # 用来标识加载的是训练集还是测试集 "train" or "test",后边赋值
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)  # 跟dict.setdefault()的效果相同，其中value部分是list，可以以列表形式存储多个value
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                # 加载训练集。将训练集的每行的挨个问题进行读，转换成词表的数字表示，存储成字典，分类为key，每个问题的数字表示为一个tensor
                if isinstance(line, dict):  # 如果是dict类型，则为训练集
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        # 用非bert时，自己编码
                        # input_id = self.encode_sentence(question)
                        # 用bert时，加载bert的预测字表
                        input_id = self.tokenizer.encode(question, max_length=self.config["max_length"],
                                                         pad_to_max_length=True)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                # 加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    # input_id = self.encode_sentence(question)
                    input_id = self.tokenizer.encode(question, max_length=self.config["max_length"],
                                                     pad_to_max_length=True)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type  # 逗号后是出错后抛出的信息
            return len(self.data)  #

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample()  # 随机生成一个训练样本
        else:
            return self.data[index]

    # 依照一定概率生成负样本或正样本。正样本，包括一个标准问的两个问题，target是1.负样本包括两个标准问的问题，target是-1
    # 负样本从随机两个不同的标准问题中各随机选取一个
    # 正样本从随机一个标准问题中随机选取两个
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        # 随机正样本
        if random.random() <= self.config["positive_sample_rate"]:
            p = random.choice(standard_question_index)  # 从所有的标准问中取得一个编号
            # 如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
            if len(self.knwb[p]) < 2:
                return self.random_train_sample()
            else:
                s1, s2 = random.sample(self.knwb[p], 2)  # 从knwb中的序号p的样本中，随机sample两个句子出来
                return [s1, s2, torch.LongTensor([1])]
        # 随机负样本
        else:
            p, n = random.sample(standard_question_index, 2)  # 从所有标准问中取得两个编号
            s1 = random.choice(self.knwb[p])  # 形成不同标准问下的负样本
            s2 = random.choice(self.knwb[n])
            return [s1, s2, torch.LongTensor([-1])]  # 一条输入，但是有两个样本


# 加载字表或词表
def load_vocab(vocab_path):
    #bert的情况
    tokenizer = BertTokenizer(vocab_path)
    return tokenizer
    # 非bert
    # token_dict = {}
    # with open(vocab_path, encoding="utf8") as f:
    #     for index, line in enumerate(f):
    #         token = line.strip()
    #         token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    # return token_dict


# 加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())
        # return json.load(f)


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
