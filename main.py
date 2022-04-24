# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import SiameseNetwork, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
表示型文本匹配
全流程
1-每个标准问对应多个疑似问，最多的有898条疑似问题
2-训练阶段，随机确定正样本或者负样本，正样本是一个标注问的两个疑似问，label为1。负样本是不同标注问的疑似问，label为-1。一起输入cosinloss
3-预测阶段，输入语句和真实label。将输入语句进行编码，与所有疑似问计算余弦相似度，然后排序获取最相似的，转换为标准问。判断标准问是否等于真实label
4-lstm能达到90%，线性层能达到87%
"""

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = SiameseNetwork(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id1, input_id2, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id1, input_id2, labels)
            train_loss.append(loss.item())
            # if index % int(len(train_data) / 2) == 0:
            #     logger.info("batch loss %f" % loss)
            loss.backward()
            optimizer.step()
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        evaluator.eval(epoch)
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)
    return

if __name__ == "__main__":
    main(Config)