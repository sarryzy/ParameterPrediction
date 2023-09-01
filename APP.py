"""
1.指定函数形式
2.生成训练数据
3.设置训练网络
4.测试
"""
import csv
import math
import os
import generateDatas
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torch import nn
import torch
import matplotlib.pyplot as plt
from typing import List
import math
import subprocess
import train
from MyClass import MetaData
paras=MetaData()

# 生成数据
generateDatas.generateData(paras.limit,paras.x,paras.filePath,paras.dataSum)
print("============开始训练=============")
subprocess.run(['python','train.py'])
print("============开始测试=============")
subprocess.run(['python','test.py'])