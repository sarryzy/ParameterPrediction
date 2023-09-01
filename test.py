import time

import matplotlib.pyplot as plt

import Utils
from MyClass import MetaData
from train import MyModel
import torch
from scipy.stats import pearsonr
import numpy as np

paras=MetaData()
trainModel=torch.load(paras.savePthPath)
limit=paras.limit
n = len(limit)
x=paras.x
while True:
    print("======================")
    a = np.random.rand(n) * np.random.choice([-1, 1])
    for j in range(n):
        rang = limit[j]
        a[j] = (rang[0] + rang[1]) / 2 + (rang[1] - rang[0]) / 2 * a[j]
    a = list(a)
    noise=np.random.rand(1)*0
    print("当前噪声：",noise.item())
    print("理论值：",Utils.rounList(a))
    y=paras.f(a)
    noise=torch.randn(y.shape)*noise
    noise=noise.detach().numpy()
    y+=noise
    plt.plot(x,y,'.')
    z=np.hstack((paras.x.reshape(-1,1),y.reshape(-1,1)))
    z=Utils.dataPreTreatment(z)
    z=Utils.rounList(z)
    z=torch.tensor(z,dtype=torch.float32)
    z=trainModel.model(z)
    z=z.detach().numpy()
    z=Utils.unnormalizeLabel(z)
    print("模型计算值为:",z)
    data1=Utils.getPointsFromParas(a)
    data2=Utils.getPointsFromParas(z)
    plt.plot(x,data1,'.')
    plt.plot(x,data2,'.')
    r,p=pearsonr(data1,data2)
    R=round(r**2,4)
    p=Utils.getAverageError(a,z)
    print("平均相对误差为:{}%".format(round(p*100,2)))
    print("R²:",R)
    mark=(1-p)*R*100 # 评分
    print("mark:",mark)
    plt.title("R²:{},mark:{}".format(R,round(mark)),color='red')
    plt.show()
    print("======================")
    time.sleep(3)
