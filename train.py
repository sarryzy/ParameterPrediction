import csv
import math
import os

from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torch import nn
import torch
import matplotlib.pyplot as plt
from typing import List
import math

import Utils
from MyClass import MetaData

paras=MetaData()

class MyData(Dataset):
    def __init__(self):
        self.root_dir=paras.filePath
        self.csvList=os.listdir(self.root_dir)
        # print(self.csvList)

    def __getitem__(self, idx):
        csvname = self.csvList[idx]
        csvpath = os.path.join(self.root_dir, csvname)
        data = []
        label = []
        with open(csvpath, 'r') as f:
            k = paras.start + paras.step
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 1:
                    label.append(float(row[0]))
                elif len(row) == 2 and abs(float(row[0])-k) <=0.5 and len(data)<paras.sampling:
                    k += paras.step*(paras.n//paras.sampling)
                    data.append(float(row[1]))
        data = torch.tensor(data, dtype=torch.float32)
        label = Utils.normalizeLabel(label)
        label = torch.tensor(label, dtype=torch.float32)
        return data, label

    def __len__(self):
        return len(self.csvList)

class MyModel(nn.Module):
    def __init__(self,inputSize,outputSize,hidden=2):
        super(MyModel,self).__init__()
        self.layer1=nn.Linear(inputSize,100)
        modules=[]
        for i in range(hidden):
            modules.append(nn.Linear(100,100))
        self.model=nn.Sequential(
            self.layer1,
            *modules,
            nn.Linear(100,outputSize)
        )


    def forward(self,x):
        return self.model(x)

if __name__=="__main__":
    device = torch.device("cpu")
    idx = 100
    dataset = MyData()
    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)
    trainModel = MyModel(paras.sampling,paras.parametersSize)
    trainModel = trainModel.to(device)
    print(trainModel)
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)
    opt = torch.optim.Adam(trainModel.parameters())
    pre_loss = 1000000
    for i in range(10000):
        print("正在迭代第{}次".format(i))
        trainModel.train()
        maxloss = 0
        input, output = 0,0
        for data in dataloader:
            input, output = data
            input = input.to(device)
            output = output.to(device)
            trainOutput = trainModel.model(input)
            loss = loss_fn(output, trainOutput)
            maxloss = max(maxloss, loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        trainModel.eval()
        now_loss = maxloss
        data = Utils.getPoints(idx)
        data = torch.tensor(data, dtype=torch.float32).to(device)
        data = trainModel.model(data)
        data = data.detach().numpy()
        data = Utils.unnormalizeLabel(data) # 此即为计算出来的参数数据
        print(now_loss)
        print("计算得到的参数为",data)
        output=Utils.getLabel(idx)
        print("实际参数为:",output)
        if now_loss < pre_loss:
            pre_loss = now_loss
            Utils.printGraph(idx, data)
            plt.show()
            torch.save(trainModel, paras.savePthPath)
            if pre_loss<1 and i>=10:
                print("训练已达到要求")
                break