import csv
import math
import os
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torch import nn
import torch
import matplotlib.pyplot as plt
from typing import List
import math


class MetaData:
    def __init__(self):
        self.dataSum=10000
        self.start = -10
        self.end = 10
        self.step = 0.1
        if self.step<1:
            self.x=np.arange(self.start*1/self.step,self.end*1/self.step,1)*self.step
        else: self.x=np.arange(self.start,self.end,self.step)
        self.n=len(self.x)
        self.sampling=self.n//10
        self.limit=[[0,1],[1,5],[1,10],[-1,1]]
            #
            # [[0, 10], [1, 5], [1, 10], [1, 5], [1, 10], [1, 5], [1, 10]]
            # [200,500],[5,7],[1.7,1.9],[140,150],[10,30]
        self.parametersSize=len(self.limit)
        self.filePath="datas/file6"
        self.savePthPath="models/pth6.pth"


    def f(self,a):
        return a[0]+a[1]/(a[2]*math.sqrt(math.pi/2))*math.e**(-2*(self.x-a[3])*(self.x-a[3])/(a[2]*a[2]))
        # return (a[0]+a[1]*math.e**(-self.x/a[2])+a[3]*math.e**(-self.x/a[4])+a[5]*math.e**(-self.x/a[6]))
        # return  a[0] / (1 + (a[1] / self.x) ** a[2]) / (1 + (self.x / a[3]) ** a[4])

# paras=MetaData()
# print(paras.sampling)