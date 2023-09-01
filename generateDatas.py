import csv
import shutil
import os
import numpy as np

from MyClass import MetaData


# y=a0/(1+(a1/x)^a2)/(1+(x/a3)^a4)
# a0 (200,500),a1(5,7),a2(1.7,1.9),a3(140,150),a4(10,30)
paras=MetaData()

def generateData(limit,x,fileName,dataSum=10000):
    logName=fileName
    if os.path.exists(logName):shutil.rmtree(logName)
    os.mkdir(logName)
    n=len(limit)
    for i in range(dataSum):
        a=np.random.rand(n)*np.random.choice([-1,1])
        for j in range(n):
            rang=limit[j]
            a[j]=(rang[0]+rang[1])/2+(rang[1]-rang[0])/2*a[j]
        a=list(a)
        fileName = logName+"/csvName_{}.csv".format(i)
        if i and i%1000==0:print("已生成{}个数据".format(i))
        with open(fileName,'w',newline="") as fi:
            writer=csv.writer(fi)
            # print("a:",a)
            for j in range(n):
                a[j]=round(a[j],2)
                writer.writerow([str(a[j])])
            y=paras.f(a)
            for j in range(len(x)):
                x[j]=round(x[j],2)
                y[j]=round(y[j],2)
                writer.writerow([x[j],y[j]])






