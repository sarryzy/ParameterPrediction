import csv
import matplotlib.pyplot as plt

from MyClass import MetaData
paras=MetaData()
limit=paras.limit

def getLabel(idx:int):
    # 得到第idx的label
    fileName = paras.filePath+"/csvName_{}.csv".format(idx)
    data = []
    with open(fileName, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 1:
                data.append(float(row[0]))
    return data

def getPoints(idx:int):
    # 选取代表点
    fileName = paras.filePath+"/csvName_{}.csv".format(idx)
    data = []
    with open(fileName, 'r') as f:
        k=paras.start+paras.step
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2 and abs(float(row[0])-k)<=1e-2 and len(data)<paras.sampling:
                k+=paras.step*(paras.n//paras.sampling)
                data.append(float(row[1]))
    return data
def normalizeLabel(label):
    # 将原有的分布转化为-1-1上的分布
    for i in range(len(limit)):
        low,upp=limit[i][0],limit[i][1]
        label[i]=(label[i]-(low+upp)/2)/((upp-low)/2)
    return label

def unnormalizeLabel(label):
    # 将-1-1上的分布转化为原有的分布
    for i in range(len(limit)):
        low,upp=limit[i][0],limit[i][1]
        label[i]=(low+upp)/2+(upp-low)/2*label[i]
    return label

def printGraph(idx,data):
    # 根据idx和得到的系数数据data进行绘图
    a=getLabel(idx)
    b=data
    x=paras.x
    a=paras.f(a)
    b=paras.f(b)
    plt.plot(x,a,'.')
    plt.plot(x,b,'.')

def rounList(lis):
    # 对列表进行四舍五入
    for i in range(len(lis)):
        lis[i]=round(lis[i],2)
    return lis

def dataPreTreatment(lis):
    # 对数据进行前处理
    eps = 1e-2
    k = paras.start+paras.step
    res = []
    n = len(lis)
    for i in range(n):
        x = lis[i][0]
        y = lis[i][1]
        if abs(x - k) < eps and len(res)<paras.sampling:
            res.append(y)
            k += paras.step*(paras.n//paras.sampling)
        # elif x > k > lis[i - 1][0]:
        #     res.append(lis[i - 1][1] + (k - lis[i - 1][0]) / (x - list[i - 1][0]) * (y - list[i - 1][1]))
        #     k += paras.step*(paras.n//paras.sampling)
    if len(res) != paras.sampling:
        raise ValueError("输入数据不符合要求")
    return res

def getPointsFromParas(lis):
    return paras.f(lis)

def grade(a,b):
    r=0
    x = paras.x
    y1=paras.f(a)
    y2=paras.f(b)
    for i in range(len(x)):
        # print("相对误差为:",abs(y1[i]-y2[i])/y1[i])
        if y1[i]==0: continue
        r=max(r,abs(y1[i]-y2[i])/y1[i])
    return r

def getAverageError(a,b):
    # 输入两个数组，得到两个数组的平均误差
    res=0
    for i in range(len(a)):
        if a[i]==0 : continue
        p=abs(a[i]-b[i])/a[i]
        if p<1:res+=p
    return res/len(a)