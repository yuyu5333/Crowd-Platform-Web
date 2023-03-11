from re import X
import numpy as np
import xlrd
import copy
import sys
# 准备node结点的数据
# knnsearch
minm = 0
maxm = 0
mine = 0
maxe = 0.41
mintime = 1.6
maxtime = 8
# sys.path.append('./alexnet.xlsx')

class GADS:
    def __init__(self):
        self.id = '' 
        self.m1=0.
        self.m2 = 0.
        self.e1 = 0.
        self.e2 = 0.
        self.tee = 0.
        self.tran = 0.
        self.time = 0.
        self.energy=0.
        self.memory=0.
        self.connect = []

# AlexNet共有26种分割状态
# node = [GADS()] * 26
# NUM = 26
node = []

def readnodedata(NUM ,file_name):
    for i in range(0, NUM):
        file = xlrd.open_workbook(file_name)  # 打开文件
        #file = openpyxl.load_workbook('example.xlsx')
        sheet = file.sheet_by_name("Sheet1")
        no = GADS()
        no.id = i  # 从0开始的id
        no.m1 = sheet.cell_value(i + 2, 7)
        no.m2 = sheet.cell_value(i + 2, 8)
        no.e1 = sheet.cell_value(i + 2, 9)
        no.e2 = sheet.cell_value(i + 2, 10)
        no.tee = sheet.cell_value(i + 2, 11)
        no.tran = sheet.cell_value(i + 2, 12)
        if i == 0:
            no.connect = [NUM-1, i + 1]
        elif i == NUM-1:
            no.connect = [i-1, 0]
        else:
            no.connect = [i-1, i+1]
        # 测试print(no.tee)
        node.append(no)
    #print("网络各结点指标构建结束")
    return node


def min_maxnormalization(node):
    for no in node:
        e1 = (no.e1 - mine) / (maxe - mine)  # e1的归一化
        no.e1 = e1
        e2 = (no.e2 - mine) / (maxe - mine)  # e2的归一化
        no.e2 = e2
        time = (no.time - mintime) / (maxtime - mintime)  # time的归一化
        no.time = time
    return node  # 返回归一化之后的结点


def ensuresubset(nowid, step, NUM):
    nowid = int(nowid)
    step = int(step)
    nodesubset = []
    flag = [0] * NUM
    nodesubset.append(node[nowid])  # 子集中首先是自己
    flag[nowid] = 1  # 在子集中的标1
    for i in range(step):
        tempnodesubset = copy.deepcopy(nodesubset)
        for no in nodesubset:
            for j in range(len(no.connect)):
                # print(no.connect[j])
                if flag[no.connect[j]] != 1:
                    tempnodesubset.append(node[no.connect[j]])
                    flag[no.connect[j]] = 1  # 并标记该结点已经进入子集了
        nodesubset = tempnodesubset
    return nodesubset


def upgradetime(nodesubset, bandwith):
    tempnodesubset = copy.deepcopy(nodesubset)
    for no in tempnodesubset:
        no.time = no.tee + no.tran * (1/bandwith)
        no.energy=no.e1 +no.e2
        no.memory=no.m1 + no.m2
        # 总时延是边缘计算时间+移动端计算时间+传输时延*带宽
    #  已经正则化之后了
    tempnodesubset = min_maxnormalization(tempnodesubset)
    return tempnodesubset


def creatdataset(NUM, file_name):
    readnodedata(NUM, file_name)
    # 测试print(node[15].m2)
    # 确定待搜索子集
    #index = input("当前的分割状态id为：")
    #s = input("搜索子集的步幅s：")
    index = 0
    s = 16
    nodesubset = ensuresubset(index, s, NUM)
    #  加带宽
    #bandwith = float(input("当前的带宽为（MB/s）："))
    bandwith = 1
    #  搜索子集是更新带宽之后的
    searchsubset = upgradetime(nodesubset, bandwith)
    #print("待搜索子集及其指标为：")
    #for no in searchsubset:
    #    print(no.id, no.m1, no.m2, no.e1, no.e2, no.time)
    return searchsubset


def creatdataset_knn(NUM, file_name):
    readnodedata(NUM ,file_name)
    #  加带宽 knn的baseline就是为了表示找到最近邻是可行的
    bandwith = float(input("当前的带宽为（MB/s）："))
    #  搜索子集是更新带宽之后的
    searchsubset = upgradetime(node, bandwith)
    #print("待搜索子集及其指标为：")
   # for no in searchsubset:
    #    print(no.id, no.m1, no.m2, no.e1, no.e2, no.time)
    return searchsubset

def optimal(parament):
    num = 26
    readnodedata(num, "hmt/views/alexnet.xlsx")#路径一定要从app名字开始
    # 测试print(node[15].m2)
    # 确定待搜索子集
    # index = input("当前的分割状态id为：")
    # s = input("搜索子集的步幅s：")
    index = 0
    s = 1
    nodesubset = ensuresubset(index, s, num)
    #  加带宽
    #  搜索子集是更新带宽之后的
    searchsubset = upgradetime(nodesubset, 1)
    #print("待搜索子集及其指标为：")
    num1=num2=num3=0
    latancy=searchsubset[0].time
    energy=searchsubset[0].energy
    memory=searchsubset[0].memory
    for no in searchsubset:
        if latancy<no.time:
            latancy=no.time
            num1=no.id
        if energy<no.energy:
            energy=no.energy
            num2=no.id
        if memory<no.memory:
            memory=no.memory
            num3=no.id
    if parament=='energy':
        return(num2,energy)
    if parament=='latency':
        return(num1,latancy)
    if parament=='memory':
        return(num3,memory)   



# if __name__ == "__main__":
    # # 准备网络的node节点数
    # num = 26
    # readnodedata(num, "alexnet.xlsx")
    # # 测试print(node[15].m2)
    # # 确定待搜索子集
    # index = input("当前的分割状态id为：")
    # s = input("搜索子集的步幅s：")
    # nodesubset = ensuresubset(index, s, num)
    # #  加带宽
    # bandwith = float(input("当前的带宽为（MB/s）："))
    # #  搜索子集是更新带宽之后的
    # searchsubset = upgradetime(nodesubset, bandwith)
    # #print("待搜索子集及其指标为：")
    # for no in searchsubset:
    #     print(no.id, no.m1, no.m2, no.e1, no.e2, no.time)
    # x=optimal('latency')
    # print()