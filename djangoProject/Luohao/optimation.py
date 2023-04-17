from re import X
import xlrd
import copy
import sys
import numpy as np
# import openpyxl

class GADS:
    def __init__(self):
        self.id = '' 
        self.m1 =0.   
        self.m2 = 0.  
        self.e1 = 0.  
        self.e2 = 0.
        self.msum=0.
        self.esum=0.
        self.tee = 0.
        self.tran = 0
        self.time = 0.
        self.connect = []

def ensuresubset(nowid, step, NUM,node):
    nowid = int(nowid)
    step = int(step)
    nodesubset = []
    flag = [0] * NUM #序列只能与整数相乘，得到一个复制n倍的序列
    nodesubset.append(node[nowid])  # 子集中首先是自己
    flag[nowid] = 1  # 在子集中的标1
    for i in range(step):
        tempnodesubset = copy.deepcopy(nodesubset)#复制一个一模一样的对象，包括对象内部嵌套的子对象
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
        no.time = no.tee + no.tran * (2.0/bandwith)
        # 总时延是边缘计算时间+移动端计算时间+传输时延*带宽
    #  已经正则化之后了
    # tempnodesubset = min_maxnormalization(tempnodesubset)
    return tempnodesubset

def readdata(num,target,file_name):
    # file_name="hmt/views/AlexNet_Cifar.xlsx"
    file = xlrd.open_workbook(file_name)  # 打开文件
    sheet = file.sheet_by_name("Sheet1")
    node=[]
    no_latency=[]
    no_energy=[]
    for i in range(0, num):#AlexNet
        no = GADS()
        no.id = i  # 从0开始的id
        no.m1 = sheet.cell_value(i + 1, 1)
        no.m2 = sheet.cell_value(i + 1, 2)
        no.msum = sheet.cell_value(i + 1, 4)
        no.e1 = sheet.cell_value(i + 1, 6)
        no.e2 = sheet.cell_value(i + 1, 7)
        no.esum = sheet.cell_value(i + 1, 8)
        no.tee = sheet.cell_value(i + 1, 5)
        no.tran = sheet.cell_value(i + 1, 3)
        if i == 0:
            no.connect = [num-1, i + 1]
        elif i == num-1:
            no.connect = [i-1, 0]
        else:
            no.connect = [i-1, i+1]
        node.append(no)
        no_latency.append(no.msum)
        no_energy.append(no.esum)
    min_latency=min(no_latency)
    min_latency_index=no_latency.index(min_latency)
    # print(node[min_latency_index].id, node[min_latency_index].msum,node[min_latency_index].esum)
    min_energy=min(no_energy)
    min_energy_index=no_energy.index(min_energy)
    # print(min_latency)
    # return(min_latency_index,min_latency)
    latency_nodesubset = ensuresubset(min_latency_index, 1, num,node)
    # searchsubset = upgradetime(latency_nodesubset, 2.0)
    if target=='latency':
        return node[min_latency_index]
    else:
        return node[min_energy_index]
    # for no in searchsubset:
        # print(no.id, no.m1, no.m2, no.e1, no.e2, no.time)
    # return searchsubset

# if __name__ == "__main__":
#     s=readdata(26)
#     print("segmentation")
#     print(s.id, s.msum,s.esum)