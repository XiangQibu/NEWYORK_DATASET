import numpy as np

filename = './USA-road-d.NY.co/USA-road-d.NY.co' # txt文件和当前脚本在同一目录下，所以不用写具体路径
Efield = []
x_list = []
y_list = []
with open(filename, 'r') as file_to_read:
    count = 0
    while True:
        item = []
        lines = file_to_read.readline() # 整行读取数据
        if not lines:
            break
        v, id, x_loc, y_loc = [i for i in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
        item.append(int(id))  # 添加新读取的数据
        item.append(int(x_loc))
        item.append(int(y_loc))
        Efield.append(item)
        x_list.append(int(x_loc))
        y_list.append(int(y_loc))
        count += 1
"""         if count % 1000 == 0:
            print("count:",count)
            print("item:",item) """
city_show_num = 264346
x_list = np.array(x_list)[0:city_show_num]
y_list = np.array(y_list)[0:city_show_num]

city_co = np.array(Efield)
print(city_co)
print(len(x_list))


filename = './USA-road-d.NY.gr/USA-road-d.NY.gr' # txt文件和当前脚本在同一目录下，所以不用写具体路径
Efield = []
with open(filename, 'r') as file_to_read:
    count = 0
    while True:
        item = []
        lines = file_to_read.readline() # 整行读取数据
        if not lines:
            break
        a, v_i, v_j, dis_ij = [i for i in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
        item.append(int(v_i))  # 添加新读取的数据
        item.append(int(v_j))
        item.append(int(dis_ij))
        Efield.append(item)
        count += 1
"""         if count % 1000 == 0:
            print("count:",count)
            print("item:",item)  """

dis_table = np.array(Efield)
# print(dis_table)
# -*- coding: utf-8 -*-
import random
import copy
import time
import numpy as np
import sys
import math
import tkinter #//GUI模块
import threading
import pandas as pd
from functools import reduce
from adjacency_list import Vertex,Graph

# 参数
'''
ALPHA:信息启发因子，值越大，则蚂蚁选择之前走过的路径可能性就越大
      ，值越小，则蚁群搜索范围就会减少，容易陷入局部最优
BETA:Beta值越大，蚁群越就容易选择局部较短路径，这时算法收敛速度会
     加快，但是随机性不高，容易得到局部的相对最优
'''
(ALPHA, BETA, RHO, Q) = (1.0,1.0,0.3,1000)
# 城市数，蚁群
(city_num, ant_num) = (264346,100)
distance_adja_graph = Graph()
pheromone_adja_graph = Graph()
for i in range(city_num):
    distance_adja_graph.addVertex(i)
    pheromone_adja_graph.addVertex(i)
def initial_map():
        # 初始化邻接表
    distance_adja_graph.clear()
    pheromone_adja_graph.clear()
    initial_pheromone = 1
    for item in dis_table:
        #if item[0] <= 64346 and item[1] <= 64346:
        distance_adja_graph.addEdge(item[0],item[1],item[2])
        pheromone_adja_graph.addEdge(item[0],item[1],initial_pheromone)


initial_map()
# 创建度的字典
degree_table = {}
for i in range(city_num):
    # 邻居节点列表
    node_nbr = list(distance_adja_graph.vertList[i+1].connectedTo)
    degree_table[i+1] = len(node_nbr)

# 第一波筛选，选出度最大的5000个节点
pivot_list = []
dic1SortList = sorted(degree_table.items(),key = lambda x:x[1],reverse = True)
i = 8
j = 0
while i > 4:
    i = dic1SortList[j][1]
    j += 1
print(j)
pivot_list = dic1SortList[:1291]
print(pivot_list[:100])

# 第二波筛选，选出平衡值最小的2500个点
balance_value_dic = {}
for item in pivot_list:
    balance_value_dic[item[0]] = distance_adja_graph.vertList[item[0]].get_balance_value()
# print(balance_value_dic)
pivot_list2 = []
dic2SortList = sorted(balance_value_dic.items(),key = lambda x:x[1],reverse = False)
pivot_list2 = dic2SortList[:1291]

print(pivot_list2[:50])

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 20))

X = []
Y = []
for item in pivot_list2:
    x = city_co[item[0],1]
    y = city_co[item[0],2]
    X.append(x)
    Y.append(y)
    # plt.scatter(x,y,s=1,c='r')
# plt.savefig("map.png")
plt.scatter(x_list,y_list,s=0.5)
plt.scatter(X,Y,s=0.5,c='r')
plt.show()


""" import matplotlib.pyplot as plt
plt.figure(figsize=(20, 20))
plt.scatter(x_list,y_list,s=0.5) """
""" for edge in dis_table[:1000]:
    x_1 = city_co[edge[0],1]
    x_2 = city_co[edge[1],1]
    X = [x_1, x_2]
    y_1 = city_co[edge[0],2]
    y_2 = city_co[edge[1],2]
    Y = [y_1, y_2]
    plt.plot(X, Y, color = 'r', linewidth = 0.1) """
# plt.savefig("map.png")
""" start = 196
target = 1045
x_start = city_co[start,1]
x_target = city_co[target,1]
X = [x_start, x_target]
y_start = city_co[start,2]
y_target = city_co[target,2]
Y = [y_start, y_target]
plt.scatter(X,Y,c = 'r', s=2)
plt.show() """
