# -*- coding: UTF-8 -*-
from pyspark import SparkContext
from pyspark import SparkConf
import random
import copy
import time
import sys
import math
import tkinter #//GUI模块
import threading
import pandas as pd
from functools import reduce

def SetLogger( sc ):
    logger = sc._jvm.org.apache.log4j
    logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
    logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
    logger.LogManager.getRootLogger().setLevel(logger.Level.ERROR)   

def CreateSparkContext():
    sparkConf = SparkConf()                                                       \
                         .setAppName("map_view_nx_spark")                         \
                         .set("spark.ui.showConsoleProgress", "false") \

    sc = SparkContext(conf = sparkConf)
    print("master="+sc.master)
    SetLogger(sc)
    sc.addPyFile("hdfs://master:9000/ant/classAnt.py")
    SparkConf().set("spark.defalut.parallelism", ant_num)
    return (sc)

import numpy as np
import networkx as nx


def read_co():
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

    city_co = np.array(Efield)
    return city_co

def read_gr():
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
    return dis_table

# from classAnt import Ant

def update_pheromone_gragh(ant_searched,target, bc_pheromone_nx_graph_value):
    #path[0]为路径长度，path[1]为路径list
    # 获取每只蚂蚁在其路径上留下的信息素
    for path in ant_searched:
        #如果这条路无法到达，则不应该更新信息素
        if path[1][-1] != target:
            continue
        k = 1 / path[0]
        for i in range(1,len(path[1])):
            start, end = path[1][i-1], path[1][i]
            # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
            #temp_pheromone[start][end] += Q / ant.total_distance
            # temp_vertex = pheromone_adja_graph.vertList[start]
            temp_pheromone = bc_pheromone_nx_graph_value.get_edge_data(start, end)['weight'] * (1-RHO)
            # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
            temp_pheromone += Q * k
            pheromone_nx_graph.add_weighted_edges_from([(start,end,temp_pheromone)])
            
def initial_map(city_num, dis_table):
        # 初始化邻接表
    distance_nx_graph.clear()
    pheromone_nx_graph.clear()
    initial_pheromone = 1
    for item in dis_table:
        if item[0] <= city_num and item[1] <= city_num:
            distance_nx_graph.add_weighted_edges_from([(item[0],item[1],item[2])])
            pheromone_nx_graph.add_weighted_edges_from([(item[0],item[1],initial_pheromone)])
    

def initial_ants(sc, start, target):
    # global ants
    ants = [Ant(ID,target,start,city_num,ALPHA,BETA,RHO,Q) for ID in range(ant_num)]  # 初始蚁群
    ants_RDD = sc.parallelize(ants)
    return(ants_RDD)

from classAnt import *
# 参数
'''
ALPHA:信息启发因子，值越大，则蚂蚁选择之前走过的路径可能性就越大
      ，值越小，则蚁群搜索范围就会减少，容易陷入局部最优
BETA:Beta值越大，蚁群越就容易选择局部较短路径，这时算法收敛速度会
     加快，但是随机性不高，容易得到局部的相对最优
'''
(ALPHA, BETA, RHO, Q) = (1.0,2.0,0.005,5000)
# 城市数，蚁群
(city_num, ant_num) = (264346, 30)
distance_nx_graph = nx.DiGraph()
pheromone_nx_graph = nx.DiGraph()

# 设置起点和终点
start = 2000
target = 145210

if __name__ == '__main__':
    # 设置起点和终点
    #start = int(sys.argv[1])
    #target = int(sys.argv[2])
    start = 2000
    target = 2020
    
    sc=CreateSparkContext()
    print("开始读取数据。。。")
    city_co = read_co()
    dis_table = read_gr()
    print("初始化地图。。。")
    initial_map(city_num, dis_table)
    print("广播参数。。。")
    bc_distance_nx_graph = sc.broadcast(distance_nx_graph)
    bc_pheromone_nx_graph = sc.broadcast(pheromone_nx_graph)
    bc_city_co = sc.broadcast(city_co)
    print("初始化蚁群。。。")
    ants_RDD = initial_ants(sc, start,target).persist()
    print(ants_RDD)
    # 开始搜索
    print("开始搜索。。。")
    the_iter = 1
    best_path_distance = np.inf
    best_path = []
    while(1):
        tic = time.time()
        ants_RDD_searched = ants_RDD.map(lambda x:x.search_path(bc_distance_nx_graph.value, bc_pheromone_nx_graph.value, bc_city_co.value))
        ants_searched = ants_RDD_searched.collect()
        #ants_searched = ants_RDD_searched.show()

        t_collect = time.time()
        print("collect time:", t_collect - tic)

        distance_list = []
        for i in range(ant_num):
            distance_list.append(ants_searched[i][0])
        min_distance = min(distance_list)
        if min_distance < best_path_distance:
            print("found a path!")
            print("distance:", min_distance)
            print(ants_searched[distance_list.index(min_distance)][1])
            best_path_distance = min_distance
            best_path = ants_searched[distance_list.index(min_distance)][1]

        t_findmin =time.time()

        update_pheromone_gragh(ants_searched,target, bc_pheromone_nx_graph.value)
        t_update = time.time()
        print("update time:", t_update - t_findmin)

        bc_pheromone_nx_graph.unpersist()
        bc_pheromone_nx_graph = sc.broadcast(pheromone_nx_graph)
        t_bc = time.time()
        print("bc time:", t_bc - t_update)


        if best_path_distance < np.inf:
            print (u"迭代次数：",the_iter,u"最佳路径总距离：",int(best_path_distance),"路径为：",best_path)
        else:
            print(u"迭代次数：",the_iter,u"无法到达！")

        toc = time.time()
        gap = toc-tic
        print("搜索总时长：",gap)

        the_iter += 1
        if the_iter == 2:
            break

