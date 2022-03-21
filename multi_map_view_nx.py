# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import time
import sys
from classAnt import *


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

def read_subgraph():
    # 读取子图分割信息
    filename = './subgraph_list.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
    Efield = []
    with open(filename, 'r') as file_to_read:
        count = 0
        while True:
            item = []
            lines = file_to_read.readline() # 整行读取数据
            if not lines:
                break
            # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            for i in lines.split():
                item.append(int(i)) 
            Efield.append(item)
            count += 1
    """         if count % 1000 == 0:
                print("count:",count)
                print("item:",item)  """
    subgraph_list = np.array(Efield)
    return subgraph_list

def read_pivot():
    # 读取高一级图，枢纽点节点
    filename = './pivot_list.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
    Efield = []
    with open(filename, 'r') as file_to_read:
        count = 0
        while True:
            lines = file_to_read.readline() # 整行读取数据
            if not lines:
                break
            # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            pivot_point, b = lines.split()
            Efield.append(int(pivot_point))
            count += 1
    """         if count % 1000 == 0:
                print("count:",count)
                print("item:",item)  """
    pivot_list = np.array(Efield)
    return pivot_list

def read_pivotedge():
    # 读取高一级图，枢纽点之间的连边，以及最短路径
    filename = './pivot_edge_list.txt' # txt文件和当前脚本在同一目录下，所以不用写具体路径
    Efield = []
    with open(filename, 'r') as file_to_read:
        count = 0
        while True:
            item = []
            lines = file_to_read.readline() # 整行读取数据
            if not lines:
                break
            # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
            for i in lines.split():
                item.append(int(i)) 
            Efield.append(item)
            count += 1
    """         if count % 1000 == 0:
                print("count:",count)
                print("item:",item)  """
    pivot_edge_list = np.array(Efield)
    return pivot_edge_list





def update_pheromone_gragh(ants,target):
    # 获取每只蚂蚁在其路径上留下的信息素
    for ant in ants:
        #如果这条路无法到达，则不应该更新信息素
        if ant.path[-1] != target:
            continue
        for i in range(1,len(ant.path)):
            start, end = ant.path[i-1], ant.path[i]
            # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
            #temp_pheromone[start][end] += Q / ant.total_distance
            # temp_vertex = pheromone_adja_graph.vertList[start]
            temp_pheromone = pheromone_nx_graph.get_edge_data(start, end)['weight'] * (1-RHO)
            # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
            temp_pheromone += Q / ant.total_distance
            pheromone_nx_graph.add_weighted_edges_from([(start,end,temp_pheromone)])

def update_pivot_pheromone_gragh(ants,target):
    # 获取每只蚂蚁在其路径上留下的信息素
    for ant in ants:
        #如果这条路无法到达，则不应该更新信息素
        if ant.path[-1] != target:
            continue
        for i in range(1,len(ant.path)):
            start, end = ant.path[i-1], ant.path[i]
            # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
            #temp_pheromone[start][end] += Q / ant.total_distance
            # temp_vertex = pheromone_adja_graph.vertList[start]
            temp_pheromone = pivot_pheromone_nx_graph.get_edge_data(start, end)['weight'] * (1-RHO)
            # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
            temp_pheromone += Q / ant.total_distance
            pivot_pheromone_nx_graph.add_weighted_edges_from([(start,end,temp_pheromone)])

def initial_map(city_num, dis_table):
        # 初始化邻接表
    distance_nx_graph.clear()
    pheromone_nx_graph.clear()
    initial_pheromone = 1
    for item in dis_table:
        if item[0] <= city_num and item[1] <= city_num:
            distance_nx_graph.add_weighted_edges_from([(item[0],item[1],item[2])])
            pheromone_nx_graph.add_weighted_edges_from([(item[0],item[1],initial_pheromone)])
            
def initial_pivot_map(pivot_edge_list):
        # 初始化邻接表
    pivot_distance_nx_graph.clear()
    pivot_pheromone_nx_graph.clear()
    initial_pheromone = 1
    for index in range(len(pivot_edge_list)):
        #if pivot_edge_list[index,0] <= city_num and dis_table[index, 1] <= city_num:
        pivot_distance_nx_graph.add_weighted_edges_from([(pivot_edge_list[index][0],pivot_edge_list[index][1],pivot_edge_list[index][2])])
        pivot_pheromone_nx_graph.add_weighted_edges_from([(pivot_edge_list[index][0],pivot_edge_list[index][1],initial_pheromone)]) 

def initial_ants(start, target, city_N):
    global ants
    ants = [Ant(ID,target,start,city_N,ALPHA,BETA,RHO,Q) for ID in range(ant_num)]  # 初始蚁群



def search_pivot_path(start,target,city_co):
    
    
    #best_ant = Ant(-1,target,start)                          # 初始最优解
    #best_ant.total_distance = np.inf           # 初始最好的蚂蚁
    best_distance = np.inf
    
    iter = 1

    while True:
        # 遍历每一只蚂蚁
        tic = time.time()
        count = 0
        distance_list = []
        for ant in ants:
            # 搜索一条路径
            ant.search_path(pivot_distance_nx_graph,pivot_pheromone_nx_graph,city_co)
            # 与当前最优蚂蚁比较
            #print(ant.path)
            #print(ant.total_distance)
            distance_list.append(ant.total_distance)
            if ant.total_distance < best_distance and ant.path[-1] == target:
                # 更新最优解
                #best_ant = copy.deepcopy(ant)
                best_path = ant.path
                best_distance = ant.total_distance
                print('found a path!')

        # 更新信息素
        update_pivot_pheromone_gragh(ants,target)
        if best_distance < np.inf:
            toc = time.time()
            gap = toc-tic
            path_str = '%d' % best_path[0]
            # 给定策略然后进行选择
            for i in range(len(best_path) - 1):
                path_str = path_str + '------' + '%d' % best_path[i+1]
            print (u"迭代次数：",iter,u"最佳路径总距离：",int(best_distance),u"平均路径总距离：",np.mean(distance_list),"路径为：",path_str,"搜索时长：",gap)
        else:
            print(u"迭代次数：",iter,u"无法到达！")
        iter += 1

        if iter == 2:
            return int(best_distance), np.mean(distance_list), gap

def search_path(start,target,city_co):
    
    
    #best_ant = Ant(-1,target,start)                          # 初始最优解
    #best_ant.total_distance = np.inf           # 初始最好的蚂蚁
    best_distance = np.inf
    
    iter = 1

    while True:
        # 遍历每一只蚂蚁
        tic = time.time()
        count = 0
        distance_list = []
        for ant in ants:
            # 搜索一条路径
            ant.search_path(distance_nx_graph,pheromone_nx_graph,city_co)
            # 与当前最优蚂蚁比较
            count += 1
            print("ant:", count)
            #print(ant.path)
            #print(ant.total_distance)
            distance_list.append(ant.total_distance)
            if ant.total_distance < best_distance and ant.path[-1] == target:
                # 更新最优解
                #best_ant = copy.deepcopy(ant)
                best_path = ant.path
                best_distance = ant.total_distance
                print('found a path!')

        # 更新信息素
        update_pheromone_gragh(ants,target)
        if best_distance < np.inf:
            toc = time.time()
            gap = toc-tic
            path_str = '%d' % best_path[0]
            # 给定策略然后进行选择
            for i in range(len(best_path) - 1):
                path_str = path_str + '------' + '%d' % best_path[i+1]
            print (u"迭代次数：",iter,u"最佳路径总距离：",int(best_distance),u"平均路径总距离：",np.mean(distance_list),"路径为：",path_str,"搜索时长：",gap)
        else:
            print(u"迭代次数：",iter,u"无法到达！")
        iter += 1
        if iter == 2:
            return int(best_distance), np.mean(distance_list), gap




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

# 分层下的新图
# 分层下进行的新的蚁群算法，其实本质上就是换了一下图
(pivot_city_num, ant_num) = (1291,30)
pivot_distance_nx_graph = nx.DiGraph()
pivot_pheromone_nx_graph = nx.DiGraph()



if __name__ == '__main__':
    # 设置起点和终点
    start = int(sys.argv[1])
    target = int(sys.argv[2])
    # start = 2961
    # target = 145210

    city_co = read_co()
    dis_table = read_gr()
    subgraph_list = read_subgraph()
    pivot_list = read_pivot()
    pivot_edge_list = read_pivotedge()

    start_pivot = subgraph_list[start, 1]
    target_pivot = subgraph_list[target, 1]
    print("start_pivot:", start_pivot,"start_pivot:", target_pivot)


    initial_map(city_num, dis_table)
    #设置广播变量
    initial_pivot_map(pivot_edge_list)
    # 起点到枢纽点1
    initial_ants(start,start_pivot,city_num)
    D_s1, meanD_s1, gap_s1 = search_path(start, start_pivot, city_co)
    # 枢纽点1到枢纽点2
    initial_ants(start_pivot,target_pivot,city_num)
    D_12, meanD_12, gap_12 = search_pivot_path(start_pivot, target_pivot, city_co)
    # 枢纽点2到终点
    initial_ants(target_pivot,target,city_num)
    D_2t, meanD_2t, gap_2t = search_path(target_pivot, target, city_co)

    print("best_distance:", D_s1 + D_12 + D_2t)
    print("mean_distance:", meanD_s1 + meanD_12 + meanD_2t)
    print("time cost:", gap_s1 + gap_12 + gap_2t)


