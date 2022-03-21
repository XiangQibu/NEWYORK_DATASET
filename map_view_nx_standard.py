# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import time
import sys

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




from classAnt_standard import *

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

def initial_map(city_num, dis_table):
        # 初始化邻接表
    distance_nx_graph.clear()
    pheromone_nx_graph.clear()
    initial_pheromone = 1
    for item in dis_table:
        if item[0] <= city_num and item[1] <= city_num:
            distance_nx_graph.add_weighted_edges_from([(item[0],item[1],item[2])])
            pheromone_nx_graph.add_weighted_edges_from([(item[0],item[1],initial_pheromone)])

def initial_ants(start, target):
    global ants
    ants = [Ant(ID,target,start, city_num, ALPHA, BETA, RHO, Q) for ID in range(ant_num)]  # 初始蚁群

def search_path(start,target,city_co):
    
    
    # best_ant = Ant(-1,target,start)                          # 初始最优解
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
            print("ant:", count)
            ant.search_path(distance_nx_graph, pheromone_nx_graph, city_co)
            # 与当前最优蚂蚁比较
            #print(ant.path)
            #print(ant.total_distance)
            count += 1
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
                """ if strategy_graph[best_path[i]][best_path[i+1]] == 1:
                    path_str = path_str + '--freeway--'
                elif strategy_graph[best_path[i]][best_path[i+1]] == 2:
                    path_str = path_str + '--road--' """
                path_str = path_str + '------' + '%d' % best_path[i+1]
            # 给定策略的概率，按概率进行选择?这样似乎是不太合理的。因为只有策略确定了之后，才有相应的评价体系，先后顺序不能乱
            """ for i in range(len(best_path) - 1):
                if prob_graph[best_path[i]][best_path[i+1]] >= 0:
                    # 二项分布随机数
                    the_choice = np.random.binomial(1,prob_graph[best_path[i]][best_path[i+1]],size = 1)
                    print(prob_graph[best_path[i]][best_path[i+1]])
                    if the_choice == 1:
                        path_str = path_str + '--freeway--'
                    elif the_choice == 0:
                        path_str = path_str + '--road--'
                    path_str = path_str + '%d' % best_path[i+1] """
            print (u"迭代次数：",iter,u"最佳路径总距离：",int(best_distance),u"平均路径总距离：",np.mean(distance_list),"路径为：",path_str,"搜索时长：",gap)
        else:
            print(u"迭代次数：",iter,u"无法到达！")
        iter += 1
        if iter > 1:
            break

# 参数
'''
ALPHA:信息启发因子，值越大，则蚂蚁选择之前走过的路径可能性就越大
    ，值越小，则蚁群搜索范围就会减少，容易陷入局部最优
BETA:Beta值越大，蚁群越就容易选择局部较短路径，这时算法收敛速度会
    加快，但是随机性不高，容易得到局部的相对最优
'''
(ALPHA, BETA, RHO, Q) = (1.0,2.0,0.005 ,5000)
# 城市数，蚁群
(city_num, ant_num) = (264346, 30)
distance_nx_graph = nx.DiGraph()
pheromone_nx_graph = nx.DiGraph()

if __name__ == '__main__':
    # 设置起点和终点
    start = int(sys.argv[1])
    target = int(sys.argv[2])
    # start = 2000
    # target = 2020

    city_co = read_co()
    dis_table = read_gr()

    initial_map(city_num, dis_table)
    initial_ants(start,target)
    search_path(start, target, city_co)