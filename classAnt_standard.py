import numpy as np
import random

class Ant(object):

    # 初始化
    def __init__(self,ID,target,start,city_num,ALPHA,BETA,RHO,Q):

        self.ID = ID                 # ID
        self.target = target
        self.start = start
        self.city_num = city_num
        self.ALPHA = ALPHA
        self.BETA = BETA
        self.RHO = RHO
        self.Q =Q
        self.__clean_data()          # 随机初始化出生点

    # 初始数据
    def __clean_data(self):

        self.path = []               # 当前蚂蚁的路径           
        self.total_distance = 0.0    # 当前路径的总距离
        self.move_count = 0          # 移动次数
        self.current_city = -1       # 当前停留的城市
        self.open_table_city = np.ones(self.city_num) # 探索城市的状态

        #city_index = random.randint(0,city_num-1) # 随机初始出生点
        city_index = self.start
        self.current_city = city_index
        # self.current_city_node = distance_adja_graph.vertList[self.current_city]
        #print('city:',self.current_city_node.getConnections())
        # self.current_pheromone_node = pheromone_adja_graph.vertList[self.current_city]
        #print('pheromone:',self.current_pheromone_node.getConnections())
        self.path.append(city_index)
        self.open_table_city[city_index] = 0
        self.move_count = 1

    # 选择下一个城市
    def __choice_next_city(self, dis_nx_graph, phero_nx_graph, city_co):

        next_city = -1
        select_citys_prob = {}  #存储去下个城市的概率
        total_prob = 0.0
        #print('cityyyyyyy:',self.current_city_node.getConnections())
        for i in dis_nx_graph.succ[self.current_city]:
            if self.open_table_city[i]:
                # weight = int((abs(city_co[i,1] - city_co[self.target, 1]) + abs(city_co[i,2] - city_co[self.target, 2])) / 10)
                weight = dis_nx_graph.get_edge_data(self.current_city, i)['weight']
                pheromone = phero_nx_graph.get_edge_data(self.current_city, i)['weight']
                
                select_citys_prob[i] = pow(pheromone, self.ALPHA) * pow(1.0/(weight+1), self.BETA)
                # 在城市的选择概率上，如果只与信息素有关，是否就一定程度上相当是于随机选择入手，然后会初始生成部分不太优的路径，然后根据评价指标评价后，得到不同的信息素分布。
                # select_citys_prob[i.getId()] = pow(pheromone, ALPHA)
                total_prob += select_citys_prob[i]
        # 获取去下一个城市的概率
        """ for i in range(city_num):
            if self.open_table_city[i]:
                try :
                    # 计算概率：与信息素浓度成正比，与距离成反比
                    if distance_graph[self.current_city][i] < np.inf:
                        select_citys_prob[i] = pow(pheromone_graph[self.current_city][i], ALPHA) * pow((1.0/distance_graph[self.current_city][i]), BETA)
                        total_prob += select_citys_prob[i]
                    else:
                        select_citys_prob[i] = 0

                except ZeroDivisionError as e:
                    print ('Ant ID: {ID}, current city: {current}, target city: {target}'.format(ID = self.ID, current = self.current_city, target = i))
                    sys.exit(1) """

        # 轮盘选择城市
        if total_prob > 0.0:
            # 产生一个随机概率,0.0-total_prob
            temp_prob = random.uniform(0.0, total_prob)
            for ID in dis_nx_graph.succ[self.current_city]:
                if self.open_table_city[ID] and select_citys_prob[ID] != 0:
                    # 轮次相减
                    temp_prob -= select_citys_prob[ID]
                    if temp_prob <= 0.0:
                        next_city = ID
                        break


        """ if (next_city == -1):
            next_city = random.randint(0, city_num - 1)
            while ((self.open_table_city[next_city]) == False):  # if==False,说明已经遍历过了
                next_city = random.randint(0, city_num - 1)  """

        # 返回下一个城市序号
        return next_city

    """ # 计算路径总距离
    def __cal_total_distance(self):

        temp_distance = 0.0

        for i in range(1, len(self.path)):
            start, end = self.path[i-1], self.path[i]
            temp_distance += distance_graph[start][end]

        # 回路
        #end = self.path[0]
        #temp_distance += distance_graph[start][end]
        self.total_distance = temp_distance """


    # 移动操作
    def __move(self, next_city, dis_nx_graph):

        self.path.append(next_city)
        self.open_table_city[next_city] = 0
        # print(dis_nx_graph.get_edge_data(self.current_city, next_city)['weight'])
        self.total_distance += dis_nx_graph.get_edge_data(self.current_city, next_city)['weight']
        self.current_city = next_city
        # self.current_city_node = distance_adja_graph.vertList[self.current_city]
        # self.current_pheromone_node = pheromone_nx_graph.vertList[self.current_city]
        self.move_count += 1

    # 搜索路径
    def search_path(self, dis_nx_graph, phero_nx_graph, city_co):

        # 初始化数据
        self.__clean_data()

        # 搜素路径，遍历完所有城市为止
        # while self.move_count < self.city_num:
        while True:
            # 移动到下一个城市
            next_city =  self.__choice_next_city(dis_nx_graph, phero_nx_graph, city_co)
            # print(next_city)
            if next_city == -1:
                self.__clean_data()
                continue
                # return np.inf, self.path
            self.__move(next_city, dis_nx_graph)
            #print("move!")
            if next_city == self.target:
                return self.total_distance, self.path
        # return np.inf, self.path

        # 计算路径总长度
        # self.__cal_total_distance()