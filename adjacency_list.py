import numpy as np


class Vertex:
    def __init__(self,key):
        self.id = key
        self.connectedTo = {}
        self.balance_value = 0

    #从这个顶点添加一个连接到另一个
    def addNeighbor(self,nbr,weight = 0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id)

    #返回邻接表中的所有的项点
    def getConnections(self):
        return  self.connectedTo.keys()
    

    def getId(self):
        return self.id

    #返回从这个顶点到作为参数顶点的边的权重
    def getweight(self,nbr):
        return  self.connectedTo[nbr]

    def get_balance_value(self):
        n = len(self.connectedTo)
        self.balance_value = 0
        total_weight = 0
        for w in self.getConnections():
            self.balance_value += self.getweight(w) / len(w.connectedTo)
            total_weight += self.getweight(w)
        self.balance_value = self.balance_value / (n * total_weight)
        return self.balance_value




class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self,key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return  newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return  self.vertList[n]
        else:
            return  None

    def __contains__(self, n):
        return  n in self.vertList

    def addEdge(self,f,t,const = 0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not  in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t],const)
    def clear(self):
        self.vertList.clear()

    def getVertices(self):
        return  self.vertList.keys()

    def __iter__(self):
        return  iter(self.vertList.values())


if __name__ == '__main__':
    city_num = 5
    INF = np.inf
    distance_graph = [[0,8,INF,5,INF],
       [INF,0,3,INF,INF],
       [INF,INF,0,INF,6],
       [INF,INF,9,0,INF],
       [INF,INF,INF,INF,0]]
    g = Graph()
    for i in range(city_num):
        g.addVertex(i)
    print(g.vertList)
    for i in range(city_num):
        for j in range(city_num):
            if distance_graph[i][j] < np.inf:
                g.addEdge(i,j,distance_graph[i][j])

    m = g.vertList[0].get_balance_value()
    print('getConnection:',m)
    for v in g:
        for w in v.getConnections():
            print(type(w))
            print("( %s , %s )" % (v.getId(), w.getId()))
            print(v.getweight(g.vertList[w.getId()]))