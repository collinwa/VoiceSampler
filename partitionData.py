import numpy as np 
# Code adapted from https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
MAX = 1000000

class Graph(): 
  
    def __init__(self, vertices): 
        self.V = vertices 
        self.graph = [[0 for column in range(vertices)]  
                    for row in range(vertices)] 
  
    # A utility function to print the constructed MST stored in parent[] 
    def printMST(self, parent): 
        print("Edge \tWeight")
        for i in range(1, self.V): 
            print(parent[i], "-", i, "\t", self.graph[i][ parent[i] ])
    # A utility function to find the vertex with  
    # minimum distance value, from the set of vertices  
    # not yet included in shortest path tree 
    def minKey(self, key, mstSet): 
        min = MAX
  
        for v in range(self.V): 
            if key[v] < min and mstSet[v] == False: 
                min = key[v] 
                min_index = v 
  
        return min_index 
  
    # Function to construct and print MST for a graph  
    # represented using adjacency matrix representation 
    def primMST(self, proportionSplit): 
        key = [MAX] * self.V 
        parent = [None] * self.V
        key[0] = 0 
        mstSet = [False] * self.V 
  
        parent[0] = -1
        for cout in range(int(self.V * proportionSplit)): 
  
            # pick the minimum distance vertex from the set of vertices not yet processed 
            u = self.minKey(key, mstSet) 
            # mark selected vertex as visited
            mstSet[u] = True
  
            # updated selected vertex's neighbors
            for v in range(self.V): 
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]: 
                        key[v] = self.graph[u][v] 
                        parent[v] = u 
             
        ret = []
        for i in range(self.V):
          if mstSet[i]:
            ret.append((parent[i], i, self.graph[i][parent[i]]))
        return ret

# Accepts: proportionSplit is amount of data partitioned into test set
# Returns: list of tuples, each tuple in the form (parent, child, length) specifying a single edge
def splitTestTrainData(adjMat, proportionSplit=0.1):
  assert type(adjMat) == np.ndarray
  n = len(adjMat)
  g = Graph(n) 
  g.graph = adjMat 
  return g.primMST(proportionSplit)

# Tester code
data = np.random.rand(100, 100)
np.fill_diagonal(data, 0)
print(splitTestTrainData(data))

