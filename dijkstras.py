import numpy as np
from scipy.spatial.distance import euclidean
from collections import defaultdict

def neighbours(point,grid):
  y,x = point
  rows,cols = grid.shape
  return [(r,c) for r in range(y-1,y+2) 
                for c in range(x-1,x+2) 
                if  r in range(rows) and
                    c in range(cols) and
                    (r!=y or c!=x)   and
                    grid[r,c]==1] 

def create_graph(grid):
  G = defaultdict(list)
  rows, cols = grid.shape
  for u in [(r,c) for c in range(cols) for r in range(rows) if grid[r,c]==1]:
    vs = neighbours(u,grid)
    for v in vs:
      G[u].append((v,euclidean(u,v)))
  return G

# computes shortest paths between s and every other node in G
def Dijkstras(G, s):
  Q = set()
  dist = defaultdict(lambda: np.nan)
  prev = defaultdict(lambda: None)

  for v in G.keys():
    dist[v] = np.inf
    prev[v] = None
    Q.add(v)

  dist[tuple(s)] = 0

  while len(Q) > 0:
    u = min((v for v in dist if v in Q),key=dist.get)
    Q.remove(u)

    for v,length in G[u]:
      alt = dist[u] + length
      if alt < dist[v]:
        dist[v] = alt
        prev[v] = u

  return dist,prev

def shortest_path_mat(grid,s):
  rows,cols = grid.shape
  G = create_graph(grid)
  dist,_ = Dijkstras(G,s)
  return np.array([[dist[(r,c)] for c in range(cols)] for r in range(rows)])
