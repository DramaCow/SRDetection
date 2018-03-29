import numpy as np
from dijkstras import shortest_path_mat

grid=np.array([
  [1,1,1],
  [1,0,0],
  [0,0,1]
])
start_point = (0,0)
distmat = shortest_path_mat(grid,start_point)

print(grid)
print(start_point)
print(distmat)
