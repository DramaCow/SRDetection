import numpy as np
from scipy.spatial.distance import euclidean as dist

class Node:
  def __init__(self,point,value):
    self.value = value
    self.point = point
    self.parent = None
    self.H = 0
    self.G = 0

def children(point,grid):
  y,x = point
  h,w = grid.shape
  return [grid[r,c] for r in range(y-1,y+2) 
                    for c in range(x-1,x+2) 
                    if  r in range(h)  and
                        c in range(w)  and
                        (r!=y or c!=x) and
                        grid[r,c].value==1] 

def create_grid(arr):
  return np.array([[Node((r,c),int(arr[r,c])) for c in range(arr.shape[1])] for r in range(arr.shape[0])])
    
def astar(start, goal, grid):
  openset = set()
  closedset = set()
  current = grid[start]
  openset.add(current)

  #While the open set is not empty
  while openset:
    current = min(openset,key=lambda o:o.G+o.H) # current is item with lowest F score in openset

    #If it is the item we want, retrace the path and return it
    if current.point == goal:
      path = []
      while current.parent:
        path.append(current)
        current = current.parent
      path.append(current)
      return path[::-1]

    openset.remove(current)
    closedset.add(current)

    for node in children(current.point,grid):
      if node in closedset:
        continue

      if node in openset:
        new_g = current.G + dist(current.point,node.point)
        if node.G > new_g:
          node.G = new_g
          node.parent = current
      else:
        node.G = current.G + dist(current.point,node.point)
        node.H = dist(node.point,goal)
        node.parent = current
        openset.add(node)

  #Throw an exception if there is no path
  raise ValueError('No Path Found')

'''
import random
mat = np.array([[1,0,0,0],
                [1,1,1,0],
                [1,1,1,1],
                [0,0,0,0]])
grid = np.array([[Node((r,c),mat[r,c]) for c in range(mat.shape[1])] for r in range(mat.shape[0])])
start = random.sample([(r,c) for c in range(grid.shape[1]) for r in range(grid.shape[0]) if grid[r,c].value==1],1)[0]
goal = random.sample([(r,c) for c in range(grid.shape[1]) for r in range(grid.shape[0]) if grid[r,c].value==1 and (r,c)!=start],1)[0]
print(start, goal)
print(np.array([[grid[r,c].value for c in range(grid.shape[1])] for r in range(grid.shape[0])]))
print(np.array([[1 if (r,c)==start else 2 if (r,c)==goal else 0 for c in range(grid.shape[1])] for r in range(grid.shape[0])]))
print([c.point for c in children(grid[start].point,grid)])
print([node.point for node in astar(grid[start],grid[goal],grid)])
'''
