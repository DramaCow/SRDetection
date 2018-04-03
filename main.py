import numpy as np
import matplotlib.pyplot as plt
import decoder as bd
from astar import astar, create_grid, Node

# === DATA SOURCE ===
from lf import pos, spk, maze_epoch, spatial_bin_length, rips
lin_point = np.array([35,50])

# === DECODER ===
decoder = bd.Decoder(pos,spk,spatial_bin_length,lin_point)
f = decoder.calc_f_2d(maze_epoch)

rips = rips[[sum(decoder.get_n((rip[0],rip[1]))>0)>=5 for rip in rips]]
rip = max(rips,key=lambda rip: rip[1]-rip[0])
print(rip)
num_bins = np.round((rip[1]-rip[0])/15e-3).astype(int)
start = rip[0]
end = rip[0] + 15e-3*num_bins
if sum(decoder.get_n((start,end))>0) >= 5: # arbitrary threshold requirement
  for i in range(num_bins):
    n = decoder.get_n((start+15e-3*i,start+15e-3*(i+1)))
    p2 = decoder.prob_X_given_n(n,f,15e-3)
    p1 = decoder.prob_X2_to_X1(p2)
    samples = np.random.choice(int(decoder.lim1d),size=10000,p=p1)
