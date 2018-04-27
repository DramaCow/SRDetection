import sys
import numpy as np
import scipy.io as sio
from scipy.signal import hilbert, gaussian
import matplotlib.pyplot as plt
from spw_r import spw_r_detect, plot_ripples
from math import floor, ceil

spatial_bin_length = 2

def construct_mat(spk, interval, dt):
  duration = interval[1]-interval[0]
  spk_int = np.array([tetrode[(interval[0]<=tetrode)&(tetrode<=interval[1])]-interval[0] for tetrode in spk])
  N = spk_int.size
  T = ceil(duration/dt) # ceil ensures there is always enough bins
  M = np.zeros((N, T), dtype='uint8')
  for j, tetrode in enumerate(spk_int):
    for spike_time in tetrode:
      bin = floor(spike_time/dt)
      M[j][bin] = 1
  which_neurons_spike = np.max(M, axis=1)
  print(which_neurons_spike)
  return M

def per_second_fr(M, dt):
  N,T = M.shape
  duration = T*dt
  bw = int(1/dt) # bin width
  S = ceil(duration/1)
  FR = np.empty((N,S))
  for i in range(S):
    FR[:,i] = np.sum(M[:,i*bw:(i+1)*bw],1)
  FR = FR/1
  return FR

def get_data(day, epoch):
  # pos info
  pos_mat = sio.loadmat('Con/conpos%02d.mat' % (day+1))
  pos_info = pos_mat['pos'][0][day][0][epoch][0]['data'][0]
  pos = pos_info[:,0:3]
  
  # epoch info
  epoch_interval = np.array([min(pos_info[:,0]),max(pos_info[:,0])])
  
  #spk_info = np.array([units[0]['data'][0] for tetrode in spk_mat['spikes'][0][day][0][epoch][0] for units in tetrode[0] if units.size > 0])

  # multi-unit spike info
  spk_mat = sio.loadmat('Con/conspikes%02d.mat' % (day+1))
  tetrodes = spk_mat['spikes'][0][day][0][epoch][0]
  spk = [
    [[unit[0] for unit in units[0]['data'][0] if unit[0].size > 0]
      for units in tetrode[0] if units.size > 0]
    for tetrode in tetrodes
  ]
  #spk = np.array([np.sort((np.concatenate(tetrode) if len(tetrode) > 1 else np.array(tetrode)).flatten()) for tetrode in spk])
  spk = np.array([np.sort((np.concatenate(tetrode) if len(tetrode) > 1 else np.array(tetrode)).flatten()) for tetrode in spk if len(tetrode) > 0])

  return pos, epoch_interval, spk

day = 0   # int in [0,5]
epoch = 0 # int in [0,4]

pos,maze_epoch,spk = get_data(day, epoch)
print(construct_mat(spk, (maze_epoch[0],maze_epoch[0]+10), 10e-3))
