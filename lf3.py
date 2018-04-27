import sys
import numpy as np
import scipy.io as sio
from scipy.signal import hilbert, gaussian
import matplotlib.pyplot as plt
from spw_r import spw_r_detect, plot_ripples

spatial_bin_length = 2

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
  spk = np.array([np.sort((np.concatenate(tetrode) if len(tetrode) > 1 else np.array(tetrode)).flatten()) for tetrode in spk])

  return pos, epoch_interval, spk

day = 0   # int in [0,5]
epoch = 0 # int in [0,4]

pos,maze_epoch,spk = get_data(day, epoch)
