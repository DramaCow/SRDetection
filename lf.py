import sys
import numpy as np
import scipy.io as sio
from scipy.signal import hilbert, gaussian
import matplotlib.pyplot as plt
from spw_r import spw_r_detect, plot_ripples

def get_data(day,epoch):
  # pos info
  pos_mat = sio.loadmat('Con/conpos%02d.mat' % (day+1))
  pos_info = pos_mat['pos'][0][day][0][epoch][0]['data'][0]
  pos = pos_info[:,0:3]
  
  # epoch info
  this_epoch = np.array([min(pos_info[:,0]),max(pos_info[:,0])])
  
  # spike info
  spk_mat = sio.loadmat('Con/conspikes%02d.mat' % (day+1))
  spk_info = np.array([units[0]['data'][0] for tetrode in spk_mat['spikes'][0][day][0][epoch][0] for units in tetrode[0] if units.size > 0])
  spk = np.array([unit[:,0] for unit in spk_info if unit.size > 0])

  return pos,this_epoch,spk

day = 0   # int in [0,5]
epoch = 1 # int in [0,4]

#print(min(pos[:,1]), max(pos[:,1]))
#print(min(pos[:,2]), max(pos[:,2]))
spatial_bin_length = 2

#plt.plot(pos[:,1],pos[:,2],'k.')
#plt.show()

pos,maze_epoch,spk = get_data(day, epoch)
pos_maze,maze_epoch,spk_maze = pos,maze_epoch,spk
epoch = epoch + 1 # move to post epoch
pos_post,post_epoch,spk_post = get_data(day, epoch)

# eeg (lfp) info
tetrodes = range(30)#range(13,17)
eegs = np.array([
  sio.loadmat('Con/EEG/coneeg%02d-%1d-%02d.mat' % (day+1,epoch+1,tetrode+1))
  ['eeg'][0][day][0][epoch][0][tetrode][0]['data'][0].flatten() for tetrode in tetrodes
])
starttimes = np.array([
  sio.loadmat('Con/EEG/coneeg%02d-%1d-%02d.mat' % (day+1,epoch+1,tetrode+1))
  ['eeg'][0][day][0][epoch][0][tetrode][0]['starttime'][0][0][0] for tetrode in tetrodes
])
samprates = np.array([
  sio.loadmat('Con/EEG/coneeg%02d-%1d-%02d.mat' % (day+1,epoch+1,tetrode+1))
  ['eeg'][0][day][0][epoch][0][tetrode][0]['samprate'][0][0][0] for tetrode in tetrodes
])
print(len(eegs[0]))
rips,times,sigs,envs = spw_r_detect(eegs,samprates,starttimes)
#plot_ripples(rips,times,sigs,envs,stride=50,delay=0.5)
#plot_ripples(rips,times[13:17],sigs[13:17],envs[13:17],window_size=900000,stride=50,delay=None)
plt.show()
