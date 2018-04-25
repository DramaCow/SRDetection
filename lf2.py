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
  
  # spike info
  spk_mat = sio.loadmat('Con/conspikes%02d.mat' % (day+1))
  spk_info = np.array([units[0]['data'][0] for tetrode in spk_mat['spikes'][0][day][0][epoch][0] for units in tetrode[0] if units.size > 0])
  spk = np.array([unit[:,0] for unit in spk_info if unit.size > 0])

  return pos, epoch_interval, spk

def get_eegs(day, epoch):
  tetrodes = range(30)
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
  return eegs, starttimes, samprates

def get_rips(day, epoch):
  # eeg (lfp) info
  eegs, starttimes, samprates = get_eegs(day, epoch)
  rips,times,sigs,envs = spw_r_detect(eegs,samprates,starttimes)
  #plot_ripples(rips,times,sigs,envs,stride=50,delay=0.5)
  #plot_ripples(rips,times,sigs,envs,window_size=900000,stride=50,delay=None)
  #plt.show()
  return rips,times,sigs,envs

day = 0   # int in [0,5]
epoch = 0 # int in [0,4]

pre_pos, pre_epoch, pre_spk = get_data(day, epoch)
pre_eegs, pre_starttimes, pre_samprates = get_eegs(day, epoch)
maze_pos, maze_epoch, maze_spk = get_data(day, epoch+1)
maze_eegs, maze_starttimes, maze_samprates = get_eegs(day, epoch)

print(pre_epoch)
print(maze_epoch)

window_size = 32
pre_training_idx = np.random.randint(low=0, high=len(pre_eegs[0]-window_size), size=(10000,))
maze_training_idx = np.random.randint(low=0, high=len(maze_eegs[0]-window_size), size=(10000,))

pre_training_samples = np.array([[eeg[idx:idx+window_size] for eeg in pre_eegs] for idx in pre_training_idx])
maze_training_samples = np.array([[eeg[idx:idx+window_size] for eeg in maze_eegs] for idx in maze_training_idx])

pre_training_means = np.mean(pre_training_samples,axis=2)
maze_training_means = np.mean(maze_training_samples,axis=2)
