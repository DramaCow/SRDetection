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

def generate_lfp_samples(eegs,window_size,num_samples):
  inds = np.random.randint(low=0, high=len(eegs[0])-window_size, size=(num_samples,))
  samples = np.array([[eeg[idx:idx+window_size] for eeg in eegs] for idx in inds])
  return samples

def get_lfp_samples(eegs,window_size,num_samples):
  inds = np.linspace(start=0, stop=len(eegs[0])-window_size, num=num_samples).astype(int)
  samples = np.array([[eeg[idx:idx+window_size] for eeg in eegs] for idx in inds])
  return samples

def get_lfp_features(samples):
  means = np.mean(samples,axis=2)
  #return means/np.mean(means,axis=0)
  return means

day = 0   # int in [0,5]
epoch = 0 # int in [0,4]

pre_pos, pre_epoch, pre_spk = get_data(day, epoch)
pre_eegs, pre_starttimes, pre_samprates = get_eegs(day, epoch)
maze_pos, maze_epoch, maze_spk = get_data(day, epoch+1)
maze_eegs, maze_starttimes, maze_samprates = get_eegs(day, epoch+1)
post_pos, post_epoch, post_spk = get_data(day, epoch+2)
post_eegs, post_starttimes, post_samprates = get_eegs(day, epoch+2)

window_time = 10e-3
window_size = int(window_time*1250)
#print('delay =',window_time*1000,'ms')

num_training_samples = 10000
pre_training_samples  = generate_lfp_samples(pre_eegs,window_size,num_training_samples)
maze_training_samples = generate_lfp_samples(maze_eegs,window_size,num_training_samples)
features_pre  = get_lfp_features(pre_training_samples)
features_maze = get_lfp_features(maze_training_samples)
X_train = np.concatenate((features_pre, features_maze), axis=0)
y_train = np.concatenate((np.zeros(num_training_samples),np.ones(num_training_samples)),axis=0)

num_testing_samples = 1000
pre_testing_samples  = generate_lfp_samples(pre_eegs,window_size,num_testing_samples)
maze_testing_samples = generate_lfp_samples(maze_eegs,window_size,num_testing_samples)
features_pre  = get_lfp_features(pre_testing_samples)
features_maze = get_lfp_features(maze_testing_samples)
X_test = np.concatenate((features_pre, features_maze), axis=0)
y_test = np.concatenate((np.zeros(num_testing_samples),np.ones(num_testing_samples)),axis=0)

num_post_samples = 1000
post_samples = get_lfp_samples(post_eegs,window_size,num_post_samples)
features_post = get_lfp_features(post_samples)
X_post = features_post

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = RandomForestClassifier(n_estimators=20, max_depth=16)
print(type(X_train))
print(X_train.shape)
clf.fit(X_train, y_train)
errors = sum(np.abs(clf.predict(X_test)-y_test))
accuracy = (len(y_test)-errors)/len(y_test)
print(accuracy)

#replay = clf.predict(X_post)
#print(replay)
