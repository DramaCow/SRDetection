import sys
import numpy as np
import scipy.io as sio
from scipy.signal import hilbert, gaussian
import matplotlib.pyplot as plt
from spw_r import spw_r_detect, plot_ripples

spatial_bin_length = 2

def get_lfp_data(day, epoch):
  # pos info
  pos_mat = sio.loadmat('Con/conpos%02d.mat' % (day+1))
  pos_info = pos_mat['pos'][0][day][0][epoch][0]['data'][0]
  pos = pos_info[:,0:3]
  
  # epoch info
  epoch_interval = np.array([min(pos_info[:,0]),max(pos_info[:,0])])
  
  # tetrode info
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

  return pos, epoch_interval, eegs, starttimes, samprates

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

_, pre_epoch, pre_eegs, pre_starttimes, pre_samprates = get_lfp_data(day, epoch)
_, maze_epoch, maze_eegs, maze_starttimes, maze_samprates = get_lfp_data(day, epoch+1)
_, post_epoch, post_eegs, post_starttimes, post_samprates = get_lfp_data(day, epoch+2)

window_time = 10e-3
window_size = int(window_time*1250)

num_training_samples = 10000
samples_pre  = generate_lfp_samples(pre_eegs,window_size,num_training_samples)
samples_maze = generate_lfp_samples(maze_eegs,window_size,num_training_samples)
features_pre  = get_lfp_features(samples_pre)
features_maze = get_lfp_features(samples_maze)
X_train = np.concatenate((features_pre, features_maze), axis=0)
y_train = np.concatenate((np.zeros(num_training_samples),np.ones(num_training_samples)),axis=0)

num_testing_samples = 1000
samples_pre  = generate_lfp_samples(pre_eegs,window_size,num_testing_samples)
samples_maze = generate_lfp_samples(maze_eegs,window_size,num_testing_samples)
features_pre  = get_lfp_features(samples_pre)
features_maze = get_lfp_features(samples_maze)
X_test = np.concatenate((features_pre, features_maze), axis=0)
y_test = np.concatenate((np.zeros(num_testing_samples),np.ones(num_testing_samples)),axis=0)

num_samples_post = 1000
samples_post = get_lfp_samples(post_eegs,window_size,num_samples_post)
features_post = get_lfp_features(samples_post)
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
