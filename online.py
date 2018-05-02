import sys
import numpy as np
import scipy.io as sio
from scipy.signal import hilbert, gaussian
import matplotlib.pyplot as plt
from spw_r import spw_r_detect, plot_ripples
from math import floor, ceil
from poptrack import *
from ind_model import *

# =====================
# === LFP FUNCTIONS ===
# =====================

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

  return pos, epoch_interval, (eegs, starttimes, samprates)

def get_lfp_samples(lfps,window_size,inds):
  eegs,starttimes,samprates = lfps
  samples = np.array([[eeg[idx:idx+window_size] for eeg in eegs] for idx in inds])
  return samples

def generate_lfp_samples(lfps,window_size,internval,num_samples):
  inds = np.random.randint(low=0, high=len(lfps[0][0])-window_size, size=(num_samples,))
  samples = get_lfp_samples(lfps,window_size,inds)
  return samples

def get_lfp_features(samples):
  means = np.mean(samples,axis=2)
  #return means/np.mean(means,axis=0)
  return means

# =====================
# === MUA FUNCTIONS ===
# =====================

def raster(event_times_list, y_labels=None, **kwargs):
  """
  Creates a raster plot
  Parameters
  ----------
  event_times_list : iterable
                     a list of event time iterables
  color : string
          color of vlines
  Returns
  -------
  ax : an axis containing the raster plot
  """
  ax = plt.gca()
  for i, trial in enumerate(event_times_list):
    plt.vlines(trial, i-0.475, i+0.475, **kwargs)
  plt.ylim(-0.5, len(event_times_list) - 0.5)
  if y_labels == None:
    plt.yticks(range(0, len(event_times_list)))
  else:
    plt.yticks(y_labels)
  return ax

def display_raster(M, dt):
  N, T = M.shape[0], M.shape[1]
  fig = plt.figure()
  spikes = [[t*dt + dt/2 for t in range(T) if M[n][t] == 1] for n in range(N)]
  ax = raster(spikes)
  plt.title('Spike Train')
  plt.xlabel('Time (s)')
  plt.ylabel('Neuron')
  plt.show()
  #plt.savefig(filename+'.png')

def construct_mat(spk, interval, duration, dt):
  spk_int = np.array([tetrode[(interval[0]<=tetrode)&(tetrode<=interval[1])]-interval[0] for tetrode in spk])
  N = spk.size
  T = ceil(duration/dt) # ceil ensures there is always enough bins
  M = np.zeros((N, T), dtype='uint8')
  for j, tetrode in enumerate(spk_int):
    for spike_time in tetrode:
      bin = floor(spike_time/dt)
      M[j][bin] = 1
  #which_neurons_spike = np.max(M, axis=1)
  #print(which_neurons_spike)
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

def get_mua_data(day, epoch):
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

def get_mua_samples(spk,window_size,times):
  Ms = np.array([construct_mat(spk, (time-window_size,time), window_size, 10e-3) for time in times])
  return Ms

def generate_mua_samples(spk,window_size,interval,num_samples):
  times = np.random.uniform(interval[0]+window_size, interval[1], size=(num_samples,))
  Ms = get_mua_samples(spk,window_size,times)
  return Ms

def get_mua_features(samples):
  ind_feat = np.log(np.array([
    [prob_x_given_ind(sample[:,col],ind_params_pre)/prob_x_given_ind(sample[:,col],ind_params_maze)
    for col in range(sample.shape[1])] for sample in samples
  ]))
  poptrack_feat = np.log(np.array([
    [prob_x_given_poptrack(sample[:,col],poptrack_params_pre)/prob_x_given_poptrack(sample[:,col],poptrack_params_maze)
    for col in range(sample.shape[1])] for sample in samples
  ]))
  rowsum_feat = np.array([np.sum(sample,axis=0) for sample in samples])
  colsum_feat = np.array([np.sum(sample,axis=1) for sample in samples])
  features = np.concatenate((ind_feat,poptrack_feat,rowsum_feat,colsum_feat),axis=1)
  return features

# ======================
# === BOTH FUNCTIONS ===
# ======================

# ============
# === MAIN ===
# ============

# === LFP ===
if 1:
  day = 0   # int in [0,5]
  epoch = 0 # int in [0,4]
  
  _, epoch_pre,  pre_lfps  = get_lfp_data(day, epoch)
  _, epoch_maze, maze_lfps = get_lfp_data(day, epoch+1)
  _, epoch_post, post_lfps = get_lfp_data(day, epoch+2)
  
  window_time = 10e-3
  window_size = int(window_time*1250)
  
  num_training_samples = 10000
  samples_pre  = generate_lfp_samples(pre_lfps,window_size,epoch_pre,num_training_samples)
  samples_maze = generate_lfp_samples(maze_lfps,window_size,epoch_maze,num_training_samples)
  features_pre  = get_lfp_features(samples_pre)
  features_maze = get_lfp_features(samples_maze)
  X_train = np.concatenate((features_pre, features_maze), axis=0)
  y_train = np.concatenate((np.zeros(num_training_samples),np.ones(num_training_samples)),axis=0)
  
  num_testing_samples = 1000
  samples_pre  = generate_lfp_samples(pre_lfps,window_size,epoch_pre,num_testing_samples)
  samples_maze = generate_lfp_samples(maze_lfps,window_size,epoch_maze,num_testing_samples)
  features_pre  = get_lfp_features(samples_pre)
  features_maze = get_lfp_features(samples_maze)
  X_test = np.concatenate((features_pre, features_maze), axis=0)
  y_test = np.concatenate((np.zeros(num_testing_samples),np.ones(num_testing_samples)),axis=0)
  
  num_samples_post = 1000
  samples_post = generate_lfp_samples(post_lfps,window_size,epoch_post,num_samples_post)
  features_post = get_lfp_features(samples_post)
  X_post = features_post
  
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.datasets import make_classification
  
  clf = RandomForestClassifier(n_estimators=20, max_depth=16)
  clf.fit(X_train, y_train)
  errors = sum(np.abs(clf.predict(X_test)-y_test))
  accuracy = (len(y_test)-errors)/len(y_test)
  print(accuracy)
  
  #replay = clf.predict(X_post)
  #print(replay)

# === MUA ===
if 1:
  day   = 0 # int in [0,5]
  epoch = 0 # int in [0,4]
  
  _,epoch_pre, spk_pre  = get_mua_data(day, epoch)
  _,epoch_maze,spk_maze = get_mua_data(day, epoch+1)
  _,epoch_post,spk_post = get_mua_data(day, epoch+2)
  
  dt = 10e-3
  
  M_pre = construct_mat(
    spk_pre, (epoch_pre[0],epoch_pre[1]),
    epoch_pre[1]-epoch_pre[0], dt)
  M_maze = construct_mat(
    spk_maze, (epoch_maze[0],epoch_maze[1]),
    epoch_maze[1]-epoch_maze[0], dt)
  
  ind_params_pre  = ind_model(M_pre )
  ind_params_maze = ind_model(M_maze)
  poptrack_params_pre  = poptrack(M_pre )
  poptrack_params_maze = poptrack(M_maze)
  
  num_training_samples = 10000
  samples_pre  = generate_mua_samples(spk_pre,100e-3,epoch_pre,num_training_samples)
  samples_maze = generate_mua_samples(spk_maze,100e-3,epoch_maze,num_training_samples)
  features_pre  = get_mua_features(samples_pre)
  features_maze = get_mua_features(samples_maze)
  X_train = np.concatenate((features_pre, features_maze), axis=0)
  y_train = np.concatenate((np.zeros(num_training_samples),np.ones(num_training_samples)),axis=0)
  
  num_testing_samples = 1000
  samples_pre  = generate_mua_samples(spk_pre,100e-3,epoch_pre,num_testing_samples)
  samples_maze = generate_mua_samples(spk_maze,100e-3,epoch_maze,num_testing_samples)
  features_pre  = get_mua_features(samples_pre)
  features_maze = get_mua_features(samples_maze)
  X_test = np.concatenate((features_pre, features_maze), axis=0)
  y_test = np.concatenate((np.zeros(num_testing_samples),np.ones(num_testing_samples)),axis=0)
  
  num_post_samples = 1000
  samples_post  = generate_mua_samples(spk_post,100e-3,epoch_post,num_post_samples)
  features_post = get_mua_features(samples_post)
  X_post = features_post
  
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.datasets import make_classification
  
  clf = RandomForestClassifier(n_estimators=20, max_depth=32)
  clf.fit(X_train, y_train)
  errors = sum(np.abs(clf.predict(X_test)-y_test))
  accuracy = (len(y_test)-errors)/len(y_test)
  print(accuracy)
  
  #replay = clf.predict(X_post)
  #print(replay)
