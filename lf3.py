import sys
import numpy as np
import scipy.io as sio
from scipy.signal import hilbert, gaussian
import matplotlib.pyplot as plt
from spw_r import spw_r_detect, plot_ripples
from math import floor, ceil

spatial_bin_length = 2

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

def shifting(bitlist):
  out = 0
  for bit in bitlist:
    out = (out << 1) | bit
  return out

def binary(decimal):
  return np.array([int(x) for x in bin(decimal)[2:]])

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

def ind_model(M):
  N,T = M.shape
  count = np.sum(M,axis=1)
  p = count/T
  return p

def prob_x_given_ind(x,params):
  on  = x
  off = 1-x
  prob = np.prod(params[on])*np.prod(1-params[off])
  return prob

def generate_samples(spk,interval,window_size,num_samples):
  times = np.random.uniform(interval[0]+window_size, interval[1], size=(num_samples,))
  Ms = np.array([construct_mat(spk, (time-window_size,time), window_size, 10e-3) for time in times])
  return Ms

day = 0   # int in [0,5]
epoch = 0 # int in [0,4]

dt = 10e-3
duration = 3000
pos,maze_epoch,spk = get_data(day, epoch)
M = construct_mat(spk, (maze_epoch[0],maze_epoch[0]+duration), duration, dt)
#display_raster(M,dt)

params = ind_model(M)

samples = generate_samples(spk,maze_epoch,100e-3,3)
print(samples)

'''
V = np.array([shifting(M[:,col]) for col in range(M.shape[1])])
count = np.zeros(2**M.shape[0])
for v in V:
  count[v] += 1
p = count/sum(count)
plt.plot(p)
plt.show()
'''
