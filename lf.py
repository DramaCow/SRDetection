import numpy as np
import scipy.io as sio
from scipy.signal import hilbert, gaussian
import matplotlib.pyplot as plt
from bandpass import butter_bandpass_filter

def vec_to_intervals(vec):
  intervals = np.empty((0,2))
  start = None
  for i in range(len(vec)):
    if vec[i]:
      if start is None:
        start = i
    else:
      if start is not None:
        end = i-1
        intervals = np.append(intervals,[[start,end]],axis=0)
        start = None
  if start is not None:
    intervals = np.append(intervals,[[start,len(vec)-1]],axis=0)
  return intervals

def keep_intervals_ge_length(intervals,length):
  return np.array([interval for interval in intervals if (interval[1] - interval[0]) >= length])

def keep_intersects(A,B):
  if (A.size == 0 or B.size == 0):
    return np.empty((0,2))
  else:
    return np.array([interval for interval in A if any(np.logical_and(interval[0] <= B[:,0],B[:,1] <= interval[1]))])

day = 0   # int in [0,5]
epoch = 0 # int in [0,4]

# pos info
pos_mat = sio.loadmat('Con/conpos%02d.mat' % (day+1))
pos_info = pos_mat['pos'][0][day][0][epoch][0]['data'][0]
pos = pos_info[:,0:3]

# epoch info
maze_epoch = np.array([min(pos_info[:,0]),max(pos_info[:,0])])

# spike info
spk_mat = sio.loadmat('Con/conspikes%02d.mat' % (day+1))
spk_info = np.array([units[0]['data'][0] for tetrode in spk_mat['spikes'][0][day][0][epoch][0] for units in tetrode[0] if units.size > 0])
spk = np.array([unit[:,0] for unit in spk_info if unit.size > 0])

# eeg (lfp) info
tetrodes = range(1)
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
signals = np.array([butter_bandpass_filter(eeg,150,250,samprate) for (eeg,samprate) in zip(eegs,samprates)])
envs = np.array([np.abs(hilbert(signal)) for signal in signals])
means = np.array([np.mean(env) for env in envs])
sd3s = np.array([np.mean(env)+1*np.std(env) for env in envs])

#samps_per_bin = 15*1e-3*samprates
larges = np.array([
  keep_intervals_ge_length(vec_to_intervals(vec),23) for vec in np.array([env > mean for (env,mean) in zip(envs,means)]).astype(int)
])
peaks = np.array([
  keep_intervals_ge_length(vec_to_intervals(vec),23) for vec in np.array([env > sd3 for (env,sd3) in zip(envs,sd3s)]).astype(int)
])
rips = np.array([keep_intersects(large,peak) for (large,peak) in zip(larges,peaks)])

print(rips)

'''
for samprate,signal,env,mean,sd3,large,peak in zip(samprates,signals,envs,means,sd3s,larges,peaks):
  fig = plt.figure()
  for i in range(30):
    start = i*int(samprate/2)
    end = (i+1)*int(samprate/2)
    #plt.plot(signal[start:end])
    #plt.plot(env[start:end],'r-')
    #plt.plot([0,int(samprate/2)],[mean,mean])
    #plt.plot([0,int(samprate/2)],[sd3,sd3])
    #plt.ylim([min(signal),max(signal)])
    plt.plot(large[start:end],'r-')
    plt.plot(2*peak[start:end],'r-')
    plt.show(block=False) ; plt.pause(0.01) ; fig.clf()
'''
