import numpy as np
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

# assumes intervals are ordered by start time
def merge_intervals(intervals):
  result = np.empty((0,2))
  startj = intervals[0,0]
  endj   = intervals[0,1]
  for i in range(1,len(intervals)):
    startc = intervals[i,0]
    endc   = intervals[i,1]
    if startc <= endj:
      endj = max(endc, endj)
    else:
      result = np.append(result, np.array([[startj, endj]]), axis=0)
      startj = startc
      endj   = endc
  return np.append(result, np.array([[startj, endj]]), axis=0)

def plot_intervals(intervals):
  fint = intervals.flatten()
  line = np.insert(fint, range(2,len(fint),2), np.nan)
  plt.plot(line, np.zeros(len(line)))
  plt.show()

def spw_r_detect(eegs,samprates):
  signals = np.array([butter_bandpass_filter(eeg,150,250,samprate) for (eeg,samprate) in zip(eegs,samprates)])
  envs = np.array([np.abs(hilbert(signal)) for signal in signals])
  means = np.array([np.mean(env) for env in envs])
  sd3s = np.array([np.mean(env)+1*np.std(env) for env in envs])

  #samps_per_bin = 15*1e-3*samprates
  larges = np.array([
    keep_intervals_ge_length(vec_to_intervals(vec),23)
      for vec in np.array([env > mean for (env,mean) in zip(envs,means)]).astype(int)
  ])
  peaks = np.array([
    keep_intervals_ge_length(vec_to_intervals(vec),23)
      for vec in np.array([env > sd3 for (env,sd3) in zip(envs,sd3s)]).astype(int)
  ])
  rips = np.vstack(np.array([keep_intersects(large,peak) for (large,peak) in zip(larges,peaks)]))
  rips = merge_intervals(rips[rips[:,0].argsort()]) # sort rows by first column; then merge overlaps

  return rips,signals,envs
