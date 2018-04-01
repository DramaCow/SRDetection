import numpy as np
from scipy.signal import hilbert, gaussian
import matplotlib.pyplot as plt
from bandpass import butter_bandpass_filter

# convert binary vector to list of intervals (w/ start and end times)
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

# keep intervals greater than length
def keep_intervals_ge_length(intervals,length):
  return np.array([interval for interval in intervals if (interval[1] - interval[0]) >= length])

# keep intervals in A that cover any intervals in B
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

def plot_ripples(samprates,rips,sigs,envs,duration=None,stride=20,delay=0.1):
  lims = np.array([max(np.abs(np.min(sig)),np.abs(np.max(sig))) for sig in sigs])
  sigs_n = np.array([sig/(2*lim) for (sig,lim) in zip(sigs,lims)])
  envs_n = np.array([env/(2*lim) for (env,lim) in zip(envs,lims)])

  duration = duration if duration is not None else np.min([len(sig) for sig in sigs])
  
  fig = plt.figure()
  for i in np.arange(0,duration,stride):
    ax = plt.axes()
    for j,(samprate,sig,env) in enumerate(zip(samprates,sigs_n,envs_n)):
      start = i
      end = i+int(samprate/2)
  
      rips_visible = rips[np.logical_and(rips[:,1]>=start,rips[:,0]<=end)]
      for rip in rips_visible:
        ax.axvspan(rip[0], rip[1], alpha=0.3)
  
      plt.plot(range(start,end),sig[start:end]+j,'k-')
      plt.plot(range(start,end),env[start:end]+j,'r-')
      plt.xlim([start,end])
      plt.ylim([-0.5,len(sigs_n)-0.5])
    plt.show(block=False) ; plt.pause(delay) ; fig.clf()

def spw_r_detect(eegs,samprates,min_length=15e-3):
  signals = np.array([butter_bandpass_filter(eeg,150,250,samprate) for (eeg,samprate) in zip(eegs,samprates)])
  envs = np.array([np.abs(hilbert(signal)) for signal in signals])
  means = np.array([np.mean(env) for env in envs])
  sd3s = np.array([np.mean(env)+3*np.std(env) for env in envs])

  # intervals w/ envelope greater than mean (for longer than min_length)
  larges = np.array([
    keep_intervals_ge_length(vec_to_intervals(vec),min_length*samprate)
      for (samprate,vec) in zip(samprates,np.array([env > mean for (env,mean) in zip(envs,means)]).astype(int))
  ])
  # intervals w/ envelope greater than mean + 3*s.d (for longer than min_length)
  peaks = np.array([
    keep_intervals_ge_length(vec_to_intervals(vec),min_length*samprate)
      for (samprate,vec) in zip(samprates,np.array([env > sd3 for (env,sd3) in zip(envs,sd3s)]).astype(int))
  ])

  # keep intervals in largest that contain intervals in peaks
  rips = np.vstack(np.array([keep_intersects(large,peak) for (large,peak) in zip(larges,peaks)]))

  # sort rows by first column; then merge overlaps
  if rips.size > 0:
    rips = merge_intervals(rips[rips[:,0].argsort()]) 

  return rips,signals,envs
