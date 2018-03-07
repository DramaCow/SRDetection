import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage.filters import gaussian_filter
from math import ceil, floor, factorial

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

def find_closest(A, targets):
  inds = np.clip(A.searchsorted(targets), 1, len(A)-1)
  left = A[inds-1]
  right = A[inds]
  return inds-(targets-left < right-targets)

def sum_neighbours(M,x,y):
  (nx,ny) = M.shape
  val = 0
  if x > 0:    val += M[y,x-1]
  if x < nx-1: val += M[y,x+1]
  if y > 0:    val += M[y-1,x]
  if y < ny-1: val += M[y+1,x]
  return val

class Decoder:
  def __init__(self):
    # === LOAD ===
    print("loading data...", end="")

    # load necessary mat files
    spk = sio.loadmat('Tmaze_spiking_data.mat')
    loc = sio.loadmat('Tmaze_location_data.mat')
    rst = sio.loadmat('rippspin-times-FGHIJ.mat')
    
    print("COMPLETE.")
    # === PREPROCESS ===
    print("preprocessing data...", end="")

    # gets start and end times of each epoch
    self.pre_epoch  = np.concatenate(spk['epochs'][4,1:3]).ravel()
    self.maze_epoch = np.concatenate(spk['epochs'][4,3:5]).ravel()
    self.post_epoch = np.concatenate(spk['epochs'][4,5:7]).ravel()
    
    # looks at hippocampal neuron spike times only
    self.hc = [hc['tspk'].flatten() for hc in spk['Jcells'][spk['Jcells']['area'] == 'hc']]
    
    # ignores 0,0 positions (erroneous)
    self.pos = loc['Jpositiondata'][~np.all(loc['Jpositiondata'][:,1:3]==0,1),:]
    self.pos[:,0] = self.pos[:,0]/1e6
    self.maze_epoch = [max(self.maze_epoch[0],min(self.pos[:,0])), min(self.maze_epoch[1],max(self.pos[:,0]))]

    # get ripple periods (+/- 100ms around detected SPW-R peak times)
    self.rip = merge_intervals(np.append(rst['Jrip']-0.1, rst['Jrip']+0.1, axis=1))
    
    print("COMPLETE.")
    # === PARAMETERS ===

    # discretisation parameters
    self.n_spatial_bins = 32
    self.spatial_bin_size = np.amax(self.pos[:,1:3],0)/(self.n_spatial_bins-1)
    self.time_bin_size = 1
    self.n_time_bins = int(np.ceil(np.diff(self.maze_epoch)/self.time_bin_size))
    self.tau = 10

    # === DECODING ===
    
    # calculate (approximate) occupancy (total time spent in location bins)
    print("calculating 2D occupancy map...", end="")
    occ = self.posmat(self.pos,np.diff(self.maze_epoch)) # count no. ticks in each pos-bin & norm. by total dur.
    posmask = occ > 0                                    # any pos-bin that was accessed marked accessible
    accmask = np.array([[posmask[y,x] or sum_neighbours(posmask,x,y)>2 for x in range(posmask.shape[0])] for y in range(posmask.shape[1])])
    print("COMPLETE.")
    
    # approximate position of neuron firing
    self.f = np.empty((len(self.hc),self.n_spatial_bins,self.n_spatial_bins))
    for i in range(0,len(self.hc)):
      print("processing neurons: %d / %d\r" % (i, len(self.hc)), end="")
      tspk = self.get_spike_times(i,self.maze_epoch)         # get times neuron spiked during maze epoch
      self.f[i] = self.posmat(self.pos_at_time(tspk))        # count number of spikes occuring at each pos-bin
      self.f[i][posmask] = self.f[i][posmask] / occ[posmask] # fr = spike count / time spent in each pos-bin
      self.f[i] = gaussian_filter(self.f[i],1)*accmask     # blur a little
    print("processing neurons...COMPLETE.")
    print("all done.")

  def get_spike_times(self, i, interval):
    return self.hc[i][np.logical_and(interval[0]<=self.hc[i], self.hc[i]<=interval[1])]

  def posmat(self,pos,a=None):
    bin_pos = np.round(pos[:,1:3]/self.spatial_bin_size)
    occ = np.zeros((self.n_spatial_bins,self.n_spatial_bins))
    for y in range(0,self.n_spatial_bins):
      for x in range(0,self.n_spatial_bins):
        occ[y,x] = np.sum(np.all(bin_pos == [x,y],axis=1))
    if a != None:
      occ = (a/np.sum(np.sum(occ)))*occ
    occ[np.isnan(occ)] = 0
    return occ

  def pos_at_time(self,times):
    return np.append(np.array([times]).T, self.pos[find_closest(self.pos[:,0], times), 1:3], axis=1)

  def approx_N_and_Pos(self):
    tidx = np.floor((self.pos[:,0]-np.min(self.pos[:,0]))/self.time_bin_size)
    pos = np.round(np.array([np.mean(self.pos[tidx==i,1:3],axis=0) for i in range(self.n_time_bins)])/self.spatial_bin_size).astype(int)
    N = np.empty((len(self.hc),self.n_time_bins))
    for i in range(0,len(self.hc)):
      print("processing neurons: %d / %d\r" % (i, len(self.hc)), end="")
      tspk = self.get_spike_times(i,self.maze_epoch)
      tidx = np.floor((tspk-np.min(tspk))/self.time_bin_size)
      N[i] = np.array([np.sum(tidx==i) for i in range(self.n_time_bins)])
    print("processing neurons...COMPLETE.")
    print("all done.")
    return (N, pos)

  def P_NgivenX(self,n,x):
    xidx = tuple(x)
    ngtz = n[n > 0]
    return np.prod([((self.tau*self.f[i][xidx]**n[i])/factorial(n[i]))*np.exp(-self.tau*self.f[i][xidx]) for i in range(len(ngtz))])

decoder = Decoder()

#for i in range(len(decoder.f)):
#  plt.imshow(decoder.f[i],origin='lower')
#  plt.show()

truepos = decoder.pos[decoder.pos[:,0]<=decoder.maze_epoch[0]+1,:]
trueposmat = decoder.posmat(truepos)
print(truepos)
plt.imshow(trueposmat,origin='lower')
plt.show()

(N, pos) = decoder.approx_N_and_Pos()

#for i in range(len(decoder.hc)):
#  tspk = decoder.get_spike_times(i,(decoder.maze_epoch[0],decoder.maze_epoch[0]+1))
#  print(tspk)

for i in range(N.shape[1]):
  nvec = N[:,i]
  probmat = np.array([[decoder.P_NgivenX(nvec, [x,y]) for x in range(decoder.n_spatial_bins)] for y in range(decoder.n_spatial_bins)])
  plt.imshow(probmat,origin='lower')
  plt.show()
