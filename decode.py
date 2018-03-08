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
  (ny,nx) = M.shape
  val = 0
  if x > 0:    val += M[y,x-1]
  if x < nx-1: val += M[y,x+1]
  if y > 0:    val += M[y-1,x]
  if y < ny-1: val += M[y+1,x]
  return val

def matmax(M):
  (ny,nx) = M.shape
  maxval = -np.inf
  maxidx = [-1,-1]
  for y in range(ny):
    for x in range(nx):
      if M[y,x] > maxval:
        maxval = M[y,x]
        maxidx = [y,x]
  return (maxval, maxidx)

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

    # identifiers
    experiment = 4 # max is 4
    epoch_id = experiment
    cells_id = chr(ord('F')+experiment)+'cells'
    posit_id = chr(ord('F')+experiment)+'positiondata'
    rippl_id = chr(ord('F')+experiment)+'rip'

    # gets start and end times of each epoch
    self.pre_epoch  = np.concatenate(spk['epochs'][epoch_id,1:3]).ravel()
    self.maze_epoch = np.concatenate(spk['epochs'][epoch_id,3:5]).ravel()
    self.post_epoch = np.concatenate(spk['epochs'][epoch_id,5:7]).ravel()
    
    # looks at hippocampal neuron spike times only
    self.hc = [hc['tspk'].flatten() for hc in spk[cells_id][spk[cells_id]['area'] == 'hc']]
    
    # ignores 0,0 positions (erroneous)
    self.pos = loc[posit_id][~np.all(loc[posit_id][:,1:3]==0,1),:]
    self.pos[:,0] = self.pos[:,0]/1e6
    self.maze_epoch = [max(self.maze_epoch[0],min(self.pos[:,0])), min(self.maze_epoch[1],max(self.pos[:,0]))]

    # get ripple periods (+/- 100ms around detected SPW-R peak times)
    self.rip = merge_intervals(np.append(rst[rippl_id]-0.1, rst[rippl_id]+0.1, axis=1))

    print("COMPLETE.")
    # === PARAMETERS ===
    print("setting parameters...", end="")

    # discretisation parameters
    self.n_spatial_bins = 32
    self.spatial_bin_size = np.amax(self.pos[:,1:3],0)/(self.n_spatial_bins-1)

    # determine tranversable areas of map
    occ = self.posmat(self.pos,self.maze_epoch[1]-self.maze_epoch[0]) # count no. ticks in each pos-bin & norm. by total dur.
    posmask = occ > 0                                                 # any pos-bin that was accessed marked accessible
    self.accmask = np.array(
      [[posmask[y,x] or sum_neighbours(posmask,x,y)>2 for x in range(posmask.shape[0])] for y in range(posmask.shape[1])]
    )

    print("COMPLETE.")
    print("all done.")
    
  def calc_f_2d(self,interval):
    # calculate (approximate) occupancy (total time spent in location bins)
    print("calculating 2D occupancy map...", end="")
    occ = self.posmat(self.get_pos(interval),interval[1]-interval[0]) # count no. ticks in each pos-bin & norm. by total dur.
    posmask = occ > 0                                                 # any pos-bin that was accessed marked accessible
    print("COMPLETE.")
    # approximate position of neuron firing
    f = np.empty((len(self.hc),self.n_spatial_bins,self.n_spatial_bins))
    for i in range(0,len(self.hc)):
      print("processing neurons: %d / %d\r" % (i, len(self.hc)), end="")
      tspk = self.get_spike_times(i,interval)      # get times neuron spiked during interval
      f[i] = self.posmat(self.pos_at_time(tspk))   # count number of spikes occuring at each pos-bin
      f[i][posmask] = f[i][posmask] / occ[posmask] # fr = spike count / time spent in each pos-bin
      oldmax = np.max(np.max(f[i]))
      f[i] = gaussian_filter(f[i],1.0)*self.accmask  # blur a little
      #newmax = np.max(np.max(f[i]))
      #if newmax != 0:
      #  f[i] = (oldmax/newmax)*f[i]    # conserve old max value
    print("processing neurons...COMPLETE.")
    print("all done.")
    return f
    
  def get_spike_times(self,i,interval):
    return self.hc[i][np.logical_and(interval[0]<=self.hc[i], self.hc[i]<=interval[1])]

  def get_pos(self,interval):
    return self.pos[np.logical_and(interval[0]<=self.pos[:,0], self.pos[:,0]<=interval[1]),:]

  def posmat(self,pos,a=None):
    bin_pos = self.pos2pbi(pos[:,1:3])
    occ = np.zeros((self.n_spatial_bins,self.n_spatial_bins))
    for y in range(0,self.n_spatial_bins):
      for x in range(0,self.n_spatial_bins):
        occ[y,x] = np.sum(np.all(bin_pos == [x,y],axis=1))
    if a != None:
      occ = (a/np.sum(np.sum(occ)))*occ
    occ[np.isnan(occ)] = 0
    return occ

  def pos2pbi(self,pos):
    pbir = np.round(pos/self.spatial_bin_size)
    return np.append([pbir[:,1]],[pbir[:,0]],axis=0).T.astype(int)

  def pos_at_time(self,times):
    return np.append(np.array([times]).T, self.pos[find_closest(self.pos[:,0], times), 1:3], axis=1)

  def prob_n_given_x(self,n,x,f,tau):
    xidx = tuple(x) # TODO: check these coordinates are right way round! Rem. arrays accessed row-column (unlike cart.)
    ngtz = n[n > 0]
    return np.prod([((tau*f[i][xidx])**n[i]/factorial(n[i]))*np.exp(-tau*f[i][xidx]) for i in range(len(ngtz))])

  def prob_x_given_n(self,n,f,tau):
    prob = np.array(
      [[self.prob_n_given_x(n,(y,x),f,tau) for x in range(self.n_spatial_bins)] for y in range(self.n_spatial_bins)]
    )
    C = 1/np.sum(np.sum(prob)) if np.sum(np.sum(prob)) > 0 else 0
    return C*prob

  def ex_n_given_x(self,x,f,tau):
    xidx = tuple(x) # TODO: check these coordinates are right way round! Rem. arrays accessed row-column (unlike cart.)
    return np.array([f[i][xidx]*tau for i in range(len(self.hc))])

  def approx_n_and_pos(self,interval,time_bin_size):
    n_time_bins = int(np.ceil((interval[1]-interval[0])/time_bin_size))
    pos = self.get_pos(interval)
    tidx = np.floor((pos[:,0]-np.min(pos[:,0]))/time_bin_size)
    pbi = self.pos2pbi(np.array([np.mean(pos[tidx==i,1:3],axis=0) for i in range(n_time_bins)]))
    N = np.empty((len(self.hc),n_time_bins))
    for i in range(0,len(self.hc)):
      print("processing neurons: %d / %d\r" % (i, len(self.hc)), end="")
      tspk = self.get_spike_times(i,interval)
      if tspk.size: # check is not empty
        tidx = np.floor((tspk-np.min(tspk))/time_bin_size)
        N[i] = np.array([np.sum(tidx==i) for i in range(n_time_bins)])
      else:
        N[i] = np.zeros(n_time_bins)
    print("processing neurons...COMPLETE.")
    print("all done.")
    return (N, pbi)

  def random_time_pos(self,interval):
    rand = np.random.uniform(interval[0],interval[1])
    pos = self.pos_at_time([rand])
    time = pos[0,0]
    pbi = np.floor(pos[:,1:3]/self.spatial_bin_size).flatten().astype(int)
    return (time,pbi)

decoder = Decoder()

if 0:
  rec = (decoder.maze_epoch[0],decoder.maze_epoch[0]+1)
  truepos = decoder.get_pos(rec)
  trueposmat = decoder.posmat(truepos)
  plt.imshow(trueposmat,origin='lower')
  plt.show()

if 1:
  [time, pos] = decoder.random_time_pos(decoder.maze_epoch)
  interval = decoder.maze_epoch
  print(time)
  print(pos)

  f = decoder.calc_f_2d(interval)

  for i in range(len(f)):
    maxval = np.max(np.max(f[i]))
    if maxval > 0:
      #print('neuron',i,'maxval =',maxval)
      plt.title("Neuron %d" % i)
      plt.imshow(f[i],origin='lower')
      plt.show(block=False) ; plt.pause(0.1)
  plt.close()

  ex_n = decoder.ex_n_given_x((pos[1],pos[0]),f,0.25)
  (n,_) = decoder.approx_n_and_pos((time-0.125,time+0.125),0.25)
  probmat = decoder.prob_x_given_n(n,f,0.25)
  maxval, maxidx = matmax(probmat)
  print(maxval, maxidx)
  plt.imshow(probmat)
  plt.show()
  l1, = plt.plot(n[:,0])
  l2, = plt.plot(ex_n)
  plt.legend([l1,l2],["actual","expected"])
  plt.show()
