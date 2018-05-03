import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from math import factorial
from astar import astar, create_grid, Node
from dijkstras import shortest_path_mat

from scipy import ndimage

def plot_fr_field_2d(f,delay=None):
  for i in range(len(f)):
    maxval = np.max(np.max(f[i]))
    if maxval > 0:
      plt.title("Neuron %d" % i)
      plt.imshow(f[i],origin='lower')
      if delay:
        plt.show(block=False) ; plt.pause(delay)
      else:
        #plt.show()
        plt.savefig('occfr%d.png' % i)
  plt.close()

def plot_fr_field_1d(f,delay=None):
  for i in range(len(f)):
    maxval = np.max(f[i])
    if maxval > 0:
      plt.title("Neuron %d" % i)
      plt.plot(f[i]/maxval)
      if delay:
        plt.show(block=False) ; plt.pause(delay) ; plt.cla()
      else:
        plt.show()
  plt.close()

def find_closest(A, targets):
  inds = np.clip(A.searchsorted(targets), 1, len(A)-1)
  left = A[inds-1]
  right = A[inds]
  return inds-(targets-left < right-targets)

def sum_neighbours(M,i,j):
  (h,w) = M.shape
  val = 0
  if i > 0:   val += M[j,i-1]
  if i < w-1: val += M[j,i+1]
  if j > 0:   val += M[j-1,i]
  if j < h-1: val += M[j+1,i]
  return val

def matmax(M):
  (h,w) = M.shape
  maxval = -np.inf
  maxidx = [-1,-1]
  for j in range(h):
    for i in range(w):
      if M[j,i] > maxval:
        maxval = M[j,i]
        maxidx = np.array([j,i])
  return maxval, maxidx

def vecmax(vec):
  return np.max(vec), np.argmax(vec)

class Decoder:
  def __init__(self,pos,spk,spatial_bin_length,lin_point=None):
    self.pos = pos
    self.spk = spk

    # discretisation parameters
    #self.map_dimensions = np.array([17,31])
    #map_size = np.array([max(self.pos[:,2]),max(self.pos[:,1])])
    #self.spatial_bin_size = map_size/(self.map_dimensions-1)
    self.spatial_bin_length = spatial_bin_length
    self.spatial_bin_size = np.array([spatial_bin_length, self.spatial_bin_length])
    map_size = np.array([max(self.pos[:,2]),max(self.pos[:,1])])
    self.map_dimensions = np.ceil((map_size/self.spatial_bin_size)+1).astype(int)
    print(self.map_dimensions)

    # calculate prior and determine tranversable areas of map
    self.p_x = self.occ_mat(self.pos,1) # prior probability (occupancy normalised to probability)
    posmask = self.p_x > 0              # ah pos-bin that was accessed marked accessible
    self.accmask = np.array(
      [[posmask[j,i] or sum_neighbours(posmask,i,j)>2 for i in range(posmask.shape[1])] for j in range(posmask.shape[0])]
    )

    # only consider the biggest island (to prevent invalid path in shortest path calculation)
    labeled_array, num_features = ndimage.label(self.accmask, np.ones((3,3)))
    label_counts = np.array([np.sum(labeled_array==i) for i in range(1,num_features+1)])
    self.accmask = (labeled_array==np.argmax(label_counts)+1).astype(int)
    #plt.imshow(self.accmask)
    #plt.show()

    # convert spatial inform to 1D if linearisation function has been provided
    if lin_point is not None:
      self.dist1d = np.round(shortest_path_mat(self.accmask,lin_point))#.astype(int)
      self.lim1d = np.nanmax(self.dist1d[np.isfinite(self.dist1d)]).astype(int)+1
      self.p_x1d = self.occ_vec(self.pos,1) # prior probability (occupancy normalised to probability)
      '''
      for r in range(self.dist1d.shape[0]):
        for c in range(self.dist1d.shape[1]):
          if np.isnan(self.dist1d[r,c]):
            print('.',end=' ')
          elif np.isinf(self.dist1d[r,c]):
            print('X',end=' ')
          else:
            print('%.0f' % (self.dist1d[r,c]/10),end=' ')
        print()
      print(self.occ_vec(self.pos))
      print(self.p_x1d)
      plt.imshow(self.dist1d,origin='lower')
      plt.show()
      '''

  # ===========================
  # === AUXILIARY FUNCTIONS ===
  # ===========================

  # get positions within interval
  def get_pos(self,interval):
    return self.pos[np.logical_and(interval[0]<=self.pos[:,0], self.pos[:,0]<=interval[1]),:]

  # get spike times within interval
  def get_spike_times(self,i,interval):
    return self.spk[i][np.logical_and(interval[0]<=self.spk[i], self.spk[i]<=interval[1])]

  # get number of spikes for each neuron within interval
  def get_n(self,interval):
    return np.array([self.get_spike_times(i,interval).size for i in range(len(self.spk))])

  # maintains input times but uses nearest positions
  def approx_pos_at_time(self,times):
    return np.append(np.array([times]).T, self.pos[find_closest(self.pos[:,0], times), 1:3], axis=1)

  # uses times associated with nearest positions
  def nearest_pos_at_time(self,times):
    return self.pos[find_closest(self.pos[:,0], times),:]

  # convert position to 2D co-ordinate
  def pos_to_x(self,pos):
    pos_r = np.append([pos[:,1]],[pos[:,0]],axis=0).T
    return np.round(pos_r/self.spatial_bin_size).astype(int)

  # convert 2D co-ordinate to 1D co-ordinate
  def x_to_x1d(self,xs):
    if xs.ndim == 2:
      return np.array(list(map(lambda x: self.dist1d[tuple(x)], xs)))
    else: #xs.ndim == 1
      return self.dist1d[tuple(xs)]

  # convert position to 1D co-ordinate
  def pos_to_x1d(self,pos):
    return self.x_to_x1d(self.pos_to_x(pos))

  # 2D occupancy map
  def occ_mat(self,pos,a=None):
    bin_pos = self.pos_to_x(pos[:,1:3])
    occ = np.zeros(self.map_dimensions)
    for j in range(0,self.map_dimensions[0]):
      for i in range(0,self.map_dimensions[1]):
        occ[j,i] = np.sum(np.all(bin_pos == [j,i],axis=1))
    if a != None:
      occ = (a/np.sum(np.sum(occ)))*occ
    occ[np.isnan(occ)] = 0
    return occ

  # 1D occupancy map
  def occ_vec(self,pos,a=None):
    rows,cols = self.dist1d.shape
    occ = self.occ_mat(pos)
    vec = np.array([
      sum([occ[r,c] for r in range(rows) for c in range(cols) if self.dist1d[r,c]==dist])
        for dist in range(self.lim1d)
    ])
    if a != None:
      vec = (a/np.sum(vec))*vec
    vec[np.isnan(vec)] = 0
    return vec

  # =====================================
  # === PARAMETER GENERATOR FUNCTIONS ===
  # =====================================

  # returns occupancy normalised 2D fire rate map for each neuron
  def calc_f_2d(self,interval):
    # calculate (approximate) occupancy (total time spent in location bins)
    print("calculating 2D occupancy map...", end="")
    occ = self.occ_mat(self.get_pos(interval),interval[1]-interval[0]) # count no. ticks in each pos-bin & norm. by total dur.
    posmask = occ > 0                                                  # ah pos-bin that was accessed marked accessible
    print("COMPLETE.")

    # approximate position of neuron firing
    f = np.empty((len(self.spk),self.map_dimensions[0],self.map_dimensions[1]))
    for i in range(len(self.spk)):
      print("processing neurons: %d / %d\r" % (i, len(self.spk)), end="")
      tspk = self.get_spike_times(i,interval)            # get times neuron spiked during interval
      f[i] = self.occ_mat(self.approx_pos_at_time(tspk)) # count number of spikes occuring at each pos-bin
      f[i][posmask] = f[i][posmask] / occ[posmask]       # fr = spike count / time spent in each pos-bin
      f[i] = gaussian_filter(f[i],1.0)*self.accmask      # blur a little
    print("processing neurons...COMPLETE.")
    print("all done.")
    return f

  # returns occupancy normalised 1D fire rate map for each neuron
  def calc_f_1d(self,interval):
    # calculate (approximate) occupancy (total time spent in location bins)
    print("calculating 1D occupancy map...", end="")
    occ = self.occ_vec(self.get_pos(interval),interval[1]-interval[0]) # count no. ticks in each pos-bin & norm. by total dur.
    posmask = occ > 0
    print("COMPLETE.")

    # approximate position of neuron firing
    f = np.empty((len(self.spk),len(occ)))
    for i in range(len(self.spk)):
      print("processing neurons: %d / %d\r" % (i, len(self.spk)), end="")
      tspk = self.get_spike_times(i,interval)            # get times neuron spiked during interval
      f[i] = self.occ_vec(self.approx_pos_at_time(tspk)) # count number of spikes occuring at each pos-bin
      f[i][posmask] = f[i][posmask] / occ[posmask]       # count number of spikes occuring at each pos-bin
      f[i] = gaussian_filter(f[i],1.0)                   # blur a little
    print("processing neurons...COMPLETE.")
    print("all done.")
    return f
    
  # ===========================
  # === OUTPUT FUNCTIONS 2D ===
  # ===========================

  # per-position likelihood
  def prob_n_given_x(self,n,x,f,tau):
    xidx = tuple(x)
    ngtz = n[n > 0]
    return np.prod([((tau*f[i][xidx])**n[i]/factorial(n[i]))*np.exp(-tau*f[i][xidx]) for i in range(len(ngtz))])

  # likelihood
  def prob_n_given_X(self,n,f,tau):
    return np.array(
      [[self.prob_n_given_x(n,(j,i),f,tau) for i in range(self.map_dimensions[1])] for j in range(self.map_dimensions[0])]
    )

  # posterior
  def prob_X_given_n(self,n,f,tau):
    prob = self.p_x*self.prob_n_given_X(n,f,tau)
    #prob = self.prob_n_given_X(n,f,tau)
    #prob = self.p_x
    C = 1/np.sum(np.sum(prob)) if np.sum(np.sum(prob)) > 0 else 0
    return C*prob

  # expectation
  def ex_n_given_x(self,x,f,tau):
    xidx = tuple(x)
    return np.array([f[i][xidx]*tau for i in range(len(self.spk))])

  # ===========================
  # === OUTPUT FUNCTIONS 1D ===
  # ===========================

  # per-position likelihood
  def prob_n_given_x1d(self,n,x1d,f,tau):
    ngtz = n[n > 0]
    return np.prod([((tau*f[i][x1d])**n[i]/factorial(n[i]))*np.exp(-tau*f[i][x1d]) for i in range(len(ngtz))])

  # likelihood
  def prob_n_given_X1d(self,n,f,tau):
    return np.array(
      [self.prob_n_given_x1d(n,x1d,f,tau) for x1d in range(self.lim1d)]
    )

  # posterior
  def prob_X1d_given_n(self,n,f,tau):
    prob = self.p_x1d*self.prob_n_given_X1d(n,f,tau)
    C = 1/np.sum(prob) if np.sum(prob) > 0 else 0
    return C*prob

  # expectation
  def ex_n_given_x1d(self,x1d,f,tau):
    return np.array([f[i][x1d]*tau for i in range(len(self.spk))])

  # ==================
  # === CONVERSION ===
  # ==================

  # 2D to 1D conversion
  def prob_X2_to_X1(self,p_x):  
    prob = np.zeros(self.lim1d)
    rows,cols = p_x.shape
    for r in range(rows):
      for c in range(cols):
        x1d = self.x_to_x1d(np.array([r,c]))
        if not np.isnan(x1d) and np.isfinite(x1d):
          prob[int(x1d)] += p_x[r,c]
    C = 1/np.sum(prob) if np.sum(prob) > 0 else 0
    return C*prob

  # ======================
  # === TEST FUNCTIONS ===
  # ======================

  # returns number of spikes and position within an interval
  def approx_n_and_x(self,interval,time_bin_size):
    num_time_bins = int(np.round((interval[1]-interval[0])/time_bin_size))
    pos = self.get_pos(interval)
    tidx = np.floor((pos[:,0]-np.min(pos[:,0]))/time_bin_size)
    x = self.pos_to_x(np.array([np.mean(pos[tidx==i,1:3],axis=0) for i in range(num_time_bins)]))
    n = np.empty((len(self.spk),num_time_bins))
    for i in range(0,len(self.spk)):
      #print("processing neurons: %d / %d\r" % (i, len(self.spk)), end="")
      tspk = self.get_spike_times(i,interval)
      if tspk.size: # check is not empty
        tidx = np.floor((tspk-np.min(tspk))/time_bin_size)
        n[i] = np.array([np.sum(tidx==i) for i in range(num_time_bins)])
      else:
        n[i] = np.zeros(num_time_bins)
    #print("processing neurons...COMPLETE.")
    #print("all done.")
    return (n, x)

  # picks a random time and associated position within an interval
  def random_t_x(self,interval):
    rand = np.random.uniform(interval[0],interval[1])
    pos = self.nearest_pos_at_time([rand])
    t = pos[0,0]
    x = self.pos_to_x(pos[:,1:3]).flatten()
    return (t,x)
