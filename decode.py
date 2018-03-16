import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage.filters import gaussian_filter
from math import factorial
from astar import astar, create_grid, Node

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
  return (maxval, maxidx)

class Decoder:
  def __init__(self):
    # === LOAD ===

    # load necessary mat files
    spk = sio.loadmat('Tmaze_spiking_data.mat')
    loc = sio.loadmat('Tmaze_location_data.mat')
    rst = sio.loadmat('rippspin-times-FGHIJ.mat')
    
    # === PREPROCESS ===

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

    # === PARAMETERS ===

    # discretisation parameters
    self.map_dimensions = np.array([32,32])
    self.spatial_bin_size = np.amax(self.pos[:,1:3],0)/(self.map_dimensions-1)
    self.spatial_bin_size = np.array([self.spatial_bin_size[1], self.spatial_bin_size[0]])

    # calculate prior and determine tranversable areas of map
    self.p_x = self.occ_mat(self.pos,1) # prior probability (occupancy normalised to probability)
    posmask = self.p_x > 0              # ah pos-bin that was accessed marked accessible
    self.accmask = np.array(
      [[posmask[j,i] or sum_neighbours(posmask,i,j)>2 for i in range(posmask.shape[1])] for j in range(posmask.shape[0])]
    )

  def calc_f_2d(self,interval):
    # calculate (approximate) occupancy (total time spent in location bins)
    print("calculating 2D occupancy map...", end="")
    occ = self.occ_mat(self.get_pos(interval),interval[1]-interval[0]) # count no. ticks in each pos-bin & norm. by total dur.
    posmask = occ > 0                                                  # ah pos-bin that was accessed marked accessible
    print("COMPLETE.")

    # approximate position of neuron firing
    f = np.empty((len(self.hc),self.map_dimensions[0],self.map_dimensions[1]))
    for i in range(len(self.hc)):
      print("processing neurons: %d / %d\r" % (i, len(self.hc)), end="")
      tspk = self.get_spike_times(i,interval)      # get times neuron spiked during interval
      f[i] = self.occ_mat(self.pos_at_time(tspk))  # count number of spikes occuring at each pos-bin
      f[i][posmask] = f[i][posmask] / occ[posmask] # fr = spike count / time spent in each pos-bin
      #oldmax = np.max(np.max(f[i]))
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

  def pos_at_time(self,times):
    return np.append(np.array([times]).T, self.pos[find_closest(self.pos[:,0], times), 1:3], axis=1)

  def pos_to_x(self,pos):
    pos_r = np.append([pos[:,1]],[pos[:,0]],axis=0).T
    return np.round(pos_r/self.spatial_bin_size).astype(int)

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
    C = 1/np.sum(np.sum(prob)) if np.sum(np.sum(prob)) > 0 else 0
    return C*prob

  def ex_n_given_x(self,x,f,tau):
    xidx = tuple(x)
    return np.array([f[i][xidx]*tau for i in range(len(self.hc))])

  def approx_n_and_x(self,interval,time_bin_size):
    num_time_bins = int(np.ceil((interval[1]-interval[0])/time_bin_size))
    pos = self.get_pos(interval)
    if len(pos) == 0:
      print(interval)
      print('EMPTY?!')
    tidx = np.floor((pos[:,0]-np.min(pos[:,0]))/time_bin_size) # TODO: problem with min
    x = self.pos_to_x(np.array([np.mean(pos[tidx==i,1:3],axis=0) for i in range(num_time_bins)]))
    n = np.empty((len(self.hc),num_time_bins))
    for i in range(0,len(self.hc)):
      #print("processing neurons: %d / %d\r" % (i, len(self.hc)), end="")
      tspk = self.get_spike_times(i,interval)
      if tspk.size: # check is not empty
        tidx = np.floor((tspk-np.min(tspk))/time_bin_size)
        n[i] = np.array([np.sum(tidx==i) for i in range(num_time_bins)])
      else:
        n[i] = np.zeros(num_time_bins)
    #print("processing neurons...COMPLETE.")
    #print("all done.")
    return (n, x)

  def random_t_x(self,interval):
    rand = np.random.uniform(interval[0],interval[1])
    pos = self.pos_at_time([rand])
    t = pos[0,0]
    x = self.pos_to_x(pos[:,1:3]).flatten()
    return (t,x)

decoder = Decoder()
#plot_intervals(decoder.rip)
f = decoder.calc_f_2d(decoder.maze_epoch)

#for i in range(len(f)):
#  maxval = np.max(np.max(f[i]))
#  if maxval > 0:
#    plt.title("Neuron %d" % i)
#    plt.imshow(f[i],origin='lower')
#    plt.show(block=False) ; plt.pause(0.1)
#plt.close()

fig = plt.figure()
window = 0.5
for _ in range(100):
  # generate test data
  [t,x] = decoder.random_t_x(decoder.maze_epoch)
  (n,_) = decoder.approx_n_and_x((t-window/2,t+window/2),window)
  n_ex = decoder.ex_n_given_x(x,f,window)
  print('t = %.2f, x =' % t, x, end=', ')

  # calculate argmax probability
  probmat = decoder.prob_X_given_n(n,f,window)
  [argmax_p, x_] = matmax(probmat)
  print('prob = %.4f, x_ =' % argmax_p, x_, end=', ')

  #likelihood = decoder.prob_n_given_X(n,f,window)
  #plt.imshow(likelihood,origin='lower')
  #plt.show()
  
  # plots
  plt.subplot(121)
  plt.imshow(probmat, cmap='gray', origin='lower')
  path_length = 0
  if np.all(x == x_):
    plt.scatter(x[1],x[0],color='lime')
  else:
    plt.scatter(x[1], x[0], color='r')
    plt.scatter(x_[1], x_[0], color='b')
    path = astar(tuple(x),tuple(x_),create_grid(decoder.accmask))
    path_points = np.array([list(p.point) for p in path])
    path_length = path[-1].G
    plt.plot(path_points[:,1],path_points[:,0],'y-')
  print(path_length)
  plt.subplot(122)
  l1, = plt.plot(n, 'r')
  l2, = plt.plot(n_ex, 'b')
  plt.xlabel('neuron')
  plt.title('spike count (within %.2fs window)' % window)
  plt.legend([l1,l2],["actual","expected"])
  plt.show(block=False) ; plt.pause(1) ; fig.clf()
  #plt.show()
