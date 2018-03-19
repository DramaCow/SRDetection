import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import decoder as bd
from astar import astar, create_grid, Node

experiment = 4 # max is 4

# === POSITION ===
loc = sio.loadmat('data_mj/Tmaze_location_data.mat')
posit_id = chr(ord('F')+experiment)+'positiondata'
pos = loc[posit_id][~np.all(loc[posit_id][:,1:3]==0,1),:] # ignores 0,0 positions (erroneous)
pos[:,0] = pos[:,0]/1e6

# === SPIKING ===
spk = sio.loadmat('data_mj/Tmaze_spiking_data.mat')
cells_id = chr(ord('F')+experiment)+'cells'
hc = [hc['tspk'].flatten() for hc in spk[cells_id][spk[cells_id]['area'] == 'hc']] # only look at hippocampal cells

# === EPOCHS ===
epoch_id = experiment
pre_epoch  = np.concatenate(spk['epochs'][epoch_id,1:3]).ravel()
maze_epoch = np.concatenate(spk['epochs'][epoch_id,3:5]).ravel()
maze_epoch = [max(maze_epoch[0],min(pos[:,0])), min(maze_epoch[1],max(pos[:,0]))]
post_epoch = np.concatenate(spk['epochs'][epoch_id,5:7]).ravel()

# === RIPPLES ===
rst = sio.loadmat('data_mj/rippspin-times-FGHIJ.mat')
rippl_id = chr(ord('F')+experiment)+'rip'
rip = bd.merge_intervals(np.append(rst[rippl_id]-0.1, rst[rippl_id]+0.1, axis=1)) # get ripple periods (+/- 100ms around detected SPW-R peak times)
#bd.plot_intervals(rip)

# === DECODER ===
decoder = bd.Decoder(pos,hc)
f = decoder.calc_f_2d(maze_epoch)
#bd.plot_fr_field(f,0.1)

# === TEST ===
fig = plt.figure()
window = 0.5
path_lengths = np.zeros(20)
for p in range(len(path_lengths)):
  # generate test data
  [t,x] = decoder.random_t_x(maze_epoch)
  print('t = %.2fs, x =' % t, x, end=', ')
  (n,_) = decoder.approx_n_and_x((t-window/2,t+window/2),window)
  n_ex = decoder.ex_n_given_x(x,f,window)

  # calculate argmax probability
  probmat = decoder.prob_X_given_n(n,f,window)
  [argmax_p, x_] = bd.matmax(probmat)
  print('prob = %.3f, x_ =' % argmax_p, x_, end=', ')

  # plots
  plt.subplot(121)
  plt.imshow(probmat, cmap='gray', origin='lower')
  if np.all(x == x_):
    plt.scatter(x[1],x[0],color='lime')
  else:
    plt.scatter(x[1], x[0], color='r')
    plt.scatter(x_[1], x_[0], color='b')
    path = astar(tuple(x),tuple(x_),create_grid(decoder.accmask))
    path_points = np.array([list(p.point) for p in path])
    path_lengths[p] = path[-1].G
    plt.plot(path_points[:,1],path_points[:,0],'y-')
  print('path length = %.2f' % path_lengths[p])
  plt.subplot(122)
  l1, = plt.plot(n, 'r')
  l2, = plt.plot(n_ex, 'b')
  plt.xlabel('neuron')
  plt.title('spike count (within %.2fs window)' % window)
  plt.legend([l1,l2],["actual","expected"])
  plt.show(block=False) ; plt.pause(1) ; fig.clf()
  #plt.show()
print('average path length = %.2f' % np.mean(path_lengths))
print('number of close results = %d/%d' % (np.sum(path_lengths<10),len(path_lengths)))
print('min path length = %.2f' % np.min(path_lengths))
print('max path length = %.2f' % np.max(path_lengths))
