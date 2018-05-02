import numpy as np
import matplotlib.pyplot as plt
import decoder as bd
from astar import astar, create_grid, Node

# === DATA SOURCE ===
from lf import pos, spk, maze_epoch, spatial_bin_length
lin_point = np.array([35,50])

# === DECODER ===
decoder = bd.Decoder(pos,spk,spatial_bin_length)
f = decoder.calc_f_2d(maze_epoch)
#bd.plot_fr_field_2d(f,1.0)
#decoder = bd.Decoder(pos,spk,spatial_bin_length,lin_point)
#f = decoder.calc_f_1d(maze_epoch)
#bd.plot_fr_field_1d(f,1.0)

# === DETERMINE LIN-POINT ===
#accmask = decoder.accmask.astype(int); accmask[35,50] = 2
#print(accmask)
#plt.imshow(accmask,origin='lower')
#plt.show()

points = np.array([np.array([y,x]) for y in range(decoder.accmask.shape[0]) for x in range(decoder.accmask.shape[1]) if decoder.accmask[y,x]==1])

plt.imshow(decoder.accmask, cmap='gray', origin='lower')
plt.show()

# === TEST 2D ===
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
  if not decoder.accmask[tuple(x_)]:
    for row in probmat:
      for p in row:
        print(p,end=' ')
      print()
    x_ = points[np.random.randint(len(points))]
    #print('inaccessible')

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
  print('error = %.2f' % path_lengths[p])
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
'''
# === TEST 1D ===
fig = plt.figure()
window = 0.5
disparity = np.zeros(20)
for p in range(len(disparity)):
  # generate test data
  [t,x] = decoder.random_t_x(maze_epoch)
  x1d = decoder.x_to_x1d(x)
  print('t = %.2fs, x1d =' % t, x1d, end=', ')
  (n,_) = decoder.approx_n_and_x((t-window/2,t+window/2),window)
  n_ex = decoder.ex_n_given_x1d(x1d,f,window)

  # calculate argmax probability
  probvec = decoder.prob_X1d_given_n(n,f,window)
  [argmax_p, x1d_] = bd.vecmax(probvec)
  print('prob = %.3f, x1d_ =' % argmax_p, x1d_,end=', ')
  disparity[p] = np.abs(x1d-x1d_)

  # plots
  l1, = plt.plot(probvec, 'k-')
  if x1d == x1d_:
    l2 = plt.axvline(x=x1d_,color='lime')
    plt.legend([l1,l2],['posterior','correct prediction'])
  else:
    l2 = plt.axvline(x=x1d_,color='b')
    l3 = plt.axvline(x=x1d,color='r')
    plt.legend([l1,l2,l3],['posterior','prediction','actual'])
  plt.xlabel('distance from reward arm (number of cells)')
  plt.ylabel('probability')
  plt.title('Prediction of distance from reward arm')
  print('error = %.2f' % disparity[p])
  plt.show(block=False) ; plt.pause(1) ; fig.clf()
  #plt.show()

print('average path length = %.2f' % np.mean(disparity))
print('number of close results = %d/%d' % (np.sum(disparity<10),len(disparity)))
print('min path length = %.2f' % np.min(disparity))
print('max path length = %.2f' % np.max(disparity))
'''
