import numpy as np
import matplotlib.pyplot as plt
import decoder as bd
from astar import astar, create_grid, Node

def example_linearise():
  print('hello')

# === DATA SOURCE ===
from mj import pos, spk, maze_epoch, spatial_bin_size

# === DECODER ===
decoder = bd.Decoder(pos,spk,spatial_bin_size,example_linearise)
#decoder = bd.Decoder(pos,spk,spatial_bin_size)
f = decoder.calc_f_2d(maze_epoch)
#bd.plot_fr_field(f,1.0)

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
