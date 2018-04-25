import numpy as np
import matplotlib.pyplot as plt
import decoder as bd
from astar import astar, create_grid, Node
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

# === DATA SOURCE ===
from lf import pos, spk, maze_epoch, spatial_bin_length, rips
np.loadtxt('ripples.dat',rips)
lin_point = np.array([35,50])

# === DECODER ===
decoder = bd.Decoder(pos,spk,spatial_bin_length,lin_point)
f = decoder.calc_f_2d(maze_epoch)

rips = rips[[sum(decoder.get_n((rip[0],rip[1]))>0)>=5 for rip in rips]]
#rip = max(rips,key=lambda rip: rip[1]-rip[0])
for rip in rips:
  print(rip)
  num_bins = np.round((rip[1]-rip[0])/15e-3).astype(int)
  start = rip[0]
  end = rip[0] + 15e-3*num_bins
  sloc = np.empty(0)
  dbin = np.empty(0)
  if sum(decoder.get_n((start,end))>0) >= 5: # arbitrary threshold requirement
    leg = []
    lbl = []
    for i in range(num_bins):
      n = decoder.get_n((start+15e-3*i,start+15e-3*(i+1)))
      if np.any(n>0):
        p2 = decoder.prob_X_given_n(n,f,15e-3)
        p1 = decoder.prob_X2_to_X1(p2)
        l, = plt.plot(p1)
        leg.append(l)
        lbl.append('bin #%d' % i)
        samples = np.random.choice(int(decoder.lim1d),size=100,p=p1)
        sloc = np.append(sloc,samples)
        dbin = np.append(dbin,i*np.ones(100))
    plt.legend(leg,lbl)
    plt.show()
    plt.clf()
  print(sloc, dbin)
  regr = linear_model.LinearRegression()
  regr.fit(sloc.reshape(-1,1), dbin)
  pred = regr.predict(sloc.reshape(-1,1))
  r2 = r2_score(dbin,pred)
  print(r2)
  plt.plot(sloc,dbin,'k.')
  plt.plot(sloc,pred,'r-')
  plt.show()
