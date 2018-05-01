import numpy as np

def ind_model(M):
  N,T = M.shape
  count = np.sum(M,axis=1)
  p = count/T
  return p

def prob_x_given_ind(x,params):
  on  = x
  off = 1-x
  prob = np.prod(params[on])*np.prod(1-params[off])
  return prob
