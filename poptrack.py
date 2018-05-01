import numpy as np
from scipy.special import comb

def compute_ak(p_xi_givenk, brute_thresh, nsamples):
  # Compute the vector of a_k values needed to normalize the Population
  # Tracking model probability distribution. It is equal to the probability
  # of having k active neurons as a function of k assuming neurons are
  # conditionally indepdendent.
  #
  # The algorithm attempts the brute force enumeration for the exact value,
  # then if too large uses an approximate method (importance sampling).
  # See paper for details.
  # 
  # Input:    p_xi_givenk, the matrix of the probabilities that each 
  #               neuron is active given the population rate, p(xi|k)
  #           brute_thresh, the threshold number of patterns above which
  #               brute force enumeration is abandoned (suggest ~1e4 to 1e5)
  #           nsamples, the number of samples per k for importance sampling
  #               (suggest at least 1e4). [ = m in paper]
  # Output:   ak, (N+1)-by-1 vector of ak values

  # number of units
  N = min(p_xi_givenk.shape)

  # find ak
  ak = np.empty(N+1)
  ak[0] = 1 # silent state
  ak[N] = 1 # all ON state

  for k in range(1,N):
    pvec = p_xi_givenk[:,k]
    nwords_wkactive = comb(N,k) # no. possible words w/ k active neurons

    if np.std(pvec) == 0: # if pvec homogeneous (i.e. all the same)
      x = pvec[0]
      ak[k] = nwords_wkactive*(x**k)*((1-x)**k)
    else:
      cumsumpword = 0
      for i in range(nsamples):
        onindsk  = np.random.choice(N,k,replace=False) # k random ON neurons
        offindsk = np.setdiff1d(range(0,N),onindsk,assume_unique=True)
        cumsumpword = cumsumpword + np.prod(pvec[onindsk])*np.prod(1-pvec[offindsk])
      ak[k] = cumsumpword*(nwords_wkactive/nsamples)

  return ak

def poptrack(M):
  # Fits population tracking model to binary spike data.
  #
  # Input:    M, binary data matrix (T-by-N)
  # Outputs:  p_k, the population synchrony distribution, p(k)
  #           p_xi_givenk, the probability that each individual neuron is
  #               active given the population rate, p(xi|k)
  #           p_xi, the mean firing rate of each neuron, p(xi), useful for
  #               building an independent neuron model
  #
  # The data M should be in the form of a T-by-N matrix where each row
  # is a different time sample (T total) and each column is a different
  # neuron (N total).
  # The model fitting is two-step, one step for the population synchrony
  # distribution p(k) and one for the conditional probability of each neuron
  # being active given the population synchrony level, p(xi|k).
  # Both steps include a Bayesian regularization that requires specifying at
  # least two hyperparameters: 1) the concentration parameter for the
  # dirichlet prior for p(k), alpha. 2) the scaling factor on the variance of
  # the beta prior for each p(xi|k), prior_var_scale.

  N, T = M.shape

  # hyper parameters
  alpha = 1e-2

  # per-neuron mean firing rates
  p_xi = np.mean(M, 1)

  # population rate (poprate)
  n_on = np.sum(M, 0) # number of neurons active each time bin
  nhist,_ = np.histogram(n_on, bins=N, range=[0,N])
  nhist = nhist + alpha
  p_k = nhist/sum(nhist)

  # pattern rate (prate) given poprate
  p_xi_givenk = np.empty((N, N+1))
  p_xi_givenk[:,0] = 0 # probability neuron on given no neurons are on
  p_xi_givenk[:,N] = 1 # probability neuron on given all neurons on

  for k in range(1, N):
    ts = np.where(n_on == k)[0]     # indices of timesteps w/ exactly k active neurons
    npop = len(ts)                  # no. timesteps w/ exactly k active neurons (T_k in paper)
    nactive_vec = np.sum(M[:,ts],1) # no. each neuron is active when exactly k active neurons (d_i,k in paper)

    mu = k/N               # naive prior mean
    sigma2 = 0.5*mu*(1-mu) # naive prior variance
    beta1 = (mu/sigma2)*(mu-mu**2-sigma2) # beta prior hyperparameters
    beta2 = beta1*(1/mu-1)

    p_xi_givenk[:,k] = (nactive_vec+beta1)/(beta1+beta2+npop)

  return p_k, p_xi_givenk, compute_ak(p_xi_givenk,int(1e5),int(1e5))

def prob_x_given_poptrack(x,params):
  p_k,p_xi_givenk,a_k = params
  k = np.sum(x)
  p_x = (p_k[k]/a_k[k])*(np.prod([p_xi_givenk[i,k]**x[i] * (1-p_xi_givenk[i,k])**(1-x[i]) for i in range(len(x))]))
  return p_x
