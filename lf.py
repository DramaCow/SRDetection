import numpy as np
import scipy.io as sio
from scipy.signal import hilbert, gaussian
import matplotlib.pyplot as plt
from spw_r import spw_r_detect

day = 0   # int in [0,5]
epoch = 0 # int in [0,4]

# pos info
pos_mat = sio.loadmat('Con/conpos%02d.mat' % (day+1))
pos_info = pos_mat['pos'][0][day][0][epoch][0]['data'][0]
pos = pos_info[:,0:3]

# epoch info
maze_epoch = np.array([min(pos_info[:,0]),max(pos_info[:,0])])

# spike info
spk_mat = sio.loadmat('Con/conspikes%02d.mat' % (day+1))
spk_info = np.array([units[0]['data'][0] for tetrode in spk_mat['spikes'][0][day][0][epoch][0] for units in tetrode[0] if units.size > 0])
spk = np.array([unit[:,0] for unit in spk_info if unit.size > 0])

# eeg (lfp) info
tetrodes = range(4)
eegs = np.array([
  sio.loadmat('Con/EEG/coneeg%02d-%1d-%02d.mat' % (day+1,epoch+1,tetrode+1))
  ['eeg'][0][day][0][epoch][0][tetrode][0]['data'][0].flatten() for tetrode in tetrodes
])
starttimes = np.array([
  sio.loadmat('Con/EEG/coneeg%02d-%1d-%02d.mat' % (day+1,epoch+1,tetrode+1))
  ['eeg'][0][day][0][epoch][0][tetrode][0]['starttime'][0][0][0] for tetrode in tetrodes
])
samprates = np.array([
  sio.loadmat('Con/EEG/coneeg%02d-%1d-%02d.mat' % (day+1,epoch+1,tetrode+1))
  ['eeg'][0][day][0][epoch][0][tetrode][0]['samprate'][0][0][0] for tetrode in tetrodes
])
rips,sigs,envs = spw_r_detect(eegs,samprates)

lims = np.array([max(np.abs(np.min(sig)),np.abs(np.max(sig))) for sig in sigs])
sigs = np.array([sig/(2*lim) for (sig,lim) in zip(sigs,lims)])
envs = np.array([env/(2*lim) for (env,lim) in zip(envs,lims)])

fig = plt.figure()
for i in np.arange(0,3000,5):
  ax = plt.axes()
  for j,(samprate,sig,env) in enumerate(zip(samprates,sigs,envs)):
    start = i
    end = i+int(samprate/2)

    rips_visible = rips[np.logical_and(rips[:,1]>=start,rips[:,0]<=end)]
    for rip in rips_visible:
      ax.axvspan(rip[0], rip[1], alpha=0.3)

    plt.plot(range(start,end),sig[start:end]+j,'k-')
    plt.plot(range(start,end),env[start:end]+j,'r-')
    plt.xlim([start,end])
    plt.ylim([-0.5,len(tetrodes)-0.5])
  plt.show(block=False) ; plt.pause(0.1) ; fig.clf()
