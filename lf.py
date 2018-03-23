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
tetrodes = range(1)
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

rips = spw_r_detect(eegs,samprates)
print(rips)

'''
for samprate,signal,env,mean,sd3,large,peak in zip(samprates,signals,envs,means,sd3s,larges,peaks):
  fig = plt.figure()
  for i in range(30):
    start = i*int(samprate/2)
    end = (i+1)*int(samprate/2)
    #plt.plot(signal[start:end])
    #plt.plot(env[start:end],'r-')
    #plt.plot([0,int(samprate/2)],[mean,mean])
    #plt.plot([0,int(samprate/2)],[sd3,sd3])
    #plt.ylim([min(signal),max(signal)])
    plt.plot(large[start:end],'r-')
    plt.plot(2*peak[start:end],'r-')
    plt.show(block=False) ; plt.pause(0.01) ; fig.clf()
'''
