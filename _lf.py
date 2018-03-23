import numpy as np
import scipy.io as sio
from scipy.signal import hilbert, gaussian
import matplotlib.pyplot as plt
from bandpass import butter_bandpass_filter

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
eegs = np.array([
  sio.loadmat('Con/EEG/coneeg%02d-%1d-%02d.mat' % (day+1,epoch+1,tetrode+1))
  ['eeg'][0][day][0][epoch][0][tetrode][0]['data'][0].flatten() for tetrode in range(30)
])
for tetrode in range(30):
  eeg_mat = sio.loadmat('Con/EEG/coneeg%02d-%1d-%02d.mat' % (day+1,epoch+1,tetrode+1))
  eeg = eeg_mat['eeg'][0][day][0][epoch][0][tetrode][0]['data'][0].flatten()
  samprate = eeg_mat['eeg'][0][day][0][epoch][0][tetrode][0]['samprate'][0][0][0]
  
  # bandpass eeg data
  #signal = butter_bandpass_filter(eeg, 1, 10, samprate)
  signal = butter_bandpass_filter(eeg, 150, 250, samprate)
  env = np.abs(hilbert(signal))
  sd3 = np.mean(env)+3*np.std(env)
  
  fig = plt.figure()
  for i in range(30):
    start = i*int(samprate/2)
    end = (i+1)*int(samprate/2)
    plt.plot(signal[start:end])
    plt.plot(env[start:end],'r-')
    plt.plot([0,int(samprate/2)],[sd3,sd3])
    plt.ylim([min(signal),max(signal)])
    plt.show(block=False) ; plt.pause(0.01) ; fig.clf()
