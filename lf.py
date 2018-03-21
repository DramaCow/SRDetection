import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import decoder as bd

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

# ripple info
tetrode = 0
eeg_mat = sio.loadmat('Con/EEG/coneeg%02d-%1d-%02d.mat' % (day+1,epoch+1,tetrode+1))
eeg = eeg_mat['eeg'][0][day][0][epoch][0][0]['data'][0][0].flatten()
samprate = eeg_mat['eeg'][0][day][0][epoch][0][0]['samprate'][0][0][0]

plt.plot(eeg)
plt.show()
