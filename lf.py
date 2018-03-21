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
epoch_info = np.array([min(pos_info[:,0]),max(pos_info[:,0])])

# spike info
spk_mat = sio.loadmat('Con/conspikes%02d.mat' % (day+1))
spk_info = np.array([units[0]['data'][0] for tetrode in spk_mat['spikes'][0][day][0][epoch][0] for units in tetrode[0] if len(units) > 0])
spk = np.array([unit[:,0] for unit in spk_info])
