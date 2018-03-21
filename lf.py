import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import decoder as bd

day = 0   # int in [0,5]
epoch = 0 # int in [0,4]

# pos info
pos_mat = sio.loadmat('Con/conpos%02d.mat' % (day+1))
pos_info = pos_mat['pos'][0][day][0][epoch][0][0]

# spike info
spk_mat = sio.loadmat('Con/conspikes%02d.mat' % (day+1))
spk_info = spk_mat['spikes'][0][day][0][epoch][0]
spikes = np.array([units[0]['data'][0] for tetrode in spk_info for units in tetrode[0] if len(units) > 0])
