import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import decoder as bd

experiment = 4 # max is 4

# === POSITION ===
loc = sio.loadmat('data_mj/Tmaze_location_data.mat')
posit_id = chr(ord('F')+experiment)+'positiondata'
pos = loc[posit_id][~np.all(loc[posit_id][:,1:3]==0,1),:] # ignores 0,0 positions (erroneous)
pos[:,0] = pos[:,0]/1e6

# === SPIKING ===
spd = sio.loadmat('data_mj/Tmaze_spiking_data.mat')
cells_id = chr(ord('F')+experiment)+'cells'
spk = [spk['tspk'].flatten() for spk in spd[cells_id][spd[cells_id]['area'] == 'hc']] # only look at hippocampal cells

# === EPOCHS ===
epoch_id = experiment
pre_epoch  = np.concatenate(spd['epochs'][epoch_id,1:3]).ravel()
maze_epoch = np.concatenate(spd['epochs'][epoch_id,3:5]).ravel()
maze_epoch = [max(maze_epoch[0],min(pos[:,0])), min(maze_epoch[1],max(pos[:,0]))]
post_epoch = np.concatenate(spd['epochs'][epoch_id,5:7]).ravel()

# === RIPPLES ===
# get ripple periods (+/- 100ms around detected SPW-R peak times)
rst = sio.loadmat('data_mj/rippspin-times-FGHIJ.mat')
rippl_id = chr(ord('F')+experiment)+'rip'
rip = bd.merge_intervals(np.append(rst[rippl_id]-0.1, rst[rippl_id]+0.1, axis=1)) 

spatial_bin_size = np.array([12,12])
