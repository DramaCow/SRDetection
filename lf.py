import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import decoder as bd

pos_info = sio.loadmat('Con/conpos01.mat')
epoch = 0
pos_epoch_info = pos_info['pos'][0][0][0][epoch][0][0]
print(pos_epoch_info['data'])
