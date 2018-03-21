import numpy as np
from math import floor
import h5py
import decoder as bd

duration = 300.0
dt = 1e-3

with h5py.File('data_gb/Achilles_10252013_sessInfo.mat', 'r') as f:
  # epoch info
  pre_epoch  = f['sessInfo']['Epochs']['PREEpoch'][:].flatten()
  maze_epoch = f['sessInfo']['Epochs']['MazeEpoch'][:].flatten()
  post_epoch = f['sessInfo']['Epochs']['POSTEpoch'][:].flatten()
  sessDuration = float(f['sessInfo']['Epochs']['sessDuration'][0])

  # position info
  TwoDLocation = np.array(f['sessInfo']['Position']['TwoDLocation'])
  TimeStamps = f['sessInfo']['Position']['TimeStamps'][:].flatten()
  MazeType = f['sessInfo']['Position']['MazeType']
  valididx = ~np.isnan(TwoDLocation[0]) & ~np.isnan(TwoDLocation[1])
  TwoDLocation = TwoDLocation[:,valididx]
  TwoDLocation[0] = TwoDLocation[0] - np.min(TwoDLocation[0])
  TwoDLocation[1] = TwoDLocation[1] - np.min(TwoDLocation[1])
  TimeStamps = TimeStamps[valididx]

  # spike info
  IntIDs = f['sessInfo']['Spikes']['IntIDs'][0].astype(int)     # (putative) interneuron IDs
  PyrIDs = f['sessInfo']['Spikes']['PyrIDs'][0].astype(int)     # (putative) pyramidal cell IDs
  SpikeTimes = f['sessInfo']['Spikes']['SpikeTimes'][0]         # spike times
  SpikeIDs = f['sessInfo']['Spikes']['SpikeIDs'][0].astype(int) # cluster IDs of corresponding spike

  # (multiple) spike info 
  #ShankIDs = list(sorted(set(np.floor(np.concatenate([IntIDs, PyrIDs])/100))))
  #MspikeIDs = np.floor(SpikeIDs/100)

  # rename neurons
  for i, ni in enumerate(np.concatenate([IntIDs, PyrIDs])):     # rename neuron IDs to be 0,1,2,...
  #for i, ni in enumerate(ShankIDs):     # rename neuron IDs to be 0,1,2,...
    IntIDs[IntIDs == ni] = i
    PyrIDs[PyrIDs == ni] = i
    SpikeIDs[SpikeIDs == ni] = i
  N = i + 1

  pos = np.array([TimeStamps,TwoDLocation[0],TwoDLocation[1]]).T
  spk = np.array([SpikeTimes[SpikeIDs==i] for i in PyrIDs])
  #spk = np.array([SpikeTimes[SpikeIDs==i] for i in np.concatenate([IntIDs, PyrIDs])])
