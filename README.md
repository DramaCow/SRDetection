# SRDetection

Sleep replay detection.

# Bayesian Decoder

Decoder class is defined in ```decoder.py```. Class requires: positions, neuron spike times, and size of spatial bins (for position discretisation)

## Parameter format

### Positions

Nx3 matrix, for N position recordings.

```
[[t1, x1, y1],
 [t2, x2, y2],
      ...
 [tn, xn, yn]]
```

where: ti = time of recording, xi = x-position, yi = y-position.

### Neuron spike times

N-length list of 1D array of spike times, for N neurons being recorded.

## Use

```calc_f_2d``` returns an N-length list of matrices representing the occupancy normalised fire-rate maps, for each N neurons.

```prob_n_given_X``` returns likelihood

```prob_X_given_n``` returns posterior

## Plots

```plot_fr_field(f,delay)``` displays the occupancy normalised fire-rate matrices as images for each neuron. 

# Sharp-wave ripple detections

Sharp-wave ripple detection function defined in ```spw_r.py```; specifically is the function ```spw_r_detect.py```.

## Parameter format

### EEGs

N-length list of 1D signals, for N tetrodes being recorded.

### Samprates

N-length array, where each entry is the sample rate used to record the corresponding tetrode signal.

## Use

Function ```spw_r_detect.py``` returns:

- rips = Mx2 matrix where each row corresponds to a ripple interval. Overlapping intervals are automatically merged.
- signals = bandpassed EEG signals (corresponding to each tetrode), filtered between 150-250Hz using 4th order butterworth filter.
- envs = signal envelopes corresponding to each bandpassed signal. Envelopes were computed using Hilbert transforms (TODO: smooth using gaussian kernels).

## Plots

```plot_ripples(samprates,rips,sigs,envs)``` draws signals, envelopes, and coloured axvspans to indicate intervals determined to be spw-rs.

# Importing data and pre-processing

Importing data is specific to the data format provided by experimentalists. The amount of pre-processing required is also specific to the amount of pre-processing pre-provided by experimental labs. As such, it is easiest to handle importing and pre-processing in seperate scripts specific to each dataset.

- ```mj.py``` pre-processes data from Matt Jones' lab. Provides: positions, spike times, and peak ripple times. Ripples are thus taken to be intervals +/-100ms around peak ripple times (with merging of overlapping intervals).
- ```gb.py``` pre-processes data from that NYU lab, whos name I cannot remember... Provides: positions, spike times, and EEG signals downsampled to 1250Hz? (confirm?) Much noise present in periodic frequency bands - though I have yet to observe this in python (perhaps it was an artifact of Matlab's signal processing? Though, I highly doubt it.)
- ```lf.py``` pre-processes data from Loren M Frank's lab. Provides: positions, spike times, and EEG signals downsampled to ~1500Hz. Whilst There does not seem to be unsual levels of noise, once the EEGs have been bandpassed they do not provide as salient ripple intervals as L.F.'s paper suggests. (TODO: investigate this, but for now I will continue under the assumption that the detected ripple intervals are correct.).

Once useful data has been extracted, they should be stored in common variable names: ```pos```, ```spk```, ```maze_epoch```, etc. ```main.py``` can then easily import these specific variables and switch between which dataset is being processed with ease.

# TODO

- Linearise the position data (2D --> 1D), for L.F.'s data this is computing the shortest path distance between the reward site at the end of the centre arm of the maze. Who knows how you do this for the other datasets...
- During rest epochs, divide detected ripple intervals into 15ms blocks and use the Bayesian decorder to determine whether these blocks correspond to positions from the maze epochs.

# Questions

- Do we need to linearise the positions? The benefit is that the ripple blocks can be plotted against positions s.t. the plot is 2D, but by having 2D positions the only difference is that this plot is 3D? I suppose this makes things more complicated, but then again there doesn't seem to be a general solution to linearising positions (it is map specific).
