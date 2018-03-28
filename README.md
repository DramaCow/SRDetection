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
