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
