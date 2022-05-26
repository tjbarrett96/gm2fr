import numpy as np
import gm2fr.constants as const
import scipy.stats
import matplotlib.pyplot as plt
import gm2fr.style as style
style.set_style()

# ==================================================================================================

# Un-normalized sinc function, with optional scale in numerator: sin(ax)/x.
def sinc(x, scale = 1):
  return scale * np.sinc(scale * x / np.pi)

# s(omega) "wiggle" function from finite Fourier transform
def s(f, ts, tm, t0 = 0):
  return sinc(2 * np.pi * f, (tm - t0) * const.kHz_us) - sinc(2 * np.pi * f, (ts - t0) * const.kHz_us)

# c(omega) "wiggle" function from finite Fourier transform
def c(f, ts, tm, t0 = 0):
  # use trig identity for cos(a) - cos(b) to avoid indeterminate form
  return -np.sin(np.pi * f * (tm + ts - 2 * t0) * const.kHz_us) * sinc(np.pi * f, (tm - ts) * const.kHz_us)

# ==================================================================================================

# Calculate the two-sided p-value from the specified chi-squared distribution.
def p_value(chi2, ndf):

  # Difference between the observed chi-squared and expectation value.
  diff = abs(chi2 - ndf)

  # Probability of a more extreme value on the left side of the mean.
  left = scipy.stats.chi2.cdf(ndf - diff, ndf)

  # Probability of a more extreme value on the right side of the mean. (sf = 1 - cdf)
  right = scipy.stats.chi2.sf(ndf + diff, ndf)

  # Total probability of a more extreme value than what was observed.
  return left + right

# ==================================================================================================

# One-sided FFT and frequencies (kHz), assuming time in microseconds.
def fft(time, data):
  transform = np.fft.rfft(data)
  frequencies = np.fft.rfftfreq(len(time), d = 1E-9) * 1E-3
  return frequencies, transform

# Plot an FFT of the supplied signal.
def plot_fft(t, y, output = None):

  # Calculate the FFT magnitude.
  f, transform = fft(t, y)
  mag = np.abs(transform)

  # Plot the FFT magnitude.
  plt.plot(f, mag)

  # Axis labels.
  style.xlabel("Frequency (kHz)")
  style.ylabel("Arbitrary Units")

  # Use a logarithmic vertical scale, and set the limits appropriately.
  plt.yscale("log")
  plt.xlim(0, 8000)
  plt.ylim(np.min(mag[(f < 8000)]), None)

  # Save and clear.
  if output is not None:
    plt.savefig(output)
    plt.clf()

# ==================================================================================================

def A(t, t0, f = const.info["f"].magic):
  return np.cos(2 * np.pi * f * (t - t0) * const.kHz_us)

def B(t, t0, f = const.info["f"].magic):
  return np.sin(2 * np.pi * f * (t - t0) * const.kHz_us)

# ==================================================================================================

def np_transform(trig, f, S, t):
  return np.einsum("i, ki -> k", S, trig(2 * np.pi * np.outer(f, t) * const.kHz_us))
