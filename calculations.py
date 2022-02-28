import numpy as np
import gm2fr.constants as const
import scipy
import matplotlib.pyplot as plt
import gm2fr.style as style
style.setStyle()
from gm2fr.Histogram1D import Histogram1D

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
  return np.sin(np.pi * f * (tm + ts - 2 * t0) * const.kHz_us) * sinc(np.pi * f, (tm - ts) * const.kHz_us)
  # -1?

# ==================================================================================================

# Calculate the two-sided p-value from the specified chi-squared distribution.
def pval(chi2, ndf):

  # Difference between the observed chi-squared and expectation value.
  diff = abs(chi2 - ndf)

  # Probability of a more extreme value on the left side of the mean.
  left = scipy.stats.chi2.cdf(ndf - diff, ndf)

  # Probability of a more extreme value on the right side of the mean.
  # (sf = 1 - cdf)
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
def plotFFT(t, y, output = None):

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

def npTransform(trig, f, S, t):
  return np.einsum("i, ki -> k", S, trig(2 * np.pi * np.outer(f, t) * const.kHz_us))

# ==================================================================================================

# Return the sine/cosine transform of a Histogram1D object.
def transform(signal, frequencies, t0, type = "cosine", cross_cov = False):

  dt = signal.centers[1] - signal.centers[0]
  df = frequencies[1] - frequencies[0]
  result = Histogram1D(np.arange(frequencies[0] - df/2, frequencies[-1] + df, df))

  differences = np.arange(result.length) * result.width
  cov = None

  if type in ("sine", "cosine"):

    (trig, wiggle) = (np.sin, c) if type == "sine" else (np.cos, s)
    heights = npTransform(trig, result.centers, signal.heights, signal.centers - t0)
    cov = 0.5 * scipy.linalg.toeplitz(npTransform(np.cos, differences, signal.errors**2, signal.centers - t0))

    heights -= wiggle(result.centers, signal.centers[0], signal.centers[-1], t0) / (dt * const.kHz_us)

  elif type == "magnitude":

    cos = npTransform(np.cos, result.centers, signal.heights, signal.centers - t0)
    sin = npTransform(np.sin, result.centers, signal.heights, signal.centers - t0)

    cos -= s(result.centers, signal.centers[0], signal.centers[-1], t0) / (dt * const.kHz_us)
    sin -= c(result.centers, signal.centers[0], signal.centers[-1], t0) / (dt * const.kHz_us)

    tempCos = scipy.linalg.toeplitz(npTransform(np.cos, differences, signal.errors**2, signal.centers - t0))
    tempSin = scipy.linalg.toeplitz(npTransform(np.sin, differences, signal.errors**2, signal.centers - t0))

    heights = np.sqrt(cos**2 + sin**2)
    cov = 0.5 / np.outer(heights, heights) * (
      (np.outer(cos, cos) + np.outer(sin, sin)) * tempCos \
      + (np.outer(sin, cos) - np.outer(cos, sin)) * tempSin
    )

  else:
    raise ValueError(f"Frequency transform type '{type}' not recognized.")

  result.setHeights(heights)
  result.setCov(cov)

  if cross_cov:
    col = npTransform(np.sin, differences, signal.errors**2, signal.centers - t0)
    row = col.copy()
    row[1:] = -row[1:]
    cross_cov = -0.5 * scipy.linalg.toeplitz(col, row)
    return result, cross_cov
  else:
    return result

# ==================================================================================================

def combineAtT0(cos, sin, cross_cov, t0, err = 0):

  omega = 2 * np.pi * cos.centers
  cos_t0 = np.cos(omega * t0 * const.kHz_us)
  sin_t0 = np.sin(omega * t0 * const.kHz_us)

  wc = cos.copy().clear().setHeights(cos_t0)
  ws = sin.copy().clear().setHeights(sin_t0)
  if err > 0:
    wc.setCov(np.outer(omega * sin_t0, omega * sin_t0) * err**2 * const.kHz_us**2)
    ws.setCov(np.outer(omega * cos_t0, omega * cos_t0) * err**2 * const.kHz_us**2)

  weighted_cos = wc.multiply(cos)
  weighted_sin = ws.multiply(sin)

  cov = np.outer(wc.heights, ws.heights) * cross_cov
  if err > 0:
    cov_wc_ws = -np.outer(omega * sin_t0, omega * cos_t0) * err**2 * const.kHz_us**2
    cov += np.outer(cos.heights, sin.heights) * cov_wc_ws

  return weighted_cos.add(weighted_sin, cov = cov)
