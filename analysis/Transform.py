import numpy as np
import scipy.signal as sgn
import scipy.linalg

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import time

import ROOT as root
import root_numpy as rnp

# import gm2fr.analysis.FastRotation
import gm2fr.analysis.Optimizer as opt
from gm2fr.analysis.BackgroundFit import BackgroundFit
import gm2fr.utilities as util
from gm2fr.analysis.Results import Results
from gm2fr.Histogram1D import Histogram1D

import gm2fr.style as style

import numba as nb

# ==============================================================================

class Transform:

  # Constructor for the Transform object.
  def __init__(
    self,
    # Histogram object containing the signal.
    fastRotation,
    # Cosine transform start time (us).
    start = 4,
    # Cosine transform end time (us).
    end = 200,
    # Cosine transform frequency spacing (kHz).
    df = 2,
    # Background fit model. Options: "parabola" / "sinc" / "error".
    model = "sinc",
    # Initial t0 guess (in us), optimized or fixed based on "optimize" option.
    t0 = 0.060,
    # Quad n-value.
    n = 0.108
  ):

    # The fast rotation signal.
    self.fr = fastRotation
    self.n = n

    self.start, self.end = None, None
    if self.fr is not None:
      # The start and end times for the cosine transform.
      self.start = max(start, self.fr.centers[0])
      self.end = min(end, self.fr.centers[-1])

      # Apply the selected time mask to the fast rotation data.
      frMask = (self.fr.centers >= self.start) & (self.fr.centers <= self.end)
      self.frTime = self.fr.centers[frMask]
      self.frSignal = self.fr.heights[frMask]
      self.frError = self.fr.errors[frMask]

    # Define the frequency values for evaluating the cosine transform.
    self.df = df
    self.frequency = np.arange(6631, 6780, df)
    # self.frequency = np.arange(6611, 6800, df)

    # Initialize the transform bin heights.
    self.signal = np.zeros(len(self.frequency))

    # Initialize the covariance matrix among the transform bins.
    self.cov = np.zeros((len(self.frequency), len(self.frequency)))

    # t0 value (or initial guess) and uncertainty.
    self.t0 = t0
    self.err_t0 = 0

    # The background fit model and fit object.
    self.bgModel = model
    self.bgFit = None

    self.setup()

    # FFT results.
    self.fft = None

  def setup(self):

    # Masks which select the physical and unphysical regions of frequency space.
    self.left = (self.frequency < util.min["f"])
    self.right = (self.frequency > util.max["f"])
    self.unphysical = (self.left | self.right)
    self.physical = np.logical_not(self.unphysical)

    # Convert frequency to other units.
    self.axes = {
      "f": self.frequency,
      "T": util.frequencyToPeriod(self.frequency),
      "x": util.frequencyToRadialOffset(self.frequency),
      "p": util.frequencyToMomentum(self.frequency),
      "gamma": util.frequencyToGamma(self.frequency)
    }
    self.axes["tau"] = self.axes["gamma"] * util.lifetime * 1E-3
    self.axes["dp_p0"] = util.momentumToOffset(self.axes["p"]) * 100
    self.axes["c_e"] = 1E9 * 2 * self.n * (1 - self.n) * (util.magic["beta"] / util.magic["r"] * self.axes["x"])**2

  # ============================================================================

  # Calculate the covariance matrix among frequency bins.
  def covariance(self, t0, mask = None):

    # Initialize the covariance for each frequency difference.
    differences = np.zeros(len(self.frequency))
    sums_column = np.zeros(len(self.frequency))
    sums_row = np.zeros(len(self.frequency))

    kHz_us = 1E-3

    # Calculate the covariance for each frequency difference.
    for i in range(len(self.frequency)):

      df = self.frequency[i] - self.frequency[0]
      differences[i] = 0.5 * np.sum(
        np.cos(2 * np.pi * df * (self.frTime - t0) * kHz_us) * self.frError**2
      )

      sum_smallest = self.frequency[i] + self.frequency[0]
      sum_largest = self.frequency[i] + self.frequency[-1]
      sums_column[i] = 0.5 * np.sum(
        np.cos(2 * np.pi * sum_largest * (self.frTime - t0) * kHz_us) * self.frError**2
      )
      sums_row[i] = 0.5 * np.sum(
        np.cos(2 * np.pi * sum_smallest * (self.frTime - t0) * kHz_us) * self.frError**2
      )

    # Extrapolate each difference's covariance to the full covariance matrix.
    result = scipy.linalg.toeplitz(differences) + np.fliplr(scipy.linalg.toeplitz(sums_column, np.flip(sums_row)))

    # Return the full matrix, or only the unphysical frequency regions.
    if mask is not None:
      return result[mask][:, mask]
    else:
      return result

  # ============================================================================

  # Calculate the cosine transform using Numba.
  # For speed, need to take everything as arguments; no "self" references.
  @staticmethod
  @nb.njit(fastmath = True, parallel = True)
  def __transform(t, S, f, t0, sine = False):

    # Conversion from (kHz * us) to the standard (Hz * s).
    kHz_us = 1E-3

    # Initialize the cosine transform.
    result = np.zeros(len(f))

    # Calculate the transform, parallelizing the (smaller) frequency loop.
    if not sine:
      for i in nb.prange(len(f)):
        result[i] = 0
        for j in range(len(t)):
          result[i] += S[j] * np.cos(2 * np.pi * f[i] * (t[j] - t0) * kHz_us)
    else:
      for i in nb.prange(len(f)):
        result[i] = 0
        for j in range(len(t)):
          result[i] -= S[j] * np.sin(2 * np.pi * f[i] * (t[j] - t0) * kHz_us)

    return result

  # Calculate the cosine transform using the supplied t0.
  # This is a wrapper for the Numba implementation, which can't use "self".
  def transform(self, t0, sine = False):

    # Pass this object's instance variables to the Numba implementation.
    # return Transform.__transform(
    #   self.frTime,
    #   self.frSignal,
    #   self.frequency,
    #   t0,
    #   sine
    # )

    function = np.sin if sine else np.cos
    return np.einsum(
      "i, ki -> k",
      self.frSignal,
      function(2 * np.pi * np.outer(self.frequency, (self.frTime - t0)) * util.kHz_us)
    )

  # ============================================================================

  # Process the cosine transform by fitting and removing the background.
  def process(self, update = True):

    # Calculate the final transform and full covariance matrix.
    self.signal = self.transform(self.t0)
    self.cov = self.covariance(self.t0)

    # Perform the final background fit using the covariance information.
    # TODO: move to Analyzer?
    if self.bgModel is not None:

      self.bgFit = BackgroundFit(
        transform = self,
        signal = self.signal,
        t0 = self.t0,
        cov = self.cov[self.unphysical][:, self.unphysical]
      )
      self.bgFit.fit()

      self.signal = self.bgFit.subtract()
      self.cov += self.bgFit.model.covariance(self.frequency)

      if update:
        self.bgFit.model.print()

  # ============================================================================

  def copy(self):
    result = Transform(self.fr, self.start, self.end, self.df, self.bgModel, self.t0, self.n)
    result.frequency = self.frequency.copy()
    result.signal = self.signal.copy()
    result.cov = self.cov.copy()
    result.err_t0 = self.err_t0
    result.fft = self.fft.copy() if self.fft is not None else None
    result.bgFit = self.bgFit
    result.setup()
    return result

  # ============================================================================

  def results(self, parameters = True, errors = True):

    # TODO: move to Analyzer
    if parameters:
      results = {
        "start": self.start,
        "end": self.end,
        "df": self.df,
        "t0": self.t0 * 1E3,
        "err_t0": self.err_t0 * 1E3
      }
    else:
      results = {}

    # TODO: move to Histogram
    for axis in self.axes.keys():
      results[axis] = self.getMean(axis)
      results[f"sig_{axis}"] = self.getWidth(axis)
      if errors:
        results[f"err_{axis}"] = self.getMeanError(axis)
        results[f"err_sig_{axis}"] = self.getWidthError(axis)

    return Results(results)

  # ============================================================================

  # Plot the result.
  # TODO: doesn't account for t_0 error added in Analyzer
  def plot(self, output, axis = "f", databox = True, magic = True, zero = True, label = None, ls = "o-", scale = 1):

    # Plot the specified distribution.
    plt.plot(self.axes[axis][self.physical], self.signal[self.physical] * scale, ls, label = label)
    plt.xlim(util.min[axis], util.max[axis])

    # Plot the magic quantity as a vertical line.
    if magic:
      plt.axvline(util.magic[axis], ls = ":", c = "k", label = "Magic")

    if zero:
      style.yZero()

    # Axis labels.
    label, units = util.labels[axis]['plot'], util.labels[axis]['units']
    style.xlabel(f"{label}" + (f" ({units})" if units != "" else ""))
    style.ylabel("Arbitrary Units")

    # Infobox containing mean and standard deviation.
    if databox:
      try:
        style.databox(
          (fr"\langle {util.labels[axis]['math']} \rangle",
            self.getMean(axis),
            self.getMeanError(axis),
            util.labels[axis]["units"]),
          (fr"\sigma_{{{util.labels[axis]['math']}}}",
            self.getWidth(axis),
            self.getWidthError(axis),
            util.labels[axis]["units"]),
          left = False if axis == "c_e" else True
        )
      except:
        pass

    # Add the legend.
    plt.legend(loc = "upper right" if axis != "c_e" else "center right")

    # Save to disk.
    if output is not None:
      plt.savefig(output)
      # plt.clf()

  # ============================================================================

  def plotMagnitude(self, output, scale = None):

    # Calculate the cosine, sine, and Fourier (magnitude) transforms.
    real = self.transform(self.t0)
    imag = self.transform(self.t0, sine = True)
    mag = np.sqrt(real**2 + imag**2)

    if scale is None:
      scale = 1
    else:
      scale = scale / np.max(mag)

    real *= scale
    imag *= scale
    mag *= scale

    # Plot the magnitude, real, and imaginary parts.
    plt.plot(self.frequency, mag, 'o-', label = "Fourier Magnitude")
    plt.plot(self.frequency, real, 'o-', label = "Real Part (Cosine)")
    plt.plot(self.frequency, imag, 'o-', label = "Imag. Part (Sine)")
    plt.axhline(0, ls = ':', c = "k")
    plt.legend()

    # Axis labels.
    style.xlabel("Frequency (kHz)")
    style.ylabel("Arbitrary Units")

    plt.savefig(f"{output}/magnitude.pdf")
    plt.clf()

    # Plot the phase.
    plt.plot(self.frequency, np.arctan2(imag, real), 'o-')
    style.xlabel("Frequency (kHz)")
    style.ylabel("Phase (rad)")
    plt.savefig(f"{output}/phase.pdf")
    plt.clf()

  # ============================================================================

  # Save the transform results.
  def save(self, output):

      # Save the transform and all axis units in NumPy format.
      np.savez(
        output,
        transform = self.signal[self.physical],
        **{axis: self.axes[axis][self.physical] for axis in self.axes.keys()}
      )

      # Create a ROOT histogram for the frequency distribution.
      histogram = root.TH1F(
        "transform",
        ";Frequency (kHz);Arbitrary Units",
        len(self.frequency),
        self.frequency[0] - self.df/2,
        self.frequency[-1] + self.df/2
      )

      # Copy the signal into the histogram.
      rnp.array2hist(self.signal, histogram)

      # Save the histogram.
      name = output.split(".")[0]
      outFile = root.TFile(f"{name}.root", "RECREATE")
      histogram.Write()
      outFile.Close()

  # ============================================================================

  # Calculate the mean of the specified axis within physical limits.
  def getMean(self, axis = "f"):
    mask = self.physical & (self.signal > 0)
    return np.average(
      self.axes[axis][mask],
      weights = self.signal[mask]
    )

  # ============================================================================

  # Calculate the statistical uncertainty in the mean.
  def getMeanError(self, axis = "f"):
    mask = self.physical & (self.signal > 0)
    x = self.axes[axis][mask]
    mean = self.getMean(axis)
    total = np.sum(self.signal[mask])
    cov = self.cov[mask][:, mask]
    return 1 / total * np.sqrt((x - mean).T @ cov @ (x - mean))

  # ============================================================================

  # Calculate the std. dev. of the specified axis within physical limits.
  def getWidth(self, axis = "f"):
    mask = self.physical & (self.signal > 0)
    mean = self.getMean(axis)
    return np.sqrt(np.average(
      (self.axes[axis][mask] - mean)**2,
      weights = self.signal[mask]
    ))

  # ============================================================================

  # Calculate the statistical uncertainty in the std. dev.
  def getWidthError(self, axis = "f"):
    mask = self.physical & (self.signal > 0)
    x = self.axes[axis][mask]
    mean = self.getMean(axis)
    width = self.getWidth(axis)
    total = np.sum(self.signal[mask])
    cov = self.cov[mask][:, mask]
    return 1 / (2 * width * total) * np.sqrt(
      ((x - mean)**2 - width**2).T @ cov @ ((x - mean)**2 - width**2)
    )
