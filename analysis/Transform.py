import numpy as np
import scipy.signal as sgn
import scipy.linalg

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import time

import ROOT as root
import root_numpy as rnp

import gm2fr.analysis.FastRotation
from gm2fr.analysis.BackgroundFit import BackgroundFit
import gm2fr.utilities as util

import gm2fr.style as style

import numba as nb

# ==============================================================================

class Transform:

  # ============================================================================

  # Constructor for the Transform object.
  def __init__(
    self,
    # gm2fr.analysis.FastRotation object containing the signal.
    fastRotation,
    # Cosine transform start time (us).
    start = 4,
    # Cosine transform end time (us).
    end = 200,
    # Cosine transform frequency spacing (kHz).
    df = 2,
    # Background fit model. Options: "parabola" / "sinc" / "error".
    model = "sinc",
    # Half-width of t0 window (in us) for initial coarse optimization scan.
    coarseRange = 0.015,
    # Step size of t0 window (in us) for initial coarse optimization scan.
    coarseStep = 0.0005,
    # Half-width of t0 window (in us) for subsequent fine optimization scans.
    fineRange = 0.0005,
    # Step size of t0 window (in us) for subsequent fine optimization scans.
    fineStep = 0.000025,
    # Boolean switch to enable or disable t0 optimization.
    optimize = True,
    # Initial t0 guess (in us), optimized or fixed based on "optimize" option.
    t0 = 0.060,
    # Quad n-value.
    n = 0.108
  ):

    # The fast rotation signal.
    self.fr = fastRotation

    # The start and end times for the cosine transform.
    self.start = max(start, self.fr.time[0])
    self.end = min(end, self.fr.time[-1])

    # Apply the selected time mask to the fast rotation data.
    frMask = (self.fr.time >= self.start) & (self.fr.time <= self.end)
    self.frTime = self.fr.time[frMask]
    self.frSignal = self.fr.signal[frMask]
    self.frError = self.fr.error[frMask]

    # Define the frequency values for evaluating the cosine transform.
    self.df = df
    self.frequency = np.arange(6631, 6780, df)

    # Masks which select the physical and unphysical regions of frequency space.
    self.physical = (self.frequency >= util.min["f"]) & (self.frequency <= util.max["f"])
    self.unphysical = np.logical_not(self.physical)

    # Initialize the transform bin heights.
    self.signal = np.zeros(len(self.frequency))

    # Initialize the covariance matrix among the transform bins.
    self.cov = np.zeros((len(self.frequency), len(self.frequency)))

    # Whether or not to run t0 optimization.
    self.optimization = optimize

    # t0 value (or initial guess), and optimization uncertainty.
    self.t0 = t0
    self.err_t0 = np.nan

    # The background fit model.
    self.bgModel = model

    # The nominal BackgroundFit object, and one-sigma t0 variations.
    self.bgFit = None
    self.leftFit = None
    self.rightFit = None

    # t0 optimization parameters defining the scan windows.
    self.coarseRange = coarseRange
    self.coarseStep = coarseStep
    self.fineRange = fineRange
    self.fineStep = fineStep

    # Lists of the BackgroundFit objects over the coarse and fine t0 scans.
    self.coarseScan = []
    self.fineScan = []

    # Convert frequency to other units.
    self.axes = {
      "f": self.frequency,
      "T": util.frequencyToPeriod(self.frequency),
      "x": util.frequencyToRadialOffset(self.frequency),
      "p": util.frequencyToMomentum(self.frequency),
      "gamma": util.frequencyToGamma(self.frequency)
    }
    self.axes["tau"] = self.axes["gamma"] * util.lifetime * 1E-3
    self.axes["dp/p0"] = util.momentumToOffset(self.axes["p"]) * 100
    self.axes["c_e"] = 1E9 * 2 * n * (1 - n) * (util.magic["beta"] / util.magic["r"] * self.axes["x"])**2

    # Initialize a dictionary of means and statistical uncertainties.
    self.mean = {axis: None for axis in self.axes.keys()}
    self.meanError = {axis: None for axis in self.axes.keys()}

    # Initialize a dictionary of widths and statistical uncertainties.
    self.width = {axis: None for axis in self.axes.keys()}
    self.widthError = {axis: None for axis in self.axes.keys()}

    # Initialize a list of (name, value) pairs of key results.
    self.results = []

  # ============================================================================

  # Calculate the covariance matrix among frequency bins.
  # TODO: the neglected term is also Toeplitz, but sort of rotated/transposed. could include it efficiently to be sure approx. is okay
  def covariance(self, t0, mask = False, fix = True):

    # Conversion from (kHz * us) to the standard (Hz * s).
    kHz_us = 1E-3

    # Initialize the covariance for each frequency difference.
    column = np.zeros(len(self.frequency))

    # Calculate the covariance for each frequency difference.
    for i in range(len(self.frequency)):
      df = self.frequency[i] - self.frequency[0]
      column[i] = 0.5 * np.sum(
        np.cos(2 * np.pi * df * (self.frTime - t0) * kHz_us) * self.frError**2
      )

    # Extrapolate each difference's covariance to the full covariance matrix.
    result = scipy.linalg.toeplitz(column)

    # Return the full matrix, or only the unphysical frequency regions.
    if mask:
      return result[self.unphysical][:, self.unphysical]
    else:
      return result

  # ============================================================================

  # Calculate the cosine transform using Numba.
  # For speed, need to take everything as arguments; no "self" references.
  @staticmethod
  @nb.njit(fastmath = True, parallel = True)
  def __transform(t, S, f, t0):

    # Conversion from (kHz * us) to the standard (Hz * s).
    kHz_us = 1E-3

    # Initialize the cosine transform.
    result = np.zeros(len(f))

    # Calculate the transform, parallelizing the (smaller) frequency loop.
    for i in nb.prange(len(f)):
      result[i] = 0
      for j in range(len(t)):
        result[i] += S[j] * np.cos(2 * np.pi * f[i] * (t[j] - t0) * kHz_us)

    return result

  # Calculate the cosine transform using the supplied t0.
  # This is a wrapper for the Numba implementation, which can't use "self".
  def transform(self, t0):

    # Pass this object's instance variables to the Numba implementation.
    return Transform.__transform(
      self.frTime,
      self.frSignal,
      self.frequency,
      t0
    )

  # ============================================================================

  # Find the best background fit over a range of candidate t0 times.
  def optimize(self, t0, index = 0):

    # Make the list of t0 for the scan.
    if index > 0:
      times = np.arange(t0 - self.fineRange, t0 + self.fineRange, self.fineStep)
      scanResults = self.fineScan
    elif index == 0:
      times = np.arange(t0 - self.coarseRange, t0 + self.coarseRange, self.coarseStep)
      scanResults = self.coarseScan
    else:
      raise ValueError(f"Optimization index '{index}' not recognized.")

    # Ensure the results list is reset.
    scanResults.clear()

    # Estimate the covariance matrix at the central t0.
    cov = self.covariance(t0, mask = True)

    # For each symmetry time...
    for i in range(len(times)):

      # Calculate the cosine transform.
      signal = self.transform(times[i])

      # Create the BackgroundFit object and perform the fit.
      scanResults.append(BackgroundFit(self, signal, times[i], cov))
      scanResults[i].fit()

    # Find the fit with the smallest chi2/ndf.
    minimum = min(scanResults, key = lambda fit: fit.chi2ndf)
    scanMetric = np.array([fit.chi2ndf for fit in scanResults])

    # Get the scan index corresponding to the best fit.
    optIndex = scanResults.index(minimum)

    # If there's no minimum sufficiently inside the scan window, try again.
    if optIndex < 2 or optIndex > len(times) - 3:

      # Fit a parabola to the whole distribution, and estimate the minimum.
      popt = np.polyfit(times, scanMetric, 2)
      est_t0 = -popt[1] / (2 * popt[0]) # -b/2a (quadratic formula)

      # Print an update.
      print("\nOptimal t0 not found within time window.")
      print(f"Trying again with re-estimated t0 seed: {est_t0 * 1000:.4f} ns.")

      # Make a recursive call to optimize again using the new estimate.
      if index < 5:
        self.optimize(est_t0, index + 1)
      else:
        self.t0 = est_t0

    # Otherwise, if there is a minimum sufficiently inside the scan window...
    else:

      # Remember the optimal fit from the scan.
      self.bgFit = scanResults[optIndex]

      # Fit a parabola to the 2 neighbors on either side of the minimum.
      popt = np.polyfit(
        times[(optIndex - 2):(optIndex + 3)],
        scanMetric[(optIndex - 2):(optIndex + 3)],
        2
      )

      # Estimate t0 using the minimum of the parabolic fit.
      self.t0 = -popt[1] / (2 * popt[0]) # -b/2a (quadratic formula)


      # Get the value of the minimum chi2/ndf from the fit.
      min_chi2 = np.polyval(popt, self.t0) * self.bgFit.ndf

      # Fit a parabola to the entire window, and extrapolate chi2/ndf_min + 1.
      a, b, c = popt * self.bgFit.ndf
      c -= min_chi2 + 1
      leftTime = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a) # quadratic formula
      rightTime = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a) # quadratic formula
      leftMargin = self.t0 - leftTime
      rightMargin = rightTime - self.t0
      self.err_t0 = ((leftMargin + rightMargin) / 2)

      # Print an update, completing this round of optimization.
      print(f"\nCompleted background optimization.")
      print(f"{'chi2/ndf':>16} = {self.bgFit.chi2ndf:.4f}")
      print(f"{'spread':>16} = {self.bgFit.spread:.4f}")
      print(f"{'new t0':>16} = {self.t0*1000:.4f} ns")

  # ============================================================================

  # Process the cosine transform by fitting and removing the background.
  def process(self):

    # Status update, and begin timing.
    print("\nProcessing frequency distribution...")
    begin = time.time()

    # Scan over symmetry times to find the best background fit.
    if self.optimization:

      # TODO: just have self.optimize return the newly-constructed list of BGFits
      # then can do self.coarseScan = self.optimize(...)

      # TODO: handle the repetitions after failure here.
      # have self.optimize return [scanResults], opt_t0, and if opt_t0 is None try again
      # up to some maximum number of attempts

      # Run the coarse optimization routine.
      self.optimize(self.t0, index = 0)

      # Run the fine optimization routine.
      self.optimize(self.t0, index = 1)

    # Calculate the final transform and full covariance matrix.
    self.signal = self.transform(self.t0)
    self.cov = self.covariance(self.t0)

    # Perform the final background fit using the covariance information.
    self.bgFit = BackgroundFit(
      transform = self,
      signal = self.signal,
      t0 = self.t0,
      cov = self.cov[self.unphysical][:, self.unphysical]
    )
    self.bgFit.fit()

    # Subtract the background.
    self.signal = self.bgFit.subtract()

    # Calculate means, widths, and statistical uncertainties.
    for axis in self.axes.keys():
      self.mean[axis] = self.getMean(axis)
      self.width[axis] = self.getWidth(axis)
      self.meanError[axis] = self.getMeanError(axis)
      self.widthError[axis] = self.getWidthError(axis)

    # Propagate the uncertainty from t0 optimization, if performed.
    if self.optimization:

      # Perform a background fit with a one-sigma shift in t_0 to the left.
      left_t0 = self.t0 - self.err_t0
      self.leftFit = BackgroundFit(
        transform = self,
        signal = self.transform(left_t0),
        t0 = left_t0,
        cov = self.covariance(left_t0, mask = True)
      )
      self.leftFit.fit()

      # Perform a background fit with a one-sigma shift in t_0 to the right.
      right_t0 = self.t0 + self.err_t0
      self.rightFit = BackgroundFit(
        transform = self,
        signal = self.transform(right_t0),
        t0 = right_t0,
        cov = self.covariance(right_t0, mask = True)
      )
      self.rightFit.fit()

      # Propagate the uncertainty in t0 to the distribution mean and width.
      for axis in self.axes.keys():

        # Find the change in the mean after +/- one-sigma shifts in t0.
        left_mean_err = abs(self.mean[axis] - self.getMean(axis, self.leftFit.subtract()))
        right_mean_err = abs(self.mean[axis] - self.getMean(axis, self.rightFit.subtract()))

        # Find the change in the width after +/- one-sigma shifts in t0.
        left_width_err = abs(self.width[axis] - self.getWidth(axis, self.leftFit.subtract()))
        right_width_err = abs(self.width[axis] - self.getWidth(axis, self.rightFit.subtract()))

        # Average the changes on either side of the optimal t0.
        mean_err_t0 = (left_mean_err + right_mean_err) / 2
        width_err_t0 = (left_width_err + right_width_err) / 2

        # Add these in quadrature with the basic statistical uncertainties.
        self.meanError[axis] = np.sqrt(self.meanError[axis]**2 + mean_err_t0**2)
        self.widthError[axis] = np.sqrt(self.widthError[axis]**2 + width_err_t0**2)

    # ==========================================================================

    # Append (name, value) pairs of key results to the results list.
    self.results.append(("start", self.start))
    self.results.append(("end", self.end))
    self.results.append(("df", self.df))
    self.results.append(("t0", self.t0 * 1E3))
    self.results.append(("err_t0", self.err_t0 * 1E3))

    for axis in self.axes.keys():
      self.results.append((axis, self.mean[axis]))
      self.results.append((f"err_{axis}", self.meanError[axis]))
      self.results.append((f"sig_{axis}", self.width[axis]))
      self.results.append((f"err_sig_{axis}", self.widthError[axis]))

    # ==========================================================================

    print("\nCompleted final background fit.")
    print(f"{'chi2/ndf':>16} = {self.bgFit.chi2:.4f} / {self.bgFit.ndf} = {self.bgFit.chi2ndf:.4f}")
    print(f"{'p-value':>16} = {self.bgFit.pval:.4f}")

    # Normalize the maximum of the distribution to 1.
    # norm = np.max(np.abs(self.signal))
    # self.signal /= norm
    # self.cov /= norm**2

    print(f"\nFinished background removal, in {(time.time() - begin):.2f} seconds.")

  # ============================================================================

  # Plot the result.
  def plot(self, output, axis = "f"):

    # Plot the specified distribution.
    plt.plot(self.axes[axis][self.physical], self.signal[self.physical], 'o-')

    # Plot the magic quantity as a vertical line.
    plt.axvline(util.magic[axis], ls = ":", c = "k", label = "Magic")

    # Axis labels.
    label, units = util.labels[axis]['plot'], util.labels[axis]['units']
    style.xlabel(f"{label}" + (f" ({units})" if units != "" else ""))
    style.ylabel("Arbitrary Units")

    # Infobox containing mean and standard deviation.
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

    # Add the legend.
    plt.legend(loc = "upper right" if axis != "c_e" else "center right")

    # Save to disk.
    plt.savefig(output)

    # Clear the figure.
    plt.clf()

  # ============================================================================

  def plotOptimization(self, outDir, mode = "coarse", all = False):

    if mode == "coarse":
      scanResults = self.coarseScan
    elif mode == "fine":
      scanResults = self.fineScan
    else:
      raise ValueError(f"Optimization mode '{mode}' not recognized.")

    scanMetric = np.array([result.chi2ndf for result in scanResults])
    label = r"$\chi^2$/ndf"

    # Extract the t0 times.
    times = np.array([result.t0 for result in scanResults])

    # Plot the SSE or chi2/ndf.
    plt.plot(times * 1000, scanMetric, 'o-')

    # For a chi2/ndf plot...
    if mode == "fine":
      # Show the one-sigma t0 bounds as a shaded rectangle.
      plt.axvspan(
        (self.t0 - self.err_t0) * 1000,
        (self.t0 + self.err_t0) * 1000,
        alpha = 0.2,
        fc = "k",
        ec = None
      )
      # Plot the optimized t0 as a vertical line.
      plt.axvline(self.t0 * 1000, c = "k", ls = "--")
      # Show the horizontal reference line where chi2/ndf = 1.
      plt.axhline(1, c = "k", ls = ":")

    # Axis labels.
    style.xlabel("$t_0$ (ns)")
    style.ylabel(label)
    plt.ylim(0, None)

    # Save the result to disk, and clear the figure.
    plt.savefig(f"{outDir}/{mode}_scan.pdf")
    plt.clf()

    # Plot every background fit from the scan, in a multi-page PDF.
    if all:

      # Temporarily turn off LaTeX rendering for faster plots.
      latex = plt.rcParams["text.usetex"]
      plt.rcParams["text.usetex"] = False

      # Initialize the multi-page PDF file for scan plots.
      pdf = PdfPages(f"{outDir}/AllFits_{mode}.pdf")

      # Initialize a plot of the background fit.
      scanResults[0].plot()

      # Plot each background fit, updating the initialized plot each time.
      for i in range(len(times)):
        scanResults[i].plot(update = True)
        pdf.savefig()

      # Close the multi-page PDF, and clear the current figure.
      pdf.close()
      plt.clf()

      # Resume LaTeX rendering, if it was enabled before.
      if latex:
        plt.rcParams["text.usetex"] = True

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
  def getMean(self, axis = "f", signal = None):

    if signal is None:
      signal = self.signal

    return np.average(
      self.axes[axis][self.physical],
      weights = signal[self.physical]
    )

  # ============================================================================

  # Calculate the statistical uncertainty in the mean.
  def getMeanError(self, axis = "f"):
    x = self.axes[axis][self.physical]
    mean = self.getMean(axis)
    total = np.sum(self.signal[self.physical])
    cov = self.cov[self.physical][:, self.physical]
    return 1 / total * np.sqrt((x - mean).T @ cov @ (x - mean))

  # ============================================================================

  # Calculate the std. dev. of the specified axis within physical limits.
  def getWidth(self, axis = "f", signal = None):

    if signal is None:
      signal = self.signal

    mean = self.getMean(axis, signal)
    return np.sqrt(np.average(
      (self.axes[axis][self.physical] - mean)**2,
      weights = signal[self.physical]
    ))

  # ============================================================================

  # Calculate the statistical uncertainty in the std. dev.
  def getWidthError(self, axis = "f"):
    x = self.axes[axis][self.physical]
    mean = self.getMean(axis)
    width = self.getWidth(axis)
    total = np.sum(self.signal[self.physical])
    cov = self.cov[self.physical][:, self.physical]
    return 1 / (2 * width * total) * np.sqrt(
      ((x - mean)**2 - width**2).T @ cov @ ((x - mean)**2 - width**2)
    )
