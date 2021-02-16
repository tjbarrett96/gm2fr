import numpy as np
import scipy.signal as sgn

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import time

import gm2fr.analysis.FastRotation
from gm2fr.analysis.BackgroundFit import BackgroundFit
import gm2fr.utilities as util

import gm2fr.style as style

import numba as nb

# ==============================================================================

class Transform:

  # ============================================================================

  # Constructor for the Transform object.
  # TODO: list and initialize all instance variables here, for clarity and robustness.
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
    # Cutoff (in std. dev. of residuals) for inclusion of new background points.
    cutoff = 2,
    # Background fit model. Options: "parabola" / "sinc" / "error".
    model = "parabola",
    # Half-width of t0 window (in us) for initial coarse optimization scan.
    coarseRange = 0.015,
    # Step size of t0 window (in us) for initial coarse optimization scan.
    coarseStep = 0.0005,
    # Half-width of t0 window (in us) for subsequent fine optimization scans.
    fineRange = 0.0005,
    # Step size of t0 window (in us) for subsequent fine optimization scans.
    fineStep = 0.000025,
    # Output directory.
    output = None,
    # Boolean switch to enable or disable t0 optimization.
    optimize = True,
    # If not None, fix the background fit bounds to these provided.
    bounds = None,
    # Initial t0 guess (in us), optimized or fixed based on "optimize" option.
    t0 = 0.060,
    # Which plots to include. 0 = nothing, 1 = main results, 2 = t0 scans too
    plots = 2
  ):

    # The fast rotation signal.
    self.fr = fastRotation

    # The start and end times.
    self.start = max(start, self.fr.time[0])
    self.end = min(end, self.fr.time[-1])

    # Apply the start/end mask to the fast rotation data.
    self.tMask = (self.fr.time >= self.start) & (self.fr.time <= self.end)
    self.frTime = self.fr.time[self.tMask]
    self.frSignal = self.fr.signal[self.tMask]
    self.frError = self.fr.error[self.tMask]

    # Plotting option.
    self.plots = plots

    # Define the frequency values for evaluating the cosine transform.
    self.frequency = np.arange(6631, 6780, df)
    #self.frequency = np.arange(6601, 6810, df)

    # Mask selecting only physical frequencies determined by the collimators.
    self.fMask = (self.frequency >= util.minimum["frequency"]) & (self.frequency <= util.maximum["frequency"])

    # Initialize the transform bin heights.
    self.signal = np.zeros(len(self.frequency))

    # Keep track of the normalization factor used.
    self.normalization = 1

    # Initialize the covariance matrix among the transform bins.
    self.cov = np.zeros((len(self.frequency), len(self.frequency)))

    # The background fit model.
    self.bgModel = model

    # The cutoff (in sigma) for inclusion of background points in the fit.
    self.bgCutoff = cutoff

    # The BackgroundFit object (to be processed later).
    self.bgFit = None

    # t0 optimization parameters defining the scan windows.
    self.coarseRange = coarseRange
    self.coarseStep = coarseStep
    self.fineRange = fineRange
    self.fineStep = fineStep

    # Output directory.
    self.output = output

    # Whether or not to run t0 optimization.
    self.optimization = optimize

    # Fit bounds.
    self.bounds = bounds

    # Initial t0 guess (in us).
    self.t0 = t0

    # Convert frequency to other units.
    self.axes = {
      "frequency": self.frequency,
      "period": util.frequencyToPeriod(self.frequency),
      "radius": util.frequencyToRadialOffset(self.frequency),
      "momentum": util.frequencyToMomentum(self.frequency),
      "gamma": util.frequencyToGamma(self.frequency)
    }

    # Add a couple more, using what's been initialized above.
    self.axes["lifetime"] = self.axes["gamma"] * util.lifetime * 1E-3
    self.axes["offset"] = util.momentumToOffset(self.axes["momentum"]) * 100

    # TODO: these dictionaries perhaps could go in utilities.py?

    self.labels = {
      "frequency": {"math": "f", "simple": "f", "units": "kHz", "plot": "Revolution Frequency"},
      "period": {"math": "T", "simple": "T", "units": "ns", "plot": "Revolution Period"},
      "radius": {"math": "x_e", "simple": "x", "units": "mm", "plot": "Equilibrium Radius"},
      "momentum": {"math": "p", "simple": "p", "units": "GeV", "plot": "Momentum"},
      "gamma": {"math": r"\gamma", "simple": "gamma", "units": None, "plot": "Gamma Factor"},
      "lifetime": {"math": r"\tau", "simple": "tau", "units": r"$\mu$s", "plot": "Muon Lifetime"},
      "offset": {"math": r"\delta p/p_0", "simple": "dp/p0", "units": r"\%", "plot": "Fractional Momentum Offset"}
    }

    # Initialize the structured array of results, with column headers.
    self.results = np.zeros(
      1,
      dtype =   [(f"<{self.labels[axis]['simple']}>", np.float32) for axis in self.labels.keys()] \
              + [(f"sigma_{self.labels[axis]['simple']}", np.float32) for axis in self.labels.keys()]
    )

    # # Unit strings for each axis.
    # self.units = {
    #   "frequency": "kHz",
    #   "period": "ns",
    #   "radius": "mm",
    #   "momentum": "GeV",
    #   "gamma": None,
    #   "lifetime": r"$\mu$s",
    #   "offset": r"\%"
    # }
    #
    # # Symbols for each axis.
    # self.symbols = {
    #   "frequency": "f",
    #   "period": "T",
    #   "radius": "x_e",
    #   "momentum": "p",
    #   "gamma": r"\gamma",
    #   "lifetime": r"\tau",
    #   "offset": r"\delta p / p_0"
    # }
    #
    # # Labels for each axis.
    # self.labels = {
    #   "frequency": "Revolution Frequency",
    #   "period": "Revolution Period",
    #   "radius": "Equilibrium Radius",
    #   "momentum": "Momentum",
    #   "gamma": "Gamma Factor",
    #   "lifetime": "Muon Lifetime",
    #   "offset": "Momentum Offset"
    # }

  # ============================================================================

  # Calculate the cosine transform using Numba.
  # For speed, need to take everything as arguments; no "self" references.
  @staticmethod
  @nb.njit(fastmath = True, parallel = True)
  def fastTransform(t, S, S_err, f, t0, result):

    # Conversion from (kHz * us) to the standard (Hz * s).
    kHz_us = 1E-3

    # Calculate the transform, parallelizing the (smaller) frequency loop.
    for i in nb.prange(len(f)):
      result[i] = 0
      for j in range(len(t)):
        result[i] += S[j] * np.cos(2 * np.pi * f[i] * (t[j] - t0) * kHz_us)

  # ============================================================================

  # Calculate the covariance matrix of the cosine transform using Numba.
  # For speed, need to take everything as arguments; no "self" references.
  @staticmethod
  @nb.njit(fastmath = True, parallel = True)
  def fastCovariance(t, S_err, f, t0, cov, mask):

    # Conversion from (kHz * us) to the standard (Hz * s).
    kHz_us = 1E-3

    # Ensure everything is reset to zero first.
    cov.fill(0)

    # Calculate the covariance matrix, parallelizing the first frequency loop.
    for i in nb.prange(len(f)):
      if mask[i]:
        for j in range(i, len(f)):
          if mask[j]:
            for k in range(len(t)):
              cov[i, j] += np.cos(2*np.pi*f[i]*(t[k]-t0)*kHz_us) \
                           * np.cos(2*np.pi*f[j]*(t[k]-t0)*kHz_us) * S_err[k]**2
            cov[j, i] = cov[i, j]

  # ============================================================================

  # Update this object's covariance matrix using the current transform.
  # This is a wrapper for the Numba implementation, which can't use "self".
  def covariance(self, bounds = None):

    if bounds is None:
      mask = np.ones(len(self.frequency))
    else:
      mask = (self.frequency < bounds[0]) | (self.frequency > bounds[1])

    # Pass this object's instance variables to the Numba implementation.
    Transform.fastCovariance(
      self.frTime,
      self.frError,
      self.frequency,
      self.t0,
      self.cov,
      mask = mask
    )

    # self.cov /= self.normalization**2

  # ============================================================================

  # Update this object's cosine transform using the current self.t0.
  # This is a wrapper for the Numba implementation, which can't use "self".
  def transform(self, reset = True):

    # Pass this object's instance variables to the Numba implementation.
    Transform.fastTransform(
      self.frTime,
      self.frSignal,
      self.frError,
      self.frequency,
      self.t0,
      self.signal
    )

    # Normalize the maximum value to 1.
    # self.normalization = np.max(self.signal)
    # self.signal /= self.normalization

    # Ensure the covariance matrix is reset.
    if reset:
      self.cov.fill(0)

  # ============================================================================

  # Find the "best" background fit over range of candidate t0 times.
  # TODO: save only ~10 representative sample plots from the scan to speed things up
  def optimize(
    self,
    bounds,
    index = 0,  # iteration index
    subindex = 0,   # sub-iteration index, if minimum not found
    cov = False # use covariance matrix for chi-squared
  ):

    # Make the list of t0 for the scan.
    if index > 0:
      times = np.arange(self.t0 - self.fineRange, self.t0 + self.fineRange, self.fineStep)
    elif index == 0:
      times = np.arange(self.t0 - self.coarseRange, self.t0 + self.coarseRange, self.coarseStep)
    else:
      raise ValueError(f"Optimization index '{index}' not recognized.")

    # Initialize the list of BackgroundFit objects for each time in the scan.
    scanResults = [None] * len(times)

    # Over a small t0 window, the covariance doesn't change much.
    # Just calculate it once at the start.
    if index > 0 and cov:
      self.covariance(bounds)

    # For each symmetry time...
    for i in range(len(times)):

      # Set the current t0, and update the cosine transform.
      self.t0 = times[i]
      self.transform(reset = False if index > 0 and cov else True)

      # Create the BackgroundFit object and perform the fit.
      scanResults[i] = BackgroundFit(self, bounds = bounds)
      scanResults[i].fit()

    # Find the fit with the smallest (not-yet-normalized) chi-squared.
    minimum = min(scanResults, key = lambda fit: fit.chi2dof)

    # Update each fit object's uncertainties using the optimal residual spread.
    for i in range(len(times)):
      scanResults[i].update(minimum.spread)

    # Define how to determine if BackgroundFit "a" is better than "b".
    def better(a, b):
      # if mode == "coarse" and a.newBounds != b.newBounds:
      #   return a.betterBounds(b)
      return (a.chi2dof <= b.chi2dof)

    # Find the best fit within the scan.
    optIndex = 0
    for i in range(len(times)):
      if better(scanResults[i], scanResults[optIndex]):
        optIndex = i

    # Extract the lists of chi-squareds and bounds for plotting.
    scanChiSquared = np.array([fit.chi2dof for fit in scanResults])
    scanLeftBound = np.array([fit.newBounds[0] for fit in scanResults])
    scanRightBound = np.array([fit.newBounds[1] for fit in scanResults])

    # ==========================================================================

    # Plot the scan results.
    if self.output is not None:

      # Special label when repeating after a failed optimization.
      substring = f"-{subindex}" if subindex > 0 else ""

      # Only plot each background fit during the coarse scan.
      if index == 0 and self.plots >= 2:

        # Temporarily turn off LaTeX rendering for faster plots.
        latex = plt.rcParams["text.usetex"]
        plt.rcParams["text.usetex"] = False

        # Initialize the multi-page PDF file for scan plots.
        pdf = PdfPages(f"{self.output}/background/AllFits_Opt{index}{substring}.pdf")

        # Initialize a plot of the background fit, using a dummy default object.
        BackgroundFit(self, bounds).plot()

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

      # Plot the chi-squared on the primary y-axis.
      line1, = plt.plot(times * 1000, scanChiSquared, 'o-', label = r"$\chi^2$/dof")
      line2 = plt.axvline(times[optIndex]*1000, c = "k", ls = "--", label = "Optimum")
      style.xlabel("$t_0$ (ns)")
      style.ylabel(r"$\chi^2$/dof")

      # Plot the fit bounds on a secondary y-axis.
      plt.twinx()
      plt.subplots_adjust(right = 0.88)
      line3, = plt.plot(times * 1000, scanLeftBound, 'v-', c = "C1", label = "Lower Bound")
      line4, = plt.plot(times * 1000, scanRightBound, '^-', c = "C2", label = "Upper Bound")
      style.ylabel("Frequency (kHz)")

      # Draw the legend, forcing handles from both axes into a single box.
      lines = [line1, line2, line3, line4]
      plt.legend(lines, [line.get_label() for line in lines], loc = "best")

      # Save the result to disk.
      if self.output is not None:
        plt.savefig(f"{self.output}/background/ChiSquared_Opt{index}{substring}.pdf")

      # Fully close (instead of clear) to reset the adjusted padding.
      plt.close()

    # ==========================================================================

    # If there's no minimum sufficiently inside the scan window, try again.
    if optIndex < 2 or optIndex > len(times) - 3:

      # Fit a parabola to the whole distribution, and estimate the minimum.
      popt = np.polyfit(times, scanChiSquared, 2)
      self.t0 = -popt[1] / (2 * popt[0]) # -b/2a (quadratic formula)

      # Print an update.
      print("\nOptimal t0 not found within time window.")
      print(f"Trying again with re-estimated t0 seed: {self.t0*1000:.4f} ns.")

      # Make a recursive call to optimize again using the new estimate.
      self.optimize(bounds, index, subindex + 1)

    # Otherwise, if there is a minimum sufficiently inside the scan window...
    else:

      # Remember the optimal fit from the scan.
      self.bgFit = scanResults[optIndex]

      # Plot the optimal fit result.
      if self.output is not None:
        self.bgFit.plot(output = f"{self.output}/background/BestFit_Opt{index}.pdf")

      # Fit a parabola to the 2 neighbors on either side of the minimum.
      popt = np.polyfit(
        times[(optIndex - 2):(optIndex + 3)],
        scanChiSquared[(optIndex - 2):(optIndex + 3)],
        2
      )

      # Estimate t0 using the minimum of the parabolic fit.
      self.t0 = -popt[1] / (2 * popt[0]) # -b/2a (quadratic formula)

      # Print an update, completing this round of optimization.
      print(f"\nCompleted background optimization {index}.")
      print(f"{'chi2/dof':>16} = {self.bgFit.chi2dof:.4f}")
      print(f"{'bounds':>16} = {self.bgFit.newBounds} kHz")
      print(f"{'spread':>16} = {self.bgFit.spread:.4f}")
      print(f"{'new t0':>16} = {self.t0*1000:.4f} ns")

  # ============================================================================

  # Process the cosine transform by fitting and removing the background.
  def process(self):

    # Status update, and begin timing.
    print("\nProcessing frequency distribution...")
    begin = time.time()

    # Set the fit bounds, and determine if they're fixed or should float.
    if self.bounds is not None:
      fixed = True
      bounds = self.bounds
    else:
      fixed = False
      bounds = (util.minFrequency, util.maxFrequency)

    # Remember the previous iteration's fit bounds, to check if they've changed.
    oldBounds = (None, None)
    counter = 0

    # Scan over symmetry times to find the best background fit.
    if self.optimization:

      # Run the coarse optimization routine.
      self.optimize(bounds, index = 0)

      # Countinue iterating the bounds until they don't change anymore.
      while self.bgFit.newBounds != oldBounds:

        # Update the previous bounds used, and the number of iterations.
        oldBounds = self.bgFit.newBounds
        counter += 1

        # Run the fine optimization routine.
        self.optimize(
          self.bgFit.newBounds if not fixed else bounds,
          index = counter
        )

        # If the bounds are fixed, stop here; the optimized t0 won't change.
        if fixed:
          break

      self.optimize(bounds = self.bgFit.newBounds if not fixed else bounds, index = counter + 1, cov = True)

      # Do the final transform and fit, using the optimized t0.
      # self.transform()
      # self.bgFit = BackgroundFit(
      #   self,
      #   self.bgFit.newBounds if not fixed else bounds,
      #   self.bgFit.spread
      # )
      # self.bgFit.fit()

    # Fix the supplied t0, and simply find the optimal bounds.
    # TODO: if the fixed t0 is bad, bound optimization will fail; find a way to catch and handle this elegantly
    else:

      # Calculate the cosine transform at the fixed t0.
      self.transform()

      # Perform an initial background fit.
      self.bgFit = BackgroundFit(self, bounds = bounds, uncertainty = 1)
      self.bgFit.fit()
      self.bgFit.update(self.bgFit.spread)

      # Countinue iterating the bounds until they don't change anymore.
      while not fixed and self.bgFit.newBounds != oldBounds:

        # Update the previous bounds used.
        oldBounds = self.bgFit.newBounds

        # Perform the background fit.
        self.bgFit = BackgroundFit(
          self,
          bounds = self.bgFit.newBounds if not fixed else bounds,
          uncertainty = self.bgFit.spread
        )
        self.bgFit.fit()

    # Calculate a final transform and covariance.
    self.transform()
    self.covariance()

    # Perform the final background fit.
    self.bgFit = BackgroundFit(
      self,
      bounds = self.bgFit.newBounds if not fixed else bounds,
      uncertainty = self.bgFit.spread
    )
    self.bgFit.fit()

    print("\nCompleted final background fit.")
    print(f"{'chi2/dof':>16} = {self.bgFit.chi2:.4f} / {self.bgFit.dof} = {self.bgFit.chi2dof:.4f}")

    # Plot final background fit result.
    if self.output is not None:
      self.bgFit.plot(output = f"{self.output}/BackgroundFit.pdf")

    # Subtract the background and re-normalize.
    self.signal -= self.bgFit.fitResult
    self.signal /= np.max(np.abs(self.signal))

    # Plot the background-subtracted result.
    if self.plots >= 2:
      for axis in self.axes.keys():
        self.plot(axis)
    elif self.plots >= 1:
      self.plot("frequency")
      self.plot("radius")

    # Save the resulting distributions, and optimal background fit.
    if self.output is not None:
      self.save()
      self.bgFit.save(f"{self.output}/background.npz")

    # Save key results.
    for i in self.labels.keys():
      self.results[f"<{self.labels[i]['simple']}>"] = self.getMean(i)
      self.results[f"sigma_{self.labels[i]['simple']}"] = self.getWidth(i)

    print(f"\nFinished background removal, in {(time.time() - begin):.2f} seconds.")

  # ============================================================================

  # Plot the result.
  def plot(self, axis = "frequency"):

    # Don't bother if there's no output destination.
    if self.output is not None:

      # Plot the specified distribution.
      plt.plot(self.axes[axis], self.signal, 'o-')

      # Limit the viewing range to collimator constraints.
      plt.xlim(util.minimum[axis], util.maximum[axis])

      # Axis labels.
      style.ylabel("Arbitrary Units")
      style.xlabel(
        f"{self.labels[axis]['plot']}" \
        + (f" ({self.labels[axis]['units']})" if self.labels[axis]["units"] is not None else "")
      )

      # Infobox containing mean and standard deviation.
      style.databox(
        (fr"\langle {self.labels[axis]['math']} \rangle",
          self.getMean(axis),
          self.labels[axis]["units"]),
        (fr"\sigma_{{{self.labels[axis]['math']}}}",
          self.getWidth(axis),
          self.labels[axis]["units"])
      )

      # Save to disk.
      plt.savefig(f"{self.output}/{axis}.pdf")

      # Clear the figure.
      plt.clf()

  # ============================================================================

  # Save the transform and all axis units in NumPy format.
  def save(self):
    if self.output is not None:
      np.savez(
        f"{self.output}/transform.npz",
        transform = self.signal[self.fMask],
        **{axis: self.axes[axis][self.fMask] for axis in self.axes.keys()}
      )

  # ============================================================================

  # Calculate the mean of the specified axis within physical limits.
  def getMean(self, axis = "frequency"):
    return np.average(
      self.axes[axis][self.fMask],
      weights = self.signal[self.fMask]
    )

  # ============================================================================

  # Calculate the std. dev. of the specified axis within physical limits.
  def getWidth(self, axis = "frequency"):
    mean = self.getMean(axis)
    return np.sqrt(np.average(
      (self.axes[axis][self.fMask] - mean)**2,
      weights = self.signal[self.fMask]
    ))

  # ============================================================================

  # Calculate the electric field correction.
  def getCorrection(self, n = 0.108):
    return util.radialOffsetToCorrection(
      self.axes["radius"][self.fMask],
      self.signal[self.fMask],
      n
    )

  # ============================================================================
