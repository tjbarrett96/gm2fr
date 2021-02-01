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
    # Cutoff (in std. dev. of residuals) for inclusion of background points.
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
    fineStep = 0.000025
  ):

    # The start and end times.
    self.start = start
    self.end = end

    # The fast rotation signal.
    self.fr = fastRotation

    # Fast rotation time mask.
    self.tMask = (self.fr.time >= self.start) & (self.fr.time <= self.end)
    self.frTime = self.fr.time[self.tMask]
    self.frSignal = self.fr.signal[self.tMask]

    # Center the frequencies on magic, stepping by "df" in either direction.
#    self.frequency = np.concatenate(
#      (np.flip(np.arange(6705, 6630, -df)), np.arange(6705 + df, 6780, df))
#    )
    self.frequency = np.arange(6631, 6780, df)

    # Collimator mask.
    self.fMask = (self.frequency >= util.minimum["frequency"]) & (self.frequency <= util.maximum["frequency"])

    # The transform bin heights.
    self.signal = np.zeros(len(self.frequency))

    # The cutoff (in sigma) for inclusion of background points in the fit.
    self.bgCutoff = cutoff
    self.bgModel = model

    # BackgroundFit object.
    self.bgFit = None

    # Width for the coarse t0 scan.
    self.coarseRange = coarseRange
    self.coarseStep = coarseStep
    self.fineRange = fineRange
    self.fineStep = fineStep

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

    # Unit strings for each axis.
    self.units = {
      "frequency": "kHz",
      "period": "ns",
      "radius": "mm",
      "momentum": "GeV",
      "gamma": None,
      "lifetime": r"$\mu$s",
      "offset": r"\%"
    }

    # Symbols for each axis.
    self.symbols = {
      "frequency": "f",
      "period": "T",
      "radius": "x_e",
      "momentum": "p",
      "gamma": r"\gamma",
      "lifetime": r"\tau",
      "offset": r"\delta p / p_0"
    }

    # Labels for each axis.
    self.labels = {
      "frequency": "Revolution Frequency",
      "period": "Revolution Period",
      "radius": "Equilibrium Radius",
      "momentum": "Momentum",
      "gamma": "Gamma Factor",
      "lifetime": "Muon Lifetime",
      "offset": "Momentum Offset"
    }

  # ============================================================================

  # Calculate the cosine transform using Numba.
  # For speed, need to take everything as arguments; no "self" references.
  @staticmethod
  @nb.njit(fastmath = True, parallel = True)
  def fastTransform(t, S, f, t0):

    # Initialize the transform.
    result = np.zeros(len(f))

    # Conversion from (kHz * us) to the standard (Hz * s).
    kHz_us = 1E-3

    # Calculate the transform.
    # This uses Numba's parallelized nb.prange for the (smaller) frequency loop.
    for i in nb.prange(len(f)):
      for j in range(len(t)):
        result[i] += S[j] * np.cos(2 * np.pi * f[i] * (t[j] - t0) * kHz_us)

    # Return the result, normalizing its maximum value to 1.
    return result / np.max(result)

  # ============================================================================

  # Update this object's cosine transform using the given t0.
  def transform(self, t0):

    # Pass this object's instance variables to the Numba implementation.
    self.signal = Transform.fastTransform(
      self.frTime,
      self.frSignal,
      self.frequency,
      t0
    )

    # Calculate the frequency bin covariance matrix.
    # temp = np.cos(2 * np.pi * np.outer(k * f, times - t0) * kHz_ns)
    # cov = np.einsum("ik, jk, k -> ij", temp, temp, errors**2) / scale

  # ============================================================================

  # Find the "best" background fit over range of candidate t0 times.
  # TODO: for coarse optimization, perhaps we want the UPPER frequency bound to move inward the most, since we expect less distortion on this side?
  # TODO: or maybe not... but maybe some better, more sophisticated way to define when the bounds are "optimally inward"
  # TODO: save only ~10 representative sample plots from the scan to speed things up
  # TODO: for coarse optimization, minimize the difference between bounds; if it's the same as previous best, pick the lower chi-squared
  def optimize(
    self,
    seed,
    mode,
    bounds,
    output = None, # output directory for plots
    index = None,  # iteration index
    subindex = 0   # sub-iteration index, if minimum not found
  ):

    # Make the list of t0 for the scan.
    if mode == "fine":
      times = np.arange(seed - self.fineRange, seed + self.fineRange, self.fineStep)
    elif mode == "coarse":
      times = np.arange(seed - self.coarseRange, seed + self.coarseRange, self.coarseStep)
    else:
      raise ValueError(f"Optimization mode '{mode}' not recognized.")

    # List of BackgroundFit objects for each time in the scan.
    scanResults = [None] * len(times)

    # For each symmetry time...
    for i in range(len(times)):

      # Calculate the cosine transform.
      self.transform(times[i])

      # TODO: perhaps pass 'self' to the BGFit object? then it can use self.transform.frequency, etc.
      # Create the BackgroundFit object for this transform.
      scanResults[i] = BackgroundFit(
        self.frequency,
        self.signal,
        self.bgModel,
        self.start - times[i],
        bounds = bounds,
        uncertainty = 1,
        cutoff = self.bgCutoff
      )

      # Perform the fit.
      scanResults[i].fit()

    # Find the fit with the smallest (not-yet-normalized) chi-squared.
    minimum = min(scanResults, key = lambda fit: fit.chi2dof)

    # Interpret the minimum's spread in fit residuals as the fit uncertainty.
    uncertainty = minimum.spread

    # Re-scale each chi-squared using the optimal uncertainty, and re-estimate the new fit bounds.
    for i in range(len(times)):
      scanResults[i].fitUncertainty = uncertainty
      scanResults[i].chi2dof /= uncertainty**2
      scanResults[i].predictBounds()

    # Define how to determine if BackgroundFit "a" is better than "b".
    def better(a, b):
      if mode == "coarse":
        return a.betterBounds(b)
      else:
        return (a.chi2dof <= b.chi2dof)

    # Find the best fit within the scan.
    optIndex = 0
    for i in range(len(times)):
      if better(scanResults[i], scanResults[optIndex]):
        optIndex = i

    # ==========================================================================

    # Extract the lists of chi-squareds and bounds for plotting.
    scanChiSquared = np.array([fit.chi2dof for fit in scanResults])
    scanLeftBound = np.array([fit.newBounds[0] for fit in scanResults])
    scanRightBound = np.array([fit.newBounds[1] for fit in scanResults])

    # Plot the scan results.
    if output is not None:

      substring = f"-{subindex}" if subindex > 0 else ""

      # Only plot each background fit during the coarse scan.
      if mode == "coarse":

        # Temporarily turn off LaTeX rendering for faster plots.
        plt.rcParams["text.usetex"] = False

        # Initialize the multi-page PDF file for scan plots.
        pdf = PdfPages(f"{output}/background/AllFits_Opt{index}{substring}.pdf")

        # Initialize the plot objects, for speed.
        # TODO: just use the mask here, and make the BGFit object take the mask? Don't duplicate masking code.
        # TODO: also, figure out a way to initialize and get these without duplicating the BGFit.plot() code
        # TODO: perhaps initialize a dummy BGFit object, have the plot code make dummy plots if content is None, and optionally return the line handles, e.g. getHandles = True?
        mask = (self.frequency < bounds[0]) | (self.frequency > bounds[1])
        tfLine, = plt.plot(self.frequency, self.signal, 'o-', ms = 4, label = "Cosine Transform")
        bgLine, = plt.plot(self.frequency[mask], self.signal[mask], 'ko', ms = 4, label = "Background")
        fitLine, = plt.plot(self.frequency, self.signal, 'g', label = "Background Fit")
        textObj = plt.text(self.frequency[0], 1, "test", ha = "left", va = "top")
        plt.legend(loc = "upper right")
        style.xlabel("Frequency (kHz)")
        style.ylabel("Arbitrary Units")

        # Plot each background fit.
        for i in range(len(times)):
          scanResults[i].plot(
            standalone = False,
            tfLine = tfLine,
            bgLine = bgLine,
            fitLine = fitLine,
            labelObj = textObj,
            label = f"$t_0 = {times[i]*1000:.4f}$ ns"
          )
          pdf.savefig()

        # Close the multi-page PDF, and clear the current figure.
        pdf.close()
        plt.clf()

        # Resume LaTeX rendering.
        # TODO: only turn this back on if it was on in the first place!
        plt.rcParams["text.usetex"] = True

      # Plot the chi-squared on the primary y-axis.
      line1, = plt.plot(times * 1000, scanChiSquared, 'o-', color = "C0", ms = 4, label = r"$\chi^2$/dof")
      line2 = plt.axvline(times[optIndex]*1000, color = "k", linestyle = "--", label = "Optimum")
      style.xlabel("$t_0$ (ns)")
      style.ylabel(r"$\chi^2$/dof")

      # Plot the fit bounds on a secondary y-axis.
      plt.twinx()
      plt.subplots_adjust(right = 0.88)
      line3, = plt.plot(times * 1000, scanLeftBound, 'v-', color = "C1", ms = 4, label = "Lower Bound")
      line4, = plt.plot(times * 1000, scanRightBound, '^-', color = "C2", ms = 4, label = "Upper Bound")
      style.ylabel("Frequency (kHz)")

      # Draw the legend. With two y-axes, it wants to make two legend boxes.
      # Have to specify each handle to force them all in the same box.
      lines = [line1, line2, line3, line4]
      plt.legend(lines, [line.get_label() for line in lines], loc = "best")

      # Save the result to disk.
      plt.savefig(f"{output}/background/ChiSquared_Opt{index}{substring}.pdf")

      # Fully close (instead of clear) to reset the adjusted padding.
      plt.close()

    # ==========================================================================

    # If there's no minimum sufficiently inside the scan window, try again.
    if optIndex < 2 or optIndex > len(times) - 3:

      # Fit a parabola to the whole distribution, and estimate the minimum.
      popt = np.polyfit(times, scanChiSquared, 2)
      est = -popt[1] / (2 * popt[0]) # -b/2a (quadratic formula)

      # Print an update.
      print("\nOptimal t0 not found within time window.")
      print(f"Trying again with re-estimated t0 seed: {est*1000:.4f} ns.")

      # Make a recursive call to optimize again using the new estimate.
      self.optimize(est, mode, bounds, output, index + 1, subindex + 1)

    # Otherwise, if there is a minimum sufficiently inside the scan window...
    else:

      # Remember the optimal fit from the scan.
      self.bgFit = scanResults[optIndex]

      # Update the t0 estimate.
      if mode == "coarse":

        # In coarse mode, just take the time of the optimal scan index.
        self.t0 = times[optIndex]

      else:

        # Fit a parabola using 2 neighbors on either side of the optimal scan index.
        popt = np.polyfit(
          times[(optIndex - 2):(optIndex + 3)],
          scanChiSquared[(optIndex - 2):(optIndex + 3)],
          2
        )

        # Calculate the optimal symmetry time from the parabolic fit.
        self.t0 = -popt[1] / (2 * popt[0]) # -b/2a (quadratic formula)

      # Print an update, completing this round of optimization.
      print(f"\nCompleted {mode} background optimization.")
      print(f"{'chi2/dof':>16} = {self.bgFit.chi2dof:.4f}")
      print(f"{'bounds':>16} = {self.bgFit.newBounds} kHz")
      print(f"{'spread':>16} = {self.bgFit.spread:.4f}")
      print(f"{'new t0':>16} = {self.t0*1000:.4f} ns")

  # ============================================================================

  # Process the cosine transform by fitting and removing the background.
  # TODO: automate the approximation of 'spread', instead of assuming an initial value
  # TODO: print an update for moving the bounds inward
  # TODO: fixed bounds are pretty rudimentary; probably a better way to implement it
  # TODO: too much copying of BackgroundFit code; clean it up
  # TODO: consider putting these options in the constructor, including reference to Analyzer object (for output handling)
  def process(
    self,
    t0 = 0.100,      # t_0 seed (us)
    optimize = True, # whether to optimize t0 (True) or fix t0 (False)
    output = None,   # output directory for plots
    bounds = None    # if not None, fix the fit bounds to those provided
  ):

    print("\nProcessing frequency distribution...")
    begin = time.time()

    # Set the initial t0 seed.
    self.t0 = t0

    if bounds is not None:
      fixed = True
    else:
      fixed = False
      # Start with the unphysical regions beyond the collimator boundaries.
      bounds = (util.minFrequency, util.maxFrequency)

    oldBounds = (None, None)
    spread = 0.005

    # Scan over symmetry times to find the best background fit.
    if optimize:

      # Run the coarse optimization routine.
      self.optimize(
        self.t0,
        "coarse",
        bounds,
        output = output,
        index = 0
      )

      # Status print-out and plot.
      if output is not None:
        self.bgFit.plot(output = f"{output}/background/BestFit_Opt0.pdf")
        plt.clf()

      # Countinue iterating the bounds until they don't change anymore.
      counter = 0
      while not fixed and self.bgFit.newBounds != oldBounds:

        # Update the previous bounds used.
        oldBounds = self.bgFit.newBounds
        counter += 1

        # Run the fine optimization routine.
        self.optimize(
          self.t0,
          "fine",
          self.bgFit.newBounds if not fixed else bounds,
          output = output,
          index = counter
        )

        if output is not None:
          self.bgFit.plot(output = f"{output}/background/BestFit_Opt{counter}.pdf")
          plt.clf()

        # if fixed:
        #   break

      # Perform a final cosine transform after optimizing t0.
      self.transform(self.t0)

    # Fix the supplied t0, and simply find the optimal bounds.
    # TODO: if the fixed t0 is bad, bound optimization will fail; find a way to catch and handle this elegantly
    else:

      # Calculate the cosine transform at the fixed t0.
      self.transform(self.t0)

      # Perform an initial background fit.
      # TODO: this has not yet been made independent of the initial spread estimate
      self.bgFit = BackgroundFit(
        self.frequency,
        self.signal,
        self.bgModel,
        self.start - self.t0,
        bounds = bounds,
        uncertainty = 1,
        cutoff = self.bgCutoff
      )
      self.bgFit.fit()

      self.bgFit.fitUncertainty = self.bgFit.spread
      self.bgFit.chi2dof /= self.bgFit.fitUncertainty**2
      self.bgFit.predictBounds()

      # Countinue iterating the bounds until they don't change anymore.
      while not fixed and self.bgFit.newBounds != oldBounds:

        # Update the previous bounds used.
        oldBounds = self.bgFit.newBounds

        # Perform the background fit.
        self.bgFit = BackgroundFit(
          self.frequency,
          self.signal,
          self.bgModel,
          self.start - self.t0,
          bounds = self.bgFit.newBounds if not fixed else bounds,
          uncertainty = self.bgFit.spread,
          cutoff = self.bgCutoff
        )
        self.bgFit.fit()

        # if fixed:
        #   break

    # Do the final transform and fit, using the optimized t0.
    self.transform(self.t0)
    self.bgFit = BackgroundFit(
      self.frequency,
      self.signal,
      self.bgModel,
      self.start - self.t0,
      self.bgFit.newBounds if not fixed else bounds,
      self.bgFit.spread,
      self.bgCutoff
    )
    self.bgFit.fit()

    print("\nCompleted final background fit.")
    print(f"{'chi2/dof':>16} = {self.bgFit.chi2dof:.4f}")

    # Plot final background fit result.
    if output is not None:
      self.bgFit.plot(output = f"{output}/BackgroundFit.pdf")
      plt.clf()

    # Subtract the background and re-normalize.
    self.signal -= self.bgFit.fitResult
    self.signal /= np.max(np.abs(self.signal))

    # Plot the background-subtracted result.
    if output is not None:
      for axis in self.axes.keys():
        self.plot(axis, f"{output}/{axis}.pdf")

    print(f"\nFinished background removal, in {(time.time() - begin):.2f} seconds.")

  # ============================================================================

  # Plot the result.
  def plot(self, axis = "frequency", output = None):

    if output is not None:

      # Plot the specified distribution.
      plt.plot(self.axes[axis], self.signal, 'o-', ms = 4)

      # Limit the viewing range to collimator constraints.
      plt.xlim(util.minimum[axis], util.maximum[axis])

      # Axis labels.
      style.ylabel("Arbitrary Units")
      style.xlabel(
        f"{self.labels[axis]}" \
        + (f" ({self.units[axis]})" if self.units[axis] is not None else "")
      )

      # Infobox containing mean and standard deviation.
      style.databox(
        (fr"\langle {self.symbols[axis]} \rangle",
          self.getMean(axis),
          self.units[axis]),
        (fr"\sigma_{{{self.symbols[axis]}}}",
          self.getWidth(axis),
          self.units[axis])
      )

      # Save to disk.
      plt.savefig(output)

      # Clear the figure.
      plt.clf()

  # ============================================================================

  # Calculate the mean of the specified axis between the collimator limits.
  def getMean(self, axis = "frequency"):
    return np.average(
      self.axes[axis][self.fMask],
      weights = self.signal[self.fMask]
    )

  # ============================================================================

  # Calculate the standard deviation of the specified axis between the collimator limits.
  def getWidth(self, axis = "frequency"):
    mean = self.getMean(axis)
    return np.sqrt(
      np.average(
        (self.axes[axis][self.fMask] - mean)**2,
        weights = self.signal[self.fMask]
      )
    )

  # ============================================================================

  # Calculate the electric field correction.
  def getCorrection(self, n = 0.108):
    return util.radialOffsetToCorrection(self.radius[self.fMask], self.signal[self.fMask], n)

  # ============================================================================
