from gm2fr.analysis.BackgroundFit import BackgroundFit
from gm2fr.analysis.Results import Results
import gm2fr.analysis.Model as model
import gm2fr.utilities as util
from gm2fr.Histogram1D import Histogram1D

import numpy as np
import matplotlib.pyplot as plt
import gm2fr.style as style
style.setStyle()
from matplotlib.backends.backend_pdf import PdfPages

# ==============================================================================

class Optimizer:

  def __init__(
    self,
    fr,
    frequencies,
    model,
    step = 0.000025,
    width = 0.001,
    seed = None
  ):

    self.fr = fr
    self.start = self.fr.centers[0]
    self.frequencies = frequencies
    self.model = model

    self.seed = seed if seed is not None else self.getSeed()
    self.step = step
    self.width = width

    self.times = None
    self.fits = None
    self.chi2ndf = None

    self.t0 = None
    self.err_t0 = None

    self.bgFit = None
    self.leftFit = None
    self.rightFit = None

  # ============================================================================

  # TODO: method "fit" is buggy, claims it has not found good minimum when it has
  def getSeed(self, method = "average"):

    # Select one cyclotron period after the transform start time.
    mask = (self.fr.centers >= self.start) & (self.fr.centers <= self.start + util.magic["T"] * 1E-3)

    # Refine the selection by taking the nearest pair of minima.
    start = self.fr.centers[mask][np.argmin(self.fr.heights[mask])]
    end = start + util.magic["T"] * 1E-3
    mask = (self.fr.centers >= start) & (self.fr.centers <= end)

    # Compute a naive average of the times within this turn.
    if method == "average":

      mid = np.average(self.fr.centers[mask], weights = self.fr.heights[mask])

    # Fit a simple Gaussian to this turn.
    elif method == "fit":

      fit = model.Gaussian(2, (end + start) / 2, 0.025)
      fit.fit(self.fr.centers[mask], self.fr.heights[mask], self.fr.errors[mask])
      fit.print()
      mid = fit.pOpt[1]

    else:
      raise ValueError(f"t0 seed estimation mode '{method}' not recognized.")

    # Subtract multiples of the cyclotron period to get near t0 ~ 0.
    periods = mid // (util.magic["T"] * 1E-3)
    seed = mid - periods * util.magic["T"] * 1E-3

    print(f"\nEstimated t0 seed using method '{method}': {seed*1E3:.2f} ns.")
    return seed

  # ============================================================================

  def optimize(self, index = 0):

    self.times = np.arange(self.seed - self.width/2, self.seed + self.width/2, self.step)
    self.fits = [None] * len(self.times)

    # Initialize the cosine transform histogram at the t0 seed.
    transform = Histogram1D.transform(self.fr, self.frequencies, t0 = self.seed)
    # print(transform.cov)

    for i in range(len(self.times)):

      # Update the transform in-place at the current t0, without re-estimating the covariance.
      Histogram1D.transform(self.fr, transform, t0 = self.times[i], errors = False)

      self.fits[i] = BackgroundFit(transform, t0 = self.times[i], start = self.start, model = self.model)
      # print(self.fits[i].cov)
      self.fits[i].fit()
      # print(self.fits[i].model.pCov)

    minimum = min(self.fits, key = lambda fit: fit.model.chi2ndf)
    self.chi2ndf = np.array([fit.model.chi2ndf for fit in self.fits])
    optIndex = self.fits.index(minimum)

    # print(self.fits[optIndex].model.pCov)

    # If there's no minimum sufficiently inside the scan window, try again.
    if optIndex < 2 or optIndex > len(self.times) - 3:

      # Fit a parabola to the whole distribution, and estimate the minimum.
      a, b, c = np.polyfit(self.times, self.chi2ndf, 2)
      est_t0 = -b/(2*a)

      # Print an update.
      print("\nOptimal t0 not found within time window.")
      print(f"Trying again with re-estimated t0 seed: {est_t0 * 1000:.4f} ns.")

      # Make a recursive call to optimize again using the new estimate.
      if index < 5:
        self.seed = est_t0
        self.optimize(index + 1)
      else:
        self.t0 = est_t0

    # Otherwise, if there is a minimum sufficiently inside the scan window...
    else:

      # Remember the optimal fit from the scan.
      self.bgFit = self.fits[optIndex]

      # Fit a parabola to the 2 neighbors on either side of the minimum.
      popt = np.polyfit(
        self.times[(optIndex - 2):(optIndex + 3)],
        self.chi2ndf[(optIndex - 2):(optIndex + 3)],
        2
      )

      # Estimate t0 using the minimum of the parabolic fit.
      self.t0 = -popt[1] / (2 * popt[0]) # -b/2a (quadratic formula)

      # Get the value of the minimum chi2 from the fit.
      min_chi2 = np.polyval(popt, self.t0) * self.bgFit.model.ndf

      # Fit a parabola to the entire window, and extrapolate chi2_min + 1.
      a, b, c = popt * self.bgFit.model.ndf
      c -= min_chi2 + 1
      leftTime = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a) # quadratic formula
      rightTime = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a) # quadratic formula
      leftMargin = self.t0 - leftTime
      rightMargin = rightTime - self.t0
      self.err_t0 = ((leftMargin + rightMargin) / 2)

      # self.leftTransform = Transform.Transform(
      #   self.transform.fr,
      #   self.transform.start,
      #   self.transform.end,
      #   self.transform.df,
      #   self.transform.bgModel,
      #   self.t0 - self.err_t0,
      #   self.transform.n
      # )
      # self.leftTransform.process(update = False)
      self.leftTransform = Histogram1D.transform(self.fr, self.frequencies, self.t0 - self.err_t0)
      self.leftFit = BackgroundFit(self.leftTransform, self.t0 - self.err_t0, self.start, self.model).fit()
      self.leftTransform = self.leftFit.subtract()

      # self.rightTransform = Transform.Transform(
      #   self.transform.fr,
      #   self.transform.start,
      #   self.transform.end,
      #   self.transform.df,
      #   self.transform.bgModel,
      #   self.t0 + self.err_t0,
      #   self.transform.n
      # )
      # self.rightTransform.process(update = False)
      self.rightTransform = Histogram1D.transform(self.fr, self.frequencies, self.t0 + self.err_t0)
      self.rightFit = BackgroundFit(self.rightTransform, self.t0 + self.err_t0, self.start, self.model).fit()
      self.rightTransform = self.rightFit.subtract()

      # self.leftFit = self.leftTransform.bgFit
      # self.rightFit = self.rightTransform.bgFit

      # Print an update, completing this round of optimization.
      print(f"\nCompleted background optimization.")
      print(f"{'best chi2/ndf':>16} = {self.bgFit.model.chi2ndf:.4f}")
      print(f"{'new t0':>16} = {self.t0*1000:.4f} +/- {self.err_t0*1000:.4f} ns")

  # ============================================================================

  def errors(self, ref):

    results = {"err_t0": self.err_t0}

    # Propagate the uncertainty in t0 to the distribution mean and width.
    for unit in util.frequencyTo.keys():

      conversion = util.frequencyTo[unit]
      mean = ref.copy().map(conversion).mean()
      width = ref.copy().map(conversion).std()

      # Find the change in the mean after +/- one-sigma shifts in t0.
      left_mean_err = abs(mean - self.leftTransform.copy().map(conversion).mean())
      right_mean_err = abs(mean - self.rightTransform.copy().map(conversion).mean())

      # Find the change in the width after +/- one-sigma shifts in t0.
      left_width_err = abs(width - self.leftTransform.copy().map(conversion).std())
      right_width_err = abs(width - self.rightTransform.copy().map(conversion).std())

      # Average the changes on either side of the optimal t0.
      mean_err_t0 = (left_mean_err + right_mean_err) / 2
      width_err_t0 = (left_width_err + right_width_err) / 2

      results[f"err_{unit}"] = mean_err_t0
      results[f"err_sig_{unit}"] = width_err_t0

    return Results(results)

  # ============================================================================

  def plotChi2(self, output):

    # Plot the chi2/ndf.
    plt.plot(self.times * 1000, self.chi2ndf, 'o-')

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
    style.ylabel(r"Background $\chi^2$/ndf")
    plt.ylim(0, None)

    # Save the result to disk, and clear the figure.
    plt.savefig(output)
    plt.clf()

  # ============================================================================

  def plotFits(self, output):

    # Temporarily turn off LaTeX rendering for faster plots.
    latex = plt.rcParams["text.usetex"]
    plt.rcParams["text.usetex"] = False

    # Initialize the multi-page PDF file for scan plots.
    pdf = PdfPages(output)

    # Plot each background fit, updating the initialized plot each time.
    for i in range(len(self.fits)):
      self.fits[i].plot()
      pdf.savefig()
      plt.clf()

    # Close the multi-page PDF.
    pdf.close()

    # Resume LaTeX rendering, if it was enabled before.
    if latex:
      plt.rcParams["text.usetex"] = True
