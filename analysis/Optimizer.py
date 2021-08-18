from gm2fr.analysis.BackgroundFit import BackgroundFit
import gm2fr.analysis.Transform as Transform
import gm2fr.analysis.FastRotation
from gm2fr.analysis.Results import Results

import numpy as np
import matplotlib.pyplot as plt
import gm2fr.style as style
style.setStyle()
from matplotlib.backends.backend_pdf import PdfPages

# ==============================================================================

class Optimizer:

  def __init__(
    self,
    transform,
    seed = 0.060,
    step = 0.000025,
    width = 0.001
  ):

    self.transform = transform

    self.seed = seed
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

  def optimize(self, index = 0):

    self.times = np.arange(self.seed - self.width/2, self.seed + self.width/2, self.step)
    self.fits = [None] * len(self.times)

    cov = self.transform.covariance(self.seed, mask = self.transform.unphysical)

    for i in range(len(self.times)):

      signal = self.transform.transform(self.times[i])

      self.fits[i] = BackgroundFit(self.transform, signal, self.times[i], cov)
      self.fits[i].fit()

    minimum = min(self.fits, key = lambda fit: fit.model.chi2ndf)
    self.chi2ndf = np.array([fit.model.chi2ndf for fit in self.fits])
    optIndex = self.fits.index(minimum)

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

      self.leftTransform = Transform.Transform(
        self.transform.fr,
        self.transform.start,
        self.transform.end,
        self.transform.df,
        self.transform.bgModel,
        self.t0 - self.err_t0,
        self.transform.n
      )
      self.leftTransform.process(update = False)

      self.rightTransform = Transform.Transform(
        self.transform.fr,
        self.transform.start,
        self.transform.end,
        self.transform.df,
        self.transform.bgModel,
        self.t0 + self.err_t0,
        self.transform.n
      )
      self.rightTransform.process(update = False)

      self.leftFit = self.leftTransform.bgFit
      self.rightFit = self.rightTransform.bgFit

      # Print an update, completing this round of optimization.
      print(f"\nCompleted background optimization.")
      print(f"{'best chi2/ndf':>16} = {self.bgFit.model.chi2ndf:.4f}")
      print(f"{'new t0':>16} = {self.t0*1000:.4f} +/- {self.err_t0*1000:.4f} ns")

  # ============================================================================

  def errors(self, ref):

    results = {"err_t0": self.err_t0}

    # Propagate the uncertainty in t0 to the distribution mean and width.
    for axis in ref.axes.keys():

      mean, width = ref.getMean(axis), ref.getWidth(axis)

      # Find the change in the mean after +/- one-sigma shifts in t0.
      left_mean_err = abs(mean - self.leftTransform.getMean(axis))
      right_mean_err = abs(mean - self.rightTransform.getMean(axis))

      # Find the change in the width after +/- one-sigma shifts in t0.
      left_width_err = abs(width - self.leftTransform.getWidth(axis))
      right_width_err = abs(width - self.rightTransform.getWidth(axis))

      # Average the changes on either side of the optimal t0.
      mean_err_t0 = (left_mean_err + right_mean_err) / 2
      width_err_t0 = (left_width_err + right_width_err) / 2

      results[f"err_{axis}"] = mean_err_t0
      results[f"err_sig_{axis}"] = width_err_t0

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
