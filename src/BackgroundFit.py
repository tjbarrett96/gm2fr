import numpy as np
import copy

import matplotlib.pyplot as plt

from gm2fr.src.BackgroundModels import *
import gm2fr.src.constants as const
import gm2fr.src.style as style
style.set_style()
from gm2fr.src.Histogram1D import Histogram1D

# ==============================================================================

# Class that performs the background fit and stores results.
class BackgroundFit:

  # ============================================================================

  # Constructor.
  def __init__(
    self,
    transform,
    model,
    t0 = None, # use Transform object's current t0 value if None, else override with given value
    bg_space = None
  ):

    # The Transform object whose background we are fitting.
    self.transform = transform

    # If t0 is provided, then use it, or else copy the optimal t0 from the Transform object.
    if t0 is not None:
      self.t0, self.err_t0 = t0, None
      self.cosine_histogram = self.transform.get_cosine_at_t0(t0)
    else:
      self.t0, self.err_t0 = self.transform.t0, self.transform.err_t0
      self.cosine_histogram = self.transform.opt_cosine

    # The gap between the t0 time and the transform start time.
    self.start_gap = self.transform.harmonic * (self.transform.start - self.t0)\

    # If bg_space is provided, then use it, or else use the physical frequency limit.
    if bg_space is not None:
      self.bg_space = bg_space
    else:
      self.bg_space = int(round(const.info["f"].max - const.info["f"].magic))

    # Get the x, y, and covariance values from the transform histogram.
    self.x = self.cosine_histogram.centers
    self.y = self.cosine_histogram.heights
    self.cov = self.cosine_histogram.cov

    # Apply boundary mask to the fit data.
    left_boundary = const.info["f"].magic - bg_space
    right_boundary = const.info["f"].magic + bg_space
    self.mask = (self.x <= left_boundary) | (self.x >= right_boundary)
    self.masked_x = self.x[self.mask]
    self.masked_y = self.y[self.mask]
    self.masked_cov = self.cov[self.mask][:, self.mask]

    # Extract the variance and normalized correlation matrix.
    # self.var = np.diag(self.cov)
    # norm = np.diag(1 / np.sqrt(self.var))
    # self.corr = norm @ self.cov @ norm

    if model == "constant":
      self.model = Polynomial(0)
    elif model == "parabola":
      self.model = Polynomial(2)
    elif model == "sinc":
      # self.model = Sinc(np.min(self.transform.heights), self.start - self.t0)
      self.model = Sinc(0.001, self.start_gap)
      # self.model.seeds[0] /= self.harmonic**2
    elif model == "error":
      # self.model = Error(np.min(self.transform.heights), self.start - self.t0)
      self.model = Error(self.start_gap)
      # self.model.seeds[0] /= self.harmonic**2
    elif isinstance(model, Template):
      self.model = copy.deepcopy(model)
    else:
      self.model = None

    self.result = None

  def results(self):
    return self.model.results(prefix = "bg")

  # ============================================================================

  # Perform the background fit.
  def fit(self):

    # First, fit using only the diagonal of the covariance matrix, which is more stable.
    self.model.fit(self.masked_x, self.masked_y, np.sqrt(np.diag(self.masked_cov)))

    # Update the fit parameter seeds using the above result, then use the full covariance matrix.
    self.model.seeds = self.model.p_opt
    self.model.fit(self.masked_x, self.masked_y, self.masked_cov)

    self.result = Histogram1D(
      self.cosine_histogram.edges,
      heights = self.model.eval(self.x),
      cov = self.model.covariance(self.x)
    )
    return self

  # ============================================================================

  # Plot this background fit.
  def plot(self, output = None, fill_errors = True):

    # Plot the background and fit.
    self.model.plot(
      x = self.x,
      dataLabel = "Background",
      fitLabel = "Background Fit",
      fill_errors = fill_errors
    )

    style.draw_horizontal()

    # Plot the central (non-background) region of the transform.
    style.errorbar(
      self.x[~self.mask],
      self.y[~self.mask],
      None,
      ls = "-",
      label = "Cosine Transform"
    )

    # Annotate the t_0 value and fit quality.
    style.databox(
      style.Entry(self.t0 * 1E3, "t_0", self.err_t0 * 1E3 if self.err_t0 else None, "ns"),
      style.Entry(self.model.chi2_ndf, r"\chi^2/\mathrm{ndf}", self.model.err_chi2_ndf, None),
      style.Entry(self.model.p_value, "p", None, None)
    )

    # Save to disk and clear the figure, if specified.
    if output is not None:
      style.label_and_save("Frequency (kHz)", "Arbitrary Units", output)

  # ============================================================================

  # Plot the correlation matrix for the background fit.
  # def plotCorrelation(self, output):
  #
  #   style.imshow(
  #     self.corr,
  #     label = "Correlation",
  #     vmin = -1,
  #     vmax = 1
  #   )
  #
  #   style.xlabel(f"Frequency Bin ($\Delta f = {self.transform.width}$ kHz)")
  #   style.ylabel(f"Frequency Bin ($\Delta f = {self.transform.width}$ kHz)")
  #
  #   plt.savefig(output)
  #   plt.clf()

  # ============================================================================

  # Save this background fit in NumPy format.
  # def save(self, output = None):
  #   if output is not None:
  #       np.savez(
  #         output,
  #         frequency = self.frequency,
  #         transform = self.signal,
  #         fit = self.result
  #       )
