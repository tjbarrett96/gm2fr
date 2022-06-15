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
    t0,
    start,
    model,
    err_t0 = 0
  ):

    # Keep a reference to the Transform object whose background we are fitting.
    self.transform = transform.copy()

    # Key times used in the cosine transform: start, end, and t0.
    self.t0 = t0
    self.err_t0 = err_t0
    self.start = start

    # Fit data, with boundary mask applied.
    self.mask = const.unphysical(self.transform.centers)
    self.x = self.transform.centers[self.mask]
    self.y = self.transform.heights[self.mask]

    # The covariance matrix, its inverse, the variance, and correlation matrix.
    self.cov = self.transform.cov if self.transform.cov.ndim == 2 else np.diag(self.transform.cov)
    self.cov = self.cov[self.mask][:, self.mask]

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
      self.model = Sinc(0.001, self.start - self.t0)
    elif model == "error":
      # self.model = Error(np.min(self.transform.heights), self.start - self.t0)
      self.model = Error(self.start - self.t0)
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
    self.model.fit(self.x, self.y, np.sqrt(np.diag(self.cov)))

    # Update the fit parameter seeds using the above result, then use the full covariance matrix.
    self.model.seeds = self.model.p_opt
    self.model.fit(self.x, self.y, self.cov)

    self.result = Histogram1D(
      self.transform.edges,
      heights = self.model.eval(self.transform.centers),
      cov = self.model.covariance(self.transform.centers)
    )
    return self

  # ============================================================================

  # Plot this background fit.
  def plot(self, output = None):

    # Plot the background and fit.
    self.model.plot(
      x = self.transform.centers,
      dataLabel = "Background",
      fitLabel = "Background Fit"
    )

    style.draw_horizontal()

    # Plot the central (non-background) region of the transform.
    style.errorbar(
      self.transform.centers[~self.mask],
      self.transform.heights[~self.mask],
      None,
      ls = "-",
      label = "Cosine Transform"
    )

    # Annotate the t_0 value and fit quality.
    style.databox(
      style.Entry(self.t0 * 1E3, "t_0", self.err_t0 * 1E3 if self.err_t0 > 0 else None, "ns"),
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