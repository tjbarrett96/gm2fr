import numpy as np
import scipy.optimize as opt
import scipy.special as sp
import scipy.linalg

import matplotlib.pyplot as plt
import matplotlib.text

from gm2fr.analysis.BackgroundModels import *
import gm2fr.utilities as util
import gm2fr.style as style
style.setStyle()

# ==============================================================================

# Class that performs the background fit and stores results.
class BackgroundFit:

  # ============================================================================

  # Constructor.
  def __init__(
    self,
    # The Transform object whose background will be fit.
    transform,
    # The cosine transform.
    signal,
    # The t0 time used.
    t0,
    # The covariance.
    cov = None,
    # Whether or not to include the end-time wiggle.
    wiggle = True
  ):

    # Keep a reference to the Transform object whose background we are fitting.
    self.transform = transform

    # The cosine transform data.
    self.frequency = transform.frequency
    self.signal = signal.copy()

    # Key times used in the cosine transform: start, end, and t0.
    self.start = transform.start
    self.end = transform.end
    self.t0 = t0

    # Fit data, with boundary mask applied.
    self.x = self.frequency[transform.unphysical]
    self.y = self.signal[transform.unphysical]

    # The covariance matrix, its inverse, the variance, and correlation matrix.
    self.cov = cov
    if self.cov is None:
      self.cov = np.diag(np.ones(len(self.frequency)))

    # Extract the variance.
    self.var = np.diag(self.cov)

    # Extract the normalized correlation matrix.
    norm = np.diag(1 / np.sqrt(self.var))
    self.corr = norm @ self.cov @ norm

    if transform.bgModel == "constant":
      self.model = Polynomial(0)
    elif transform.bgModel == "parabola":
      self.model = Polynomial(2)
    elif transform.bgModel == "sinc":
      self.model = Sinc(np.min(self.signal), self.start - self.t0)
    elif transform.bgModel == "error":
      self.model = Error(np.min(self.signal), self.start - self.t0)
    else:
      self.model = None

    if wiggle:
      self.y -= self.wiggle(self.x)

  def results(self):
    return self.model.results(prefix = "bg")

  # ============================================================================

  # Frequency oversampling wiggle.
  def wiggle(self, f):#, a):
    # return 1E3 / (2*np.pi*f*1E-3) * np.sin(2*np.pi*f*(self.end - self.t0)*1E-3)
    return 1E3 / (2*np.pi*f*1E-3) * (np.sin(2*np.pi*f*(self.end-self.t0)*1E-3) - np.sin(2*np.pi*f*(self.start-self.t0)*1E-3))

  # ============================================================================

  # Perform the background fit.
  def fit(self):

    # self.model.fit(self.x, self.y - self.wiggle(self.x), self.cov)
    self.model.fit(self.x, self.y, self.cov)
    self.result = self.model.eval(self.frequency)
    # self.results = self.model.results(prefix = "bg")
    # print(self.model.pOpt)

  # ============================================================================

  # Return the background-subtracted transform.
  def subtract(self):
    return self.signal - self.result - self.wiggle(self.frequency)

  # ============================================================================

  # Plot this background fit.
  def plot(self, output = None):

    # Plot the background and fit.
    self.model.plot(
      x = self.frequency,
      dataLabel = "Background",
      fitLabel = "Background Fit"
    )

    # Plot the central (non-background) region of the transform.
    style.errorbar(
      self.frequency[self.transform.physical],
      self.signal[self.transform.physical] - self.wiggle(self.frequency[self.transform.physical]),
      None,
      fmt = "o-",
      label = "Cosine Transform"
    )

    # Annotate the t_0 value and fit quality.
    style.databox(
      ("t_0", self.t0*1000, None, "ns"),
      (r"\chi^2/\mathrm{ndf}", self.model.chi2ndf, None, None),
      ("p", self.model.pval, None, None)
    )

    # Make the axis labels and legend.
    style.xlabel("Frequency (kHz)")
    style.ylabel("Arbitrary Units")
    plt.legend()

    # Save to disk and clear the figure, if specified.
    if output is not None:
      plt.savefig(output)
      plt.clf()

  # ============================================================================

  # Plot the correlation matrix for the background fit.
  def plotCorrelation(self, output):

    style.imshow(
      self.corr,
      label = "Correlation",
      vmin = -1,
      vmax = 1
    )

    style.xlabel(f"Frequency Bin ($\Delta f = {self.transform.df}$ kHz)")
    style.ylabel(f"Frequency Bin ($\Delta f = {self.transform.df}$ kHz)")

    plt.savefig(output)
    plt.clf()

  # ============================================================================

  # Save this background fit in NumPy format.
  def save(self, output = None):
    if output is not None:
        np.savez(
          output,
          frequency = self.frequency,
          transform = self.signal,
          fit = self.result
        )
