import numpy as np
import scipy.optimize as opt
import gm2fr.utilities as util
from gm2fr.analysis.Results import Results

import matplotlib.pyplot as plt
import gm2fr.style as style
style.setStyle()

# ==============================================================================

class Model:

  def __init__(self):

    # Initial values and lower/upper bounds for all model parameters.
    self.seeds = None
    self.bounds = None

    # Optional names and units for parameters.
    self.name = ""
    self.names = None
    self.units = None

    # Data to fit.
    self.x = None
    self.y = None
    self.cov = None
    self.err = None

    # Fit results.
    self.pOpt = None
    self.pCov = None
    self.pErr = None
    self.result = None
    self.residuals = None

    # Goodness-of-fit metrics.
    self.chi2 = None
    self.ndf = None
    self.chi2ndf = None
    self.pval = None

  # ============================================================================

  # Main model evaluation; to be overridden by subclass.
  def function(self):
    pass

  # Gradient with respect to parameters; to be overridden by subclass.
  def gradient(self):
    pass

  # ============================================================================

  # Evaluate the fit function using the optimized parameters.
  def eval(self, x):
    return self.function(x, *self.pOpt)

  # ============================================================================

  # Fit the given data using this model.
  def fit(self, x, y, cov = None):

    # Remember the data.
    self.x = x
    self.y = y
    self.cov = cov
    self.err = np.sqrt(np.diag(cov)) if cov.ndim == 2 else cov

    # Perform the fit.
    self.pOpt, self.pCov = opt.curve_fit(
      self.function,
      self.x,
      self.y,
      sigma = self.cov,
      p0 = self.seeds,
      bounds = self.bounds if self.bounds is not None else (-np.inf, np.inf),
      absolute_sigma = True,
      maxfev = 100_000
    )

    self.pErr = np.sqrt(np.diag(self.pCov))

    # Evaluate the fit function over the data points.
    self.result = self.eval(self.x)

    # Calculate the fit residuals.
    self.residuals = self.y - self.result

    # Calculate the chi-squared.
    if self.cov.ndim == 2:
      self.chi2 = self.residuals.T @ np.linalg.solve(self.cov, self.residuals)
    else:
      self.chi2 = np.sum((self.residuals / self.cov)**2)

    # Calculate the reduced chi-squared.
    self.ndf = len(self.x) - len(self.pOpt)
    self.chi2ndf = self.chi2 / self.ndf

    # Calculate the two-sided p-value for this chi2 & ndf.
    self.pval = util.pval(self.chi2, self.ndf)

  # ============================================================================

  # Calculate the covariance matrix of the fit result at points x.
  def covariance(self, x: np.ndarray, scale = True):

    # Fit covariance in terms of parameter gradient and parameter covariance.
    result = self.gradient(x).T @ self.pCov @ self.gradient(x)

    # Scale the (co)variance of the fit result by the reduced chi-squared,
    # to approximately incorporate the systematic uncertainty from a poor model.
    if scale and self.chi2ndf > 1:
      result *= self.chi2ndf

    return result

  # ============================================================================

  # One-sigma uncertainty in the fit result at points x.
  def uncertainty(self, x: np.ndarray):
    return np.sqrt(np.diag(self.covariance(x)))

  # ============================================================================

  # Plot the data with error bars and fit result with one-sigma error band.
  def plot(self, x = None, dataLabel = "Data", fitLabel = "Fit"):

    # Plot the data with errorbars.
    style.errorbar(self.x, self.y, self.err, zorder = 0, label = dataLabel)

    # Evaluate the fit result and one-sigma error band.
    fn_x = self.x if x is None else x
    fn = self.eval(fn_x)
    fn_err = self.uncertainty(fn_x)

    # Plot the fit result and one-sigma error band.
    plt.plot(fn_x, fn, label = fitLabel)
    plt.fill_between(fn_x, fn - fn_err, fn + fn_err, alpha = 0.25)

  # ============================================================================

  def results(self, prefix = "fit", parameters = True):

    results = {
      f"{prefix}_chi2": self.chi2,
      f"{prefix}_ndf": self.ndf,
      f"{prefix}_chi2ndf": self.chi2ndf,
      f"{prefix}_pval": self.pval
    }

    if parameters:
      for i in range(len(self.pOpt)):
        name = f"p{i}" if self.names is None else self.names[i]
        results[f"{prefix}_{name}"] = self.pOpt[i]
        results[f"err_{prefix}_{name}"] = self.pErr[i]

    return Results(results)

# ==============================================================================

  def print(self):

    print(f"\nCompleted {self.name} fit.")

    print(f"{'chi2/ndf':>12} = {self.chi2ndf:.4f}")
    print(f"{'p-value':>12} = {self.pval:.4f}")

    for i in range(len(self.pOpt)):
      name = f"p{i}" if self.names is None else self.names[i]
      unit = "" if self.units is None else self.units[i]
      print(f"{name:>12} = {self.pOpt[i]:.4f} +/- {self.pErr[i]:.4f} {unit}")
