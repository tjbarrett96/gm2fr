import numpy as np
import scipy.optimize as opt
import scipy.sparse.linalg as linalg
import gm2fr.calculations as calc
from gm2fr.analysis.Results import Results
from gm2fr.Histogram1D import Histogram1D

from scipy.sparse.linalg import LinearOperator, spilu

import matplotlib.pyplot as plt
import gm2fr.style as style
style.set_style()

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
    self.p_opt = None
    self.p_cov = None
    self.p_err = None
    self.result = None
    self.residuals = None

    # Goodness-of-fit metrics.
    self.chi2 = None
    self.ndf = None
    self.chi2_ndf = None
    self.err_chi2_ndf = None
    self.p_value = None

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
    if isinstance(x, Histogram1D):
      x = x.centers
    return self.function(x, *self.p_opt)

  # ============================================================================

  # Fit the given data using this model.
  def fit(self, x, y = None, cov = None):

    # Remember the data.
    if isinstance(x, Histogram1D) and y is None and cov is None:
      self.x = x.centers
      self.y = x.heights
      self.cov = x.cov if x.cov.ndim == 2 else x.errors
    else:
      self.x = x
      self.y = y
      self.cov = cov

    def wrapped_function(p):
      residuals = self.y - self.function(self.x, *p)
      if self.cov.ndim == 2:
        M = np.diag(1 / np.diag(self.cov))
        cov_inv_residuals, convergence = linalg.gmres(self.cov, residuals, M = M, restart = 100, atol = 1E-9)
        return residuals.T @ cov_inv_residuals
      else:
        return residuals.T @ (residuals / self.cov**2)

    opt_result = opt.minimize(wrapped_function, self.seeds, method = "BFGS")
    self.p_opt, self.p_cov = opt_result.x, opt_result.hess_inv

    # if not isinstance(self.p_cov, np.ndarray):
    #   self.p_cov = self.p_cov.todense()

    self.p_err = np.sqrt(np.diag(self.p_cov))

    # Evaluate the fit function over the data points.
    self.result = self.eval(self.x)

    # Calculate the fit residuals.
    self.residuals = self.y - self.result

    # Calculate the chi-squared.
    if self.cov is not None:

      self.err = np.sqrt(np.diag(self.cov)) if self.cov.ndim == 2 else self.cov
      self.chi2 = opt_result.fun

      # Calculate the reduced chi-squared.
      self.ndf = len(self.x) - len(self.p_opt)
      self.chi2_ndf = self.chi2 / self.ndf
      self.err_chi2_ndf = np.sqrt(2 / self.ndf) # std. dev. of reduced chi-squared distribution

      # Calculate the two-sided p-value for this chi2 & ndf.
      self.p_value = calc.p_value(self.chi2, self.ndf)

  # ============================================================================

  # Calculate the covariance matrix of the fit result at points x.
  def covariance(self, x: np.ndarray, scale = True):

    # Fit covariance in terms of parameter gradient and parameter covariance.
    result = self.gradient(x).T @ self.p_cov @ self.gradient(x)

    # Scale the (co)variance of the fit result by the reduced chi-squared,
    # to approximately incorporate the systematic uncertainty from a poor model.
    if scale and self.chi2_ndf > 1:
      result *= self.chi2_ndf

    return result

  # ============================================================================

  # One-sigma uncertainty in the fit result at points x.
  def uncertainty(self, x: np.ndarray):
    return np.sqrt(np.diag(self.covariance(x)))

  # ============================================================================

  # Plot the data with error bars and fit result with one-sigma error band.
  def plot(self, x = None, dataLabel = "Data", fitLabel = "Fit"):

    # Plot the data with errorbars.
    style.errorbar(self.x, self.y, self.err, zorder = 0, ls = "", label = dataLabel)

    # Evaluate the fit result and one-sigma error band.
    fn_x = self.x if x is None else x
    fn = self.eval(fn_x)

    # Plot the fit result and one-sigma error band.
    plt.plot(fn_x, fn, label = fitLabel)

    if self.cov is not None:
      fn_err = self.uncertainty(fn_x)
      plt.fill_between(fn_x, fn - fn_err, fn + fn_err, alpha = 0.25)

  # ============================================================================

  def results(self, prefix = "fit", parameters = True):

    results = {
      f"{prefix}_chi2": self.chi2,
      f"{prefix}_ndf": self.ndf,
      f"{prefix}_chi2_ndf": self.chi2_ndf,
      f"err_{prefix}_chi2_ndf": self.err_chi2_ndf,
      f"{prefix}_pval": self.p_value
    }

    if parameters:
      for i in range(len(self.p_opt)):
        name = f"p{i}" if self.names is None else self.names[i]
        results[f"{prefix}_{name}"] = self.p_opt[i]
        results[f"err_{prefix}_{name}"] = self.p_err[i]

    return Results(results)

# ==============================================================================

  def print(self):

    print(f"\nCompleted {self.name} fit.")

    if self.chi2_ndf is not None:
      print(f"{'chi2/ndf':>12} = {self.chi2_ndf:.4f} +/- {self.err_chi2_ndf:.4f}")
      print(f"{'p-value':>12} = {self.p_value:.4f}")

    for i in range(len(self.p_opt)):
      name = f"p{i}" if self.names is None else self.names[i]
      unit = "" if self.units is None else self.units[i]
      print(f"{name:>12} = {self.p_opt[i]:.4f} +/- {self.p_err[i]:.4f} {unit}")

# ==============================================================================

class Gaussian(Model):

  def __init__(self, scale = 1, mean = 1, width = 1):
    super().__init__()
    self.name = "gaussian"
    self.seeds = [scale, mean, width]

  def function(self, x, a, x0, s):
    return a * np.exp(-(x - x0)**2 / (2 * s**2))

  def gradient(self, x):

    result = np.zeros(shape = (len(self.seeds), len(x)))

    # The optimized parameters, and the function evaluated there.
    a, x0, s = self.p_opt
    fn = self.function(x, a, x0, s)

    # df/da
    result[0] = fn / a

    # df/d(x0)
    result[1] = fn * (x - x0) / s**2

    # df/ds
    result[2] = fn * (x - x0)**2 / s**3

    return result
