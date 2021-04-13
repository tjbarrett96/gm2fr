import numpy as np
import scipy.optimize as opt
import scipy.special as sp
import scipy.linalg

import matplotlib.pyplot as plt
import matplotlib.text

import gm2fr.utilities as util
import gm2fr.style as style
style.setStyle()

# ==============================================================================

# Class that performs the background fit and stores results.
class BackgroundFit:

  # ============================================================================

  # Constructor.
  # TODO: accept specific arguments, including cov and invCov.
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

    # Fit results.
    self.model = transform.bgModel
    self.pOpt = None
    self.pCov = None
    self.result = np.zeros(len(self.frequency))

    # Fit result details.
    self.sse = None
    self.chi2 = None
    self.ndf = None
    self.chi2ndf = None
    self.residuals = None
    self.spread = None
    self.pval = None

    # The covariance matrix, its inverse, the variance, and correlation matrix.
    self.cov = cov
    if self.cov is None:
      self.cov = np.diag(np.ones(len(self.frequency)))

    # eigenvalues, eigenvectors = np.linalg.eig(self.cov)
    # if (eigenvalues < 0).any():
    #   print("Fixing...")
    #   eigenvalues = np.real(np.maximum(eigenvalues, 1E-10))
    #   self.cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Extract the variance.
    self.var = np.diag(self.cov)

    # Extract the normalized correlation matrix.
    norm = np.diag(1 / np.sqrt(self.var))
    self.corr = norm @ self.cov @ norm

    # Map from model names to functions.
    self.modelFunctions = {
      "constant": self.constant,
      "parabola": self.parabola,
      "sinc": self.sinc,
      "error": self.error
    }

    # Map from model names to initial parameter guesses.
    self.modelSeeds = {
      "constant": [np.min(self.signal)],
      "parabola": [1, 6703, np.min(self.signal)],
      "sinc": [np.min(self.signal), 6703, None],
      "error": [np.min(self.signal), 6703, 12]#, None]
    }

    # Map from model names to parameter bounds of the form ([lower], [upper]).
    self.modelBounds = {
      "constant": (
        [-np.inf],
        [0]
      ),
      "parabola": (
        [0,      util.min["f"], -np.inf],
        [np.inf, util.max["f"], 0 ]
      ),
      "sinc": (
        [-np.inf, util.min["f"], 0     ],
        [0,       util.max["f"], np.inf]
      ),
      "error": (
        [-np.inf, util.min["f"], 0     ],#,  0  ],
        [0,       util.max["f"], np.inf]#, 0.2]
      )
    }

    # The chosen fit function, parameter seeds, and parameter bounds.
    self.function = self.modelFunctions[self.model]
    self.pSeeds = self.modelSeeds[self.model]
    self.pBounds = self.modelBounds[self.model]

    # Add the end-time wiggle function to the background model.
    if wiggle:
      self.function = lambda f, *p: self.modelFunctions[self.model](f, *p) + self.wiggle(f)

    # Get the start time gap size, for help setting initial seeds.
    kHz_us = 1E-3
    gap = (transform.start - transform.t0) * kHz_us

    # Update some of the initial seeds based on the mathematical expectation.
    if self.model == "sinc":
      self.pSeeds[2] = 1 / (2 * np.pi * gap)
      self.pBounds[0][2] = 0.90 * self.pSeeds[2]
      self.pBounds[1][2] = 1.10 * self.pSeeds[2]

    self.results = []

  # ============================================================================

  # Constant fit function.
  def constant(self, f, a):
    return a

  # ============================================================================

  # Parabolic fit function.
  def parabola(self, f, a, b, c):
    return (f - b)**2 / a + c

  # ============================================================================

  # Sinc fit function.
  def sinc(self, f, a, fc, s):
    # np.sinc == sin(pi*x)/(pi*x), so remove the factor of pi first.
    return a * np.sinc(1/np.pi * (f-fc)/s)

  # ============================================================================

  # Error fit function.
  def error(self, f, a, fc, s):#, b):

    b = np.pi * (self.start - self.t0) * 1E-3
    result =  a * np.exp(-(s*b)**2) * np.imag(np.exp(-2j*(f-fc)*b) * sp.dawsn(-(f-fc)/s+1j*s*b))

    # This function often misbehaves while exploring parameter space.
    # Specifically, both np.exp(...) and np.imag(...) can blow up, but are supposed to (partially) cancel.
    # But it doesn't work out numerically, and we just get np.inf or np.nan.
    # I'd like an elegant solution to this problem, but for now just catch these errors.
    if (np.isinf(result)).any() or (np.isnan(result)).any():
      return a

    return result

  # ============================================================================

  # Frequency oversampling wiggle.
  def wiggle(self, f):#, a):
    return 1E3 / (2*np.pi*f*1E-3) * np.sin(2*np.pi*f*(self.end - self.t0)*1E-3)

  # ============================================================================

  # Perform the background fit.
  def fit(self):

    # Perform the fit.
    self.pOpt, self.pCov = opt.curve_fit(
      self.function,
      self.x,
      self.y,
      p0 = self.pSeeds,
      # bounds = self.pBounds,
      sigma = self.cov,
      absolute_sigma = True,
      maxfev = 100_000
    )

    # Evaluate the background curve over the full transform.
    self.result = self.function(self.frequency, *self.pOpt)

    # Calculate the residuals in the fit region, and everywhere.
    self.residuals = self.y - self.function(self.x, *self.pOpt)

    # Calculate the one-sigma spread in fit residuals.
    self.spread = np.std(self.residuals)

    # Calculate the chi-squared.
    self.ndf = len(self.x) - len(self.pOpt)
    self.chi2 = self.residuals.T @ np.linalg.solve(self.cov, self.residuals)
    self.chi2ndf = self.chi2 / self.ndf

    # Calculate the two-sided p-value for this chi2 & ndf.
    self.pval = util.pval(self.chi2, self.ndf)

    # TODO: add debug print statements here, displaying everything
    print([f"{p:.4f}" for p in self.pOpt])

    # Append each key result as a (name, value) pair to the results list.
    self.results.append(("bg_chi2", self.chi2))
    self.results.append(("bg_ndf", self.ndf))
    self.results.append(("bg_chi2ndf", self.chi2ndf))
    self.results.append(("bg_pval", self.pval))
    for i, parameter in enumerate(self.pOpt):
      self.results.append((f"bg_p{i}", parameter))

  # ============================================================================

  # Return the background-subtracted transform.
  def subtract(self):
    return self.signal - self.result

  # ============================================================================

  # Plot this background fit.
  # TODO: add text label annotation (incl. list of lines) to style file
  def plot(
    self,
    # Path to desired output file.
    output = None,
    # Assume plot objects exist already in plt.gca(), and update them for speed.
    update = False
  ):

    # Make a text label for t0.
    label = f"$t_0 = {self.t0*1000:.4f}$ ns"

    # Make a plot from scratch.
    if not update:

      # Plot the transform, background points, and background fit.
      plt.plot(self.frequency, self.signal, 'o-', label = "Cosine Transform")
      plt.plot(self.x, self.y, 'ko', label = "Background")
      plt.plot(self.frequency, self.result, 'g', label = "Background Fit")

      # Display the t0 label.
      plt.text(0.04, 0.95, label, ha = "left", va = "top", transform = plt.gca().transAxes)

      # Make the axis labels and legend.
      style.xlabel("Frequency (kHz)")
      style.ylabel("Arbitrary Units")
      plt.legend()

      # Save to disk and clear the figure, if specified.
      if output is not None:
        plt.savefig(output)
        plt.clf()

    # Update the existing plot objects for speed, assuming the above order.
    else:

      # Update the transform, background points, and background fit.
      plt.gca().lines[0].set_ydata(self.signal)
      plt.gca().lines[1].set_ydata(self.y)
      plt.gca().lines[2].set_ydata(self.result)

      # Update the t0 label.
      plt.gca().findobj(matplotlib.text.Text)[0].set_text(label)

      # Rescale the y-axis.
      plt.gca().relim()
      plt.gca().autoscale()

  # ============================================================================

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
