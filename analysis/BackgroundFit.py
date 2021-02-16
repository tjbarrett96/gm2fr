import numpy as np
import scipy.optimize as opt
import scipy.special as sp

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
  def __init__(
    self,
    # The Transform object, whose background will be fit.
    transform,
    # The fit bounds to use.
    bounds = (util.minFrequency, util.maxFrequency),
    # The uncertainty to associate with the data points.
    uncertainty = 0.005
  ):

    # Remember the transform object.
    self.transform = transform

    # Cosine transform data.
    self.frequency = transform.frequency.copy()
    self.signal = transform.signal.copy()

    # Store t0, for reference.
    self.t0 = transform.t0
    self.end = transform.end

    # Map from model names to functions.
    self.modelFunctions = {
      "parabola": self.parabola,
      "sinc": self.sinc,
      "error": self.error
    }

    # Map from model names to initial parameter guesses.
    self.modelSeeds = {
      "parabola": [1, 6703, -1],
      "sinc": [-1, 6703, None],
      "error": [-1, 6703, 12, None]
    }

    # Map from model names to parameter bounds of the form ([lower], [upper]).
    self.modelBounds = {

      "parabola": (
        [0,      util.minFrequency, -np.inf],
        [np.inf, util.maxFrequency, 0 ]
      ),

      "sinc": (
        [-np.inf, util.minFrequency, 0     ],
        [0,       util.maxFrequency, np.inf]
      ),

      "error": (
        [-np.inf, util.minFrequency, 0,      -np.inf],
        [0,       util.maxFrequency, np.inf,  np.inf]
      )

    }

    self.wiggleSeeds = [0.1]
    self.wiggleBounds = ([-np.inf], [np.inf])

    # Background fit model.
    self.model = transform.bgModel

    wiggle = True
    if wiggle:
      # Merge background function with oversampling wiggle function.
      self.function = lambda f, *p: self.modelFunctions[self.model](f, *p[:-1]) + self.wiggle(f, *p[-1:])
      self.pSeeds = self.modelSeeds[self.model] + self.wiggleSeeds
      self.pBounds = (
        self.modelBounds[self.model][0] + self.wiggleBounds[0],
        self.modelBounds[self.model][1] + self.wiggleBounds[1],
      )
    else:
      self.function = self.modelFunctions[self.model]
      self.pSeeds = self.modelSeeds[self.model]
      self.pBounds = self.modelBounds[self.model]

    # Fit options.
    self.fitBounds = bounds
    self.fitUncertainty = uncertainty
    self.fitCutoff = transform.bgCutoff

    # Get the start time gap size, for help setting initial seeds.
    kHz_us = 1E-3
    gap = (transform.start - transform.t0) * kHz_us

    # Update some of the initial seeds based on the mathematical expectation.
    if self.model == "sinc":
      self.pSeeds[2] = 1 / (2 * np.pi * gap)
      self.pBounds[0][2] = 0.95 * self.pSeeds[2]
      self.pBounds[1][2] = 1.05 * self.pSeeds[2]
    elif self.model == "error":
      self.pSeeds[3] = np.pi * gap
      self.pBounds[0][3] = 0.95 * self.pSeeds[3]
      self.pBounds[1][3] = 1.05 * self.pSeeds[3]

    # Fit data, with boundary mask applied.
    mask = (self.frequency < self.fitBounds[0]) | (self.frequency > self.fitBounds[1])
    self.fitX = self.frequency[mask]
    self.fitY = self.signal[mask]

    # Check if there's any covariance data to use in the fit.
    if (self.transform.cov == 0).all():
      self.fitCov = None
    else:
      self.fitCov = self.transform.cov[mask][:, mask]
      print(self.fitCov)
      self.fitVar = np.diag(self.fitCov)
      self.fitCorr = np.diag(1/np.sqrt(self.fitVar)) @ self.fitCov @ np.diag(1/np.sqrt(self.fitVar))
      print(self.fitCorr)
      # plt.imshow(self.fitCorr)
      # plt.colorbar()
      # plt.show()
      # Remove negative eigenvalues from floating point errors.
      vals, V = np.linalg.eig(self.fitCov)
      vals = np.real(np.maximum(vals, 1E-10))
      # print(vals)
      self.fitCov = V @ np.diag(vals) @ V.T
      # print(self.fitCov)
      # this scaling is wrong -- figure out the right rescaling
      # T = np.sqrt(self.fitVar) / np.sqrt(np.diag(BB))
      # TT = T @ T.T
      # self.fitCov = BB * TT

    # Fit results.
    self.pOpt = None
    self.pCov = None
    self.fitResult = np.zeros(len(self.frequency))

    # Fit result details.
    self.chi2 = None
    self.dof = None
    self.chi2dof = None
    self.fitResiduals = None
    self.fullResiduals = None
    self.spread = None
    self.newBounds = None

    # First indices inside the collimator bounds.
    self.lowEdge = np.argmax(transform.fMask)
    self.highEdge = (len(self.frequency) - 1) - np.argmax(np.flip(transform.fMask))

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
  def error(self, f, a, fc, s, b):
    return a * np.exp(-s**2*b**2) * np.imag(np.exp(-2j*(f-fc)*b) * sp.dawsn(-(f-fc)/s+1j*s*b))

  # ============================================================================

  # Frequency oversampling wiggle.
  def wiggle(self, f, a):
    return a / (2*np.pi*f*1E-3) * np.sin(2*np.pi*f*(self.end - self.t0)*1E-3)

  # ============================================================================

  # Perform the background fit.
  def fit(self):

    if self.fitCov is not None:
      print(np.linalg.eigvals(self.fitCov))

    # Perform the fit.
    self.pOpt, self.pCov = opt.curve_fit(
      self.function,
      self.fitX,
      self.fitY,
      p0 = self.pSeeds,
      bounds = self.pBounds,
      # sigma = np.sqrt(np.diag(self.fitCov)) if self.fitCov is not None else None,
      sigma = self.fitCov,
      absolute_sigma = True if self.fitCov is not None else False,
      maxfev = 10_000
    )

    # Evaluate the background curve over the full transform.
    self.fitResult = self.function(self.frequency, *self.pOpt)

    # Calculate the residuals in the fit region, and everywhere.
    self.fitResiduals = self.fitY - self.function(self.fitX, *self.pOpt)
    self.fullResiduals = self.signal - self.fitResult

    # Calculate the one-sigma spread in fit residuals.
    self.spread = np.std(self.fitResiduals)
    self.dof = len(self.fitX) - len(self.pOpt)

    # Calculate the chi-squared.
    if self.fitCov is not None:
      # inv = np.linalg.pinv(self.fitCov)
      # print(inv)
      self.chi2 = self.fitResiduals.T @ np.linalg.pinv(self.fitCov) @ self.fitResiduals
      print(self.fitResiduals.T @ self.fitResiduals / np.mean(np.sqrt(np.diag(self.fitCov)))**2 / self.dof)
    else:
      self.chi2 = self.fitResiduals.T @ self.fitResiduals / self.fitUncertainty**2

    # Calculate the chi-squared per degree of freedom.
    self.chi2dof = self.chi2 / self.dof

    # Predict the new fit boundaries, using the fit uncertainty.
    self.update(self.fitUncertainty)

    print([f"{x:.4f}" for x in self.pOpt])

  # ============================================================================

  # Update the fit uncertainty and chi-squared, then re-estimate the new bounds.
  def update(self, uncertainty):

    if self.fitCov is None:
      # Update the chi-squared with the new uncertainty.
      self.chi2dof *= self.fitUncertainty**2 / uncertainty**2
    self.fitUncertainty = uncertainty

    # Step inward from the low edge to find the new predicted boundary.
    lowIndex = self.lowEdge
    for i in range(lowIndex, len(self.fullResiduals) - 1):
      # Break when current point *and* next 2 higher points are beyond the cutoff.
      if (np.abs(self.fullResiduals[i:i+3]) > self.fitCutoff * self.fitUncertainty).all():
        lowIndex = i
        break

    # Step inward from the high edge to find the new predicted boundary.
    highIndex = self.highEdge
    for i in range(highIndex, 0, -1):
      # Break when current point *and* next 2 lower points are beyond the cutoff.
      if (np.abs(self.fullResiduals[i-2:i+1]) > self.fitCutoff * self.fitUncertainty).all():
        highIndex = i
        break

    # Define the new predicted fit boundaries.
    self.newBounds = (self.frequency[lowIndex], self.frequency[highIndex])

  # ============================================================================

  # Check if this fit's bounds are closer together than "other".
  def betterBounds(self, other):
    thisDiff = abs(self.newBounds[1] - self.newBounds[0])
    otherDiff = abs(other.newBounds[1] - other.newBounds[0])
    return thisDiff < otherDiff

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
      plt.plot(self.fitX, self.fitY, 'ko', label = "Background")
      plt.plot(self.frequency, self.fitResult, 'g', label = "Background Fit")

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
      plt.gca().lines[1].set_ydata(self.fitY)
      plt.gca().lines[2].set_ydata(self.fitResult)

      # Update the t0 label.
      plt.gca().findobj(matplotlib.text.Text)[0].set_text(label)

      # Rescale the y-axis.
      plt.gca().relim()
      plt.gca().autoscale()

  # ============================================================================

  # Save this background fit in NumPy format.
  def save(self, output = None):
    if output is not None:
        np.savez(
          output,
          frequency = self.frequency,
          transform = self.signal,
          fitResult = self.fitResult,
          bounds = np.array([self.fitBounds[0], self.fitBounds[1]])
        )
