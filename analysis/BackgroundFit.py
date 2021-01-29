import numpy as np
import scipy.optimize as opt
import scipy.special as sp
import matplotlib.pyplot as plt

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
    frequency,
    transform,
    model = "parabola",
    gap = None, # start time gap size (us)
    bounds = (util.minFrequency, util.maxFrequency),
    uncertainty = 0.005,
    cutoff = 3
  ):
  
    # Cosine transform data.
    self.frequency = frequency.copy()
    self.transform = transform.copy()
    
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
        [0, util.minFrequency, -1],
        [1, util.maxFrequency, 0 ]
      ),
      
      "sinc": (
        [-np.inf, util.minFrequency, 0     ],
        [0,       util.maxFrequency, np.inf]
      ),
      
      "error": (
        [-np.inf, util.minFrequency, 0,      0     ],
        [0,       util.maxFrequency, np.inf, np.inf]
      )
      
    }
    
    # Fit model.
    self.model = model
    self.function = self.modelFunctions[self.model]
    self.pSeeds = self.modelSeeds[self.model]
    self.pBounds = self.modelBounds[self.model]
    
    # Fit options.
    self.fitBounds = bounds
    self.fitUncertainty = uncertainty
    self.fitCutoff = cutoff

    # Store the start time gap size, for help setting initial seeds.
    kHz_us = 1E-3
    self.gap = gap * kHz_us

    # Update some of the initial seeds based on the mathematical expectation.
    if model == "sinc":
      self.pSeeds[2] = 1 / (2 * np.pi * self.gap)
      self.pBounds[0][2] = 0.95 * self.pSeeds[2]
      self.pBounds[1][2] = 1.05 * self.pSeeds[2]
    elif model == "error":
      self.pSeeds[3] = np.pi * self.gap
      self.pBounds[0][3] = 0.95 * self.pSeeds[3]
      self.pBounds[1][3] = 1.05 * self.pSeeds[3]
    
    # Fit data, with boundary mask applied.
    mask = (self.frequency < self.fitBounds[0]) | (self.frequency > self.fitBounds[1])
    self.fitX = self.frequency[mask]
    self.fitY = self.transform[mask]
    
    # Fit results.
    self.pOpt = None
    self.pCov = None
    self.fitResult = None
    
    # Fit result details.
    self.chi2dof = None
    self.fitResiduals = None
    self.fullResiduals = None
    self.spread = None
    self.newBounds = None
    
    # Conversion factor.
    self.kHz_us = 1E-3
    
    # First indices strictly outside the collimator bounds.
    self.lowEdge = np.searchsorted(frequency, util.minimum["frequency"])
    self.highEdge = np.searchsorted(frequency, util.maximum["frequency"])
    
  # ============================================================================
    
  # Parabolic fit function.
  def parabola(self, f, a, b, c):
    return a * (f - b)**2 + c
      
  # ============================================================================

  # Sinc fit function.
  def sinc(self, f, a, fc, s):
    # np.sinc == sin(pi*x)/(pi*x), so remove the factor of pi first.
    return a * np.sinc(1/np.pi * (f-fc)/s)

  # ============================================================================

  # Error fit function.
  # TODO: clean this up!
  def error(self, f, a, fc, s, b):
    # return a * np.exp(-(x-b)**2/c**2) * np.imag(sp.erfi(-(x-b)/c + 1.0j*c*d))
    return a * np.imag(sp.dawsn(-(f-fc)/s+1j*s*b)*np.exp(-(2j*(f-fc)*b + s**2*b**2)))
  
  # ============================================================================

  # Perform the background fit.
  def fit(self):
      
    # Perform the fit.
    self.pOpt, self.pCov = opt.curve_fit(
      self.function,
      self.fitX,
      self.fitY,
      p0 = self.pSeeds,
      bounds = self.pBounds,
      maxfev = 10_000
    )
    
    # Evaluate the background curve over the full transform.
    self.fitResult = self.function(self.frequency, *self.pOpt)
    
    # Calculate the residuals in the fit region, and everywhere.
    self.fitResiduals = self.fitY - self.function(self.fitX, *self.pOpt)
    self.fullResiduals = self.transform - self.fitResult
    
    # Calculate the one-sigma spread in fit residuals.
    self.spread = np.std(self.fitResiduals)
    
    # Calculate the chi-squared, using the supplied uncertainty.
    chi2 = self.fitResiduals.T @ self.fitResiduals / self.fitUncertainty**2
    self.chi2dof = chi2 / (len(self.fitX) - len(self.pOpt))

    # Predict the new fit boundaries.
    self.predictBounds()
    
    # print([f"{x:.4f}" for x in self.pOpt])
    
  # ============================================================================
  
  # Plot the background fit.
  # TODO: make the start time & t0 arguments in the constructor, to have them available anywhere instead of passing here
  # TODO: add text label annotation (incl. list of lines) to style file
  def plot(
    self,
    standalone = True, # if True, make a fresh plot (default); if False, update the supplied handles for existing plot objects
    output = None,
    label = None,
    tfLine = None, # existing line object for cosine transform, for fast updating
    bgLine = None, # existing line object for background data, for fast updating
    fitLine = None, # existing line object for background fit, for fast updating
    labelObj = None # existing text object for t0 label, for fast updating
  ):
    
    # Make a plot from scratch.
    if standalone:
      
      # Plot the cosine transform.
      plt.plot(
        self.frequency,
        self.transform,
        'o-',
        ms = 4,
        label = "Cosine Transform"
      )
      
      # Plot the background data points.
      style.errorbar(
        self.fitX,
        self.fitY,
        self.fitUncertainty,
        fmt = 'ko',
        label = "Background"
      )
      
      # Plot the background fit curve.
      plt.plot(
        self.frequency,
        self.fitResult,
        'g',
        label = "Background Fit"
      )
      
      # Make a t0 label.
      if label is not None:
        plt.text(self.frequency[0], 1, label, ha = "left", va = "top")
      
      # Plot labels.
      plt.legend()
      style.xlabel("Frequency (kHz)")
      style.ylabel("Arbitrary Units")
    
      # Save, if specified.
      if output is not None:
        plt.savefig(output)
        
    # Update the existing plot objects from the supplied handles, for speed.
    else:
        
      # Plot the cosine transform.
      if tfLine is not None:
        tfLine.set_ydata(self.transform)
    
      # Plot the background data points.
      if bgLine is not None:
        bgLine.set_ydata(self.fitY)
    
      # Plot the background fit curve.
      if fitLine is not None:
        fitLine.set_ydata(self.fitResult)

      # Make a t0 label.
      if labelObj is not None and label is not None:
        labelObj.set_text(label)
      
      # Rescale the y-axis.
      plt.gca().relim()
      plt.gca().autoscale()
  
  # ============================================================================
  
  # Predict the new fit bounds based on the current residuals.
  def predictBounds(self):
    
    # Step inward from the low edge to find the new predicted boundary.
    # Break when current point *and* next higher point are both beyond the cutoff.
    lowIndex = self.lowEdge
    for i in range(lowIndex, len(self.fullResiduals) - 1):
      if (np.abs(self.fullResiduals[i:i+2]) > self.fitCutoff * self.fitUncertainty).all():
        lowIndex = i
        break
      
    # Step inward from the high edge to find the new predicted boundary.
    # Break when current point *and* next lower point are both beyond the cutoff.
    highIndex = self.highEdge
    for i in range(highIndex, 0, -1):
      if (np.abs(self.fullResiduals[i-1:i+1]) > self.fitCutoff * self.fitUncertainty).all():
        highIndex = i
        break
    
    # Define the new predicted fit boundaries.
    self.newBounds = (self.frequency[lowIndex], self.frequency[highIndex])
  
  # ============================================================================
  
  # Check if this fit's bounds are closer together than "other".
  def betterBounds(self, other):
#    left = (other.newBounds[0] >= self.newBounds[0])
#    right = (other.newBounds[1] <= self.newBounds[1])
#    return (left and right)
    return abs(self.newBounds[1] - self.newBounds[0]) <= abs(other.newBounds[1] - other.newBounds[0])
    
