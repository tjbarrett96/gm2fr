import numpy as np
import scipy.linalg as lg

import gm2fr.constants as const
import gm2fr.calculations as calc
from gm2fr.Histogram1D import Histogram1D

# ==================================================================================================

class Transform:

  # ================================================================================================

  def __init__(self, signal, df = 2, width = 150):

    self.signal = signal
    self.start = self.signal.centers[0]
    self.end = self.signal.centers[-1]
    self.scale = 1 / (signal.width * const.kHz_us)

    f = np.arange(const.info["f"].magic - width / 2, const.info["f"].magic + width / 2 + df, df)
    self.rawCosine = Histogram1D(f)
    self.rawSine = Histogram1D(f)
    self.crossCov = None
    self.magnitude = None

    self.t0 = None
    self.err_t0 = None
    self.optCosine = None
    self.optSine = None

    self.transform()

  # ================================================================================================

  def transform(self):

    # Mask the fast rotation signal data between the start and end times.
    # mask = (self.signal.centers >= self.start) & (self.signal.centers <= self.end)
    # time = self.signal.centers[mask]
    # signal = self.signal.heights[mask]
    # errors = self.signal.errors[mask]
    time, signal, errors = self.signal.centers, self.signal.heights, self.signal.errors

    # Compute the cosine transform, and subtract the s(f) wiggle function.
    cosTransform = calc.npTransform(np.cos, self.rawCosine.centers, signal, time)
    cosTransform -= self.scale * calc.s(self.rawCosine.centers, self.start, self.end)
    self.rawCosine.setHeights(cosTransform)

    # Compute the sine transform, and subtract the c(f) wiggle function.
    sinTransform = calc.npTransform(np.sin, self.rawSine.centers, signal, time)
    sinTransform -= self.scale * calc.c(self.rawSine.centers, self.start, self.end)
    self.rawSine.setHeights(sinTransform)

    # Compute the covariance matrix for both transforms.
    fDiff = self.rawCosine.centers - self.rawCosine.centers[0]
    cov = 0.5 * lg.toeplitz(calc.npTransform(np.cos, fDiff, errors**2, time))
    self.rawCosine.setCov(cov)
    self.rawSine.setCov(cov)

    # Compute the covariance matrix between the cosine and sine transforms.
    column = calc.npTransform(np.sin, fDiff, errors**2, time)
    self.crossCov = -0.5 * lg.toeplitz(column, -column)

    # Compute the Fourier transform magnitude (i.e. square root of the sum in quadrature).
    squareCrossCov = 4 * np.outer(self.rawCosine.heights, self.rawSine.heights) * self.crossCov
    self.magnitude = self.rawCosine.power(2).add(self.rawSine.power(2), cov = squareCrossCov).power(0.5)

  # ================================================================================================

  def combineAtT0(self, t0, err = 0, sine = False):

    # Compute cos(omega * t0) and sin(omega * t0).
    omega = 2 * np.pi * self.rawCosine.centers
    cos_t0 = np.cos(omega * t0 * const.kHz_us)
    sin_t0 = np.sin(omega * t0 * const.kHz_us)

    # Create histograms for the cosine and sine weight factors.
    cosWeight = self.rawCosine.copy().clear().setHeights(cos_t0)
    sinWeight = self.rawSine.copy().clear().setHeights(sin_t0)

    # If a t0 uncertainty is provided, set the weight factors' covariance matrices.
    if err > 0:
      cosWeight.setCov(np.outer(omega * sin_t0, omega * sin_t0) * (err * const.kHz_us)**2)
      sinWeight.setCov(np.outer(omega * cos_t0, omega * cos_t0) * (err * const.kHz_us)**2)

    if not sine:

      # Compute the t0-weighted cosine and sine transforms.
      weightedCos = cosWeight.multiply(self.rawCosine)
      weightedSin = sinWeight.multiply(self.rawSine)

      # Compute the covariance matrix between the weighted cosine and sine terms.
      weightedCrossCov = np.outer(cosWeight.heights, sinWeight.heights) * self.crossCov
      if err > 0:
        wCrossCov = -np.outer(omega * sin_t0, omega * cos_t0) * (err * const.kHz_us)**2
        weightedCrossCov += np.outer(self.rawCosine.heights, self.rawSine.heights) * wCrossCov

      return weightedCos.add(weightedSin, cov = weightedCrossCov)

    else:

      # Compute the t0-weighted cosine and sine transforms.
      weightedCos = sinWeight.multiply(self.rawCosine)
      weightedSin = cosWeight.multiply(self.rawSine)

      # Compute the covariance matrix between the weighted cosine and sine terms.
      weightedCrossCov = np.outer(sinWeight.heights, cosWeight.heights) * self.crossCov
      if err > 0:
        wCrossCov = -np.outer(omega * cos_t0, omega * sin_t0) * (err * const.kHz_us)**2
        weightedCrossCov += np.outer(self.rawCosine.heights, self.rawSine.heights) * wCrossCov

      return weightedSin.subtract(weightedCos, cov = weightedCrossCov)

  # ================================================================================================

  def setT0(self, t0, err = 0):
    self.t0 = t0
    self.err_t0 = err
    self.optCosine = self.combineAtT0(t0, err)
    self.optSine = self.combineAtT0(t0, err, sine = True)
