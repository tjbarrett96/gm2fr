import numpy as np
import scipy.linalg as lg

import gm2fr.constants as const
import gm2fr.calculations as calc
from gm2fr.Histogram1D import Histogram1D

# ==================================================================================================

class Transform:

  # ================================================================================================

  def __init__(self, signal, start = 4, end = 250, df = 2, width = 150):

    self.full_signal = signal
    self.signal = signal.copy().mask((start, end))

    self.start = self.signal.centers[0]
    self.end = self.signal.centers[-1]
    self.scale = 1 / (np.mean(self.signal.width) * const.kHz_us)
    self.fft_resolution = self.scale / len(self.signal.heights)

    f = np.arange(const.info["f"].magic - width / 2, const.info["f"].magic + width / 2 + df, df)
    self.raw_cosine = Histogram1D(f)
    self.raw_sine = Histogram1D(f)
    self.cross_cov = None
    self.magnitude = None
    self.omega = 2 * np.pi * self.raw_cosine.centers

    self.t0 = None
    self.err_t0 = None
    self.opt_cosine = None
    self.opt_sine = None

    self.transform()

  # ================================================================================================

  def transform(self):

    # Mask the fast rotation signal data between the start and end times.
    time, signal, errors = self.signal.centers, self.signal.heights, self.signal.errors

    # Compute the cosine transform, and subtract the s(f) wiggle function.
    cosTransform = calc.np_transform(np.cos, self.raw_cosine.centers, signal, time)
    cosTransform -= self.scale * calc.s(self.raw_cosine.centers, self.start, self.end)
    self.raw_cosine.set_heights(cosTransform)

    # Compute the sine transform, and subtract the c(f) wiggle function.
    sinTransform = calc.np_transform(np.sin, self.raw_sine.centers, signal, time)
    sinTransform += self.scale * calc.c(self.raw_sine.centers, self.start, self.end)
    self.raw_sine.set_heights(sinTransform)

    # Compute the covariance matrix for both transforms.
    fDiff = self.raw_cosine.centers - self.raw_cosine.centers[0]
    cov = 0.5 * lg.toeplitz(calc.np_transform(np.cos, fDiff, errors**2, time))
    self.raw_cosine.set_cov(cov)
    self.raw_sine.set_cov(cov)

    # Compute the covariance matrix between the cosine and sine transforms.
    column = calc.np_transform(np.sin, fDiff, errors**2, time)
    self.cross_cov = -0.5 * lg.toeplitz(column, -column)

    # Scale down the transform by the discrete scale factor.
    self.raw_cosine = self.raw_cosine.divide(self.scale)
    self.raw_sine = self.raw_sine.divide(self.scale)
    self.cross_cov /= self.scale**2

    # Compute the Fourier transform magnitude (i.e. square root of the sum in quadrature).
    square_cross_cov = 4 * np.outer(self.raw_cosine.heights, self.raw_sine.heights) * self.cross_cov
    self.magnitude = self.raw_cosine.power(2).add(self.raw_sine.power(2), cov = square_cross_cov).power(0.5)

  # ================================================================================================

  def get_t0_weights(self, t0, err = 0):

    # Compute cos(self.omega * t0) and sin(self.omega * t0).
    cos_t0 = np.cos(self.omega * t0 * const.kHz_us)
    sin_t0 = np.sin(self.omega * t0 * const.kHz_us)

    # Create histograms for the cosine and sine weight factors.
    cos_weight = self.raw_cosine.copy().clear().set_heights(cos_t0)
    sin_weight = self.raw_sine.copy().clear().set_heights(sin_t0)

    # If a t0 uncertainty is provided, set the weight factors' covariance matrices.
    if err > 0:
      cos_weight.set_cov(np.outer(self.omega * sin_t0, self.omega * sin_t0) * (err * const.kHz_us)**2)
      sin_weight.set_cov(np.outer(self.omega * cos_t0, self.omega * cos_t0) * (err * const.kHz_us)**2)

    return cos_t0, sin_t0, cos_weight, sin_weight

  # ================================================================================================

  def get_cosine_at_t0(self, t0, err = 0):

    cos_t0, sin_t0, cos_weight, sin_weight = self.get_t0_weights(t0, err)

    # Compute the t0-weighted cosine and sine transforms.
    weighted_cos = cos_weight.multiply(self.raw_cosine)
    weighted_sin = sin_weight.multiply(self.raw_sine)

    # Compute the covariance matrix between the weighted cosine and sine terms.
    weighted_cross_cov = np.outer(cos_weight.heights, sin_weight.heights) * self.cross_cov
    if err > 0:
      w_cross_cov = -np.outer(self.omega * sin_t0, self.omega * cos_t0) * (err * const.kHz_us)**2
      weighted_cross_cov += np.outer(self.raw_cosine.heights, self.raw_sine.heights) * w_cross_cov

    return weighted_cos.add(weighted_sin, cov = weighted_cross_cov)

  # ================================================================================================

  def get_sine_at_t0(self, t0, err = 0):

    cos_t0, sin_t0, cos_weight, sin_weight = self.get_t0_weights(t0, err)

    # Compute the t0-weighted cosine and sine transforms.
    weighted_cos = sin_weight.multiply(self.raw_cosine)
    weighted_sin = cos_weight.multiply(self.raw_sine)

    # Compute the covariance matrix between the weighted cosine and sine terms.
    weighted_cross_cov = np.outer(sin_weight.heights, cos_weight.heights) * self.cross_cov
    if err > 0:
      w_cross_cov = -np.outer(self.omega * cos_t0, self.omega * sin_t0) * (err * const.kHz_us)**2
      weighted_cross_cov += np.outer(self.raw_cosine.heights, self.raw_sine.heights) * w_cross_cov

    return weighted_sin.subtract(weighted_cos, cov = weighted_cross_cov)

  # ================================================================================================

  def set_t0(self, t0, err = 0):
    self.t0 = t0
    self.err_t0 = err
    self.opt_cosine = self.get_cosine_at_t0(t0, err)
    self.opt_sine = self.get_sine_at_t0(t0, err)
