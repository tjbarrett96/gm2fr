from gm2fr.Histogram1D import Histogram1D
from gm2fr.Histogram2D import Histogram2D
import gm2fr.calculations as calc
import gm2fr.constants as const

import numpy as np

import matplotlib.pyplot as plt
import gm2fr.style as style
style.set_style()

# ==================================================================================================

class Corrector:

  # ================================================================================================

  def __init__(self, transform, bg_transform, truth_filename):

    self.transform = transform
    self.bg_transform = bg_transform
    self.truth_joint = Histogram2D.load(truth_filename, "joint")
    self.truth_frequency = Histogram1D.load(truth_filename, "frequencies").normalize()
    self.predicted_transform = None
    self.corrected_transform = None

    self.A = None
    self.B = None

    self.A_rho = None
    self.B_rho = None

    self.scale = None
    self.peak = None
    self.distortion = None
    self.background = None
    self.wiggle = None

  # ================================================================================================

  def correct(self, distortion = True, background = False, wiggle = False):

    # Take truth distribution, map time values to A and B coefficients, and average over time.
    self.A = self.truth_joint.copy().map(x = lambda tau: calc.A(tau * 1E-3, self.transform.t0)).mean(axis = 0, empty = 0)
    self.B = self.truth_joint.copy().map(x = lambda tau: calc.B(tau * 1E-3, self.transform.t0)).mean(axis = 0, empty = 0)

    self.A_rho = self.A.multiply(self.truth_frequency)
    self.B_rho = self.B.multiply(self.truth_frequency)

    # Calculate the four main terms with appropriate scale factors.
    # Note: the factor of 1/2 in the peak scale comes from transforming rho(w) -> rho(f).
    self.scale = 1 / (self.transform.signal.width * const.kHz_us)
    self.peak = self.A_rho.multiply(self.scale / 2)
    self.distortion = self.B_rho.convolve(lambda x: calc.c(x, self.transform.start, self.transform.end, self.transform.t0)).multiply(-self.scale * self.truth_frequency.width)
    self.background = self.A_rho.convolve(lambda x: calc.sinc(2*np.pi*x, (self.transform.start - self.transform.t0) * const.kHz_us)).multiply(-self.scale * self.truth_frequency.width)
    self.wiggle = self.truth_frequency.copy().clear().set_heights(self.scale * calc.s(self.truth_frequency.centers, self.transform.start, self.transform.end, self.transform.t0))

    A_interpolated = self.A.interpolate(self.transform.opt_cosine.centers, spline = False)
    A_interpolated.heights[abs(A_interpolated.heights) < 0.03] = 0

    # Calculate the predicted transform and corrected transform.
    self.predicted_transform = self.peak
    self.corrected_transform = self.transform.opt_cosine.copy().divide(A_interpolated, zero = 1)
    if distortion:
      self.predicted_transform = self.predicted_transform.add(self.distortion)
      self.corrected_transform = self.corrected_transform.subtract(self.distortion.interpolate(self.corrected_transform.centers))
    if background:
      self.predicted_transform = self.predicted_transform.subtract(self.background)
      self.corrected_transform = self.corrected_transform.subtract(self.background.interpolate(self.corrected_transform.centers))
    if wiggle:
      self.predicted_transform = self.predicted_transform.add(self.wiggle)
      self.corrected_transform = self.corrected_transform.subtract(self.wiggle.interpolate(self.corrected_transform.centers))

  # ================================================================================================

  def plot(self, output_path):

    # Plot A(f) and B(f).
    style.draw_horizontal()
    self.A.plot(errors = True, label = "$A(f)$")
    self.B.plot(errors = True, label = "$B(f)$")
    plt.ylim(-1, 1)
    plt.xlim(const.info["f"].min, const.info["f"].max)
    style.label_and_save("Frequency (kHz)", "Coefficient", f"{output_path}/Coefficients.pdf")

    # Plot the scaled distributions A(f)p(f) and B(f)p(f).
    style.draw_horizontal()
    self.A_rho.plot(errors = True, label = r"$A(f)\rho(f)$")
    self.B_rho.plot(errors = True, label = r"$B(f)\rho(f)$")
    plt.xlim(const.info["f"].min, const.info["f"].max)
    style.label_and_save("Frequency (kHz)", "Scaled Distribution", f"{output_path}/ScaledDistributions.pdf")

    # Plot the four main terms individually.
    style.draw_horizontal()
    self.peak.plot(errors = True, label = "Peak")
    self.distortion.plot(errors = True, label = "Distortion")
    self.background.plot(errors = True, label = "Background")
    self.wiggle.multiply(5).plot(errors = True, label = "Wiggle (5x)")
    plt.xlim(const.info["f"].min, const.info["f"].max)
    style.label_and_save("Frequency (kHz)", "Term", f"{output_path}/Terms.pdf")

    self.truth_frequency.plot(errors = False, label = "True Distribution", scale = np.max(self.predicted_transform.heights) / np.max(self.truth_frequency.heights), ls = ":")
    self.predicted_transform.plot(label = "Predicted Transform")
    self.bg_transform.plot(label = "Actual Transform")
    style.label_and_save("Frequency (kHz)", "Arbitrary Units", f"{output_path}/Predicted_vs_Actual.pdf")

    self.truth_frequency.plot(errors = False, label = "True Distribution", scale = np.max(self.corrected_transform.heights) / np.max(self.truth_frequency.heights), ls = ":")
    self.bg_transform.plot(label = "Cosine Transform")
    style.label_and_save("Frequency (kHz)", "Arbitrary Units", f"{output_path}/Truth_vs_Actual.pdf")

    self.truth_frequency.plot(label = "True Distribution", ls = "--", errors = False)
    self.bg_transform.plot(label = "Cosine Transform", scale = np.max(self.truth_frequency.heights) / np.max(self.bg_transform.heights))
    self.corrected_transform.plot(label = "Corrected Transform", scale = np.max(self.truth_frequency.heights) / np.max(self.corrected_transform.heights))
    style.draw_horizontal()
    plt.xlim(const.info["f"].min, const.info["f"].max)
    style.label_and_save("Frequency (kHz)", "Arbitrary Units", f"{output_path}/Truth_vs_Corrected.pdf")
