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

  def __init__(self, transform, bg_transform, ref_filename):

    self.transform = transform
    self.bg_transform = bg_transform

    self.ref_joint = Histogram2D.load(ref_filename, "joint")
    self.ref_frequency = Histogram1D.load(ref_filename, "frequencies").normalize()

    # A(f) and B(f) coefficients, as defined in the derivation note.
    self.A = None
    self.B = None

    # The product of each coefficient and the true frequency distribution rho(f).
    self.A_rho = None
    self.B_rho = None

    # The four main terms in the analytical derivation of the cosine transform.
    self.peak = None
    self.distortion = None
    self.background = None
    self.wiggle = None

    # Predicted transform from the above terms, and corrected transform after removing from result.
    self.predicted_transform = None
    self.corrected_transform = None

  # ================================================================================================

  def correct(self, distortion = True, background = False, wiggle = False):

    # Take truth distribution, map time values to A and B coefficients, and average over time.
    self.A = self.ref_joint.copy().map(x = lambda tau: calc.A(tau * 1E-3, self.transform.t0)).mean(axis = 0, empty = 0)
    self.B = self.ref_joint.copy().map(x = lambda tau: calc.B(tau * 1E-3, self.transform.t0)).mean(axis = 0, empty = 0)

    self.A_rho = self.A.multiply(self.ref_frequency)
    self.B_rho = self.B.multiply(self.ref_frequency)

    # Peak term. The factor of 1/2 comes from transforming rho(w) -> rho(f).
    self.peak = self.A_rho.multiply(0.5)

    # Distortion term.
    self.distortion = self.B_rho.convolve(
      lambda f: calc.c(f, self.transform.start, self.transform.end, self.transform.t0)
    )

    # Background term.
    self.background = self.A_rho.multiply(-1).convolve(
      lambda f: calc.sinc(2*np.pi*f, (self.transform.start - self.transform.t0) * const.kHz_us)
    )

    # Wiggle term.
    self.wiggle = self.ref_frequency.copy().clear().set_heights(
      calc.s(self.ref_frequency.centers, self.transform.start, self.transform.end, self.transform.t0)
    )

    self.A_interpolated = self.A.interpolate(self.transform.opt_cosine.centers, spline = False)

    # Calculate the predicted transform and corrected transform.
    self.predicted_transform = self.peak
    self.corrected_transform = self.bg_transform.copy()
    if distortion:
      self.predicted_transform = self.predicted_transform.add(self.distortion)
      self.corrected_transform = self.corrected_transform.subtract(self.distortion.interpolate(self.corrected_transform.centers))
    if background:
      self.predicted_transform = self.predicted_transform.subtract(self.background)
      self.corrected_transform = self.corrected_transform.subtract(self.background.interpolate(self.corrected_transform.centers))
    if wiggle:
      self.predicted_transform = self.predicted_transform.add(self.wiggle)
      self.corrected_transform = self.corrected_transform.subtract(self.wiggle.interpolate(self.corrected_transform.centers))

    # Manual tweak: don't scale up small numbers (<5% of max value) by large factors (>6x). These are usually just mistakes.
    # fix_mask = (abs(self.A_interpolated.heights) < 0.15) & (self.corrected_transform.heights < 0.05 * np.max(self.corrected_transform.heights))
    # fix_mask = (self.corrected_transform.heights < 0.01 * np.max(self.corrected_transform.heights))
    # self.A_interpolated.heights[fix_mask] = 0

    self.corrected_transform = self.corrected_transform.divide(self.A_interpolated, zero = 1)

  # ================================================================================================

  def plot(self, output_path):

    pdf = style.make_pdf(f"{output_path}/ReferencePlots.pdf")

    # Plot A(f) and B(f).
    style.draw_horizontal()
    self.A.plot(label = "$A(f)$")
    self.B.plot(label = "$B(f)$")
    plt.ylim(-1, 1)
    style.set_physical_limits()
    style.label_and_save("Frequency (kHz)", "Coefficient", pdf)

    # Plot the scaled distributions A(f)p(f) and B(f)p(f).
    style.draw_horizontal()
    self.A_rho.plot(label = r"$A(f)\rho(f)$")
    self.B_rho.plot(label = r"$B(f)\rho(f)$")
    style.set_physical_limits()
    style.label_and_save("Frequency (kHz)", "Scaled Distribution", pdf)

    # Plot the main terms individually.
    style.draw_horizontal()
    self.peak.plot(label = "Peak")
    self.distortion.plot(label = "Distortion")
    self.background.plot(label = "Background")
    self.wiggle.multiply(5).plot(errors = True, label = "Wiggle (5x)")
    style.set_physical_limits()
    style.label_and_save("Frequency (kHz)", "Term", pdf)

    # Plot the predicted transform and actual transform.
    style.draw_horizontal()
    self.ref_frequency.plot(label = "Ref. Distribution", scale = self.predicted_transform, ls = "--")
    self.predicted_transform.plot(label = "Predicted Transform")
    self.bg_transform.plot(label = "Actual Transform")
    style.set_physical_limits()
    style.label_and_save("Frequency (kHz)", "Arbitrary Units", pdf)

    # Plot the actual transform and truth.
    style.draw_horizontal()
    self.ref_frequency.plot(label = "Ref. Distribution", scale = self.bg_transform, ls = "--")
    self.bg_transform.plot(label = "Cosine Transform")
    style.set_physical_limits()
    style.label_and_save("Frequency (kHz)", "Arbitrary Units", pdf)

    # Plot the actual transform and corrected transform.
    style.draw_horizontal()
    self.ref_frequency.plot(label = "Ref. Distribution", ls = "--")
    self.bg_transform.plot(label = "Cosine Transform", scale = self.ref_frequency)
    self.corrected_transform.plot(label = "Corrected Transform", scale = self.ref_frequency)
    style.set_physical_limits()
    style.label_and_save("Frequency (kHz)", "Arbitrary Units", pdf)

    pdf.close()
