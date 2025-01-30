from gm2fr.src.Histogram1D import Histogram1D
from gm2fr.src.Histogram2D import Histogram2D
import gm2fr.src.calculations as calc
import gm2fr.src.constants as const

import numpy as np
import scipy.interpolate as interp

import matplotlib.pyplot as plt
import gm2fr.src.style as style
style.set_style()

# ==================================================================================================

class Corrector:

  # ================================================================================================

  def __init__(self, transform, bg_transform, ref_filename, ref_t0, tweak = True, plain_cosine = None):

    self.tweak = tweak
    self.transform = transform
    self.bg_transform = bg_transform
    self.plain_cosine = plain_cosine

    self.ref_joint = Histogram2D.load(ref_filename, "joint").normalize()
    self.ref_tau = Histogram1D.load(ref_filename, "profile").normalize()
    self.ref_frequency = Histogram1D.load(ref_filename, "frequencies").normalize()
    #self.ref_t0 = ref_t0 if ref_t0 is not None else self.transform.t0

    self.ref_joint.errors = np.zeros(self.ref_joint.heights.shape)
    self.ref_frequency.cov = np.zeros((len(self.ref_frequency.heights), len(self.ref_frequency.heights)))
    self.ref_frequency.update_errors()

    # A(f) and B(f) coefficients, as defined in the derivation note.
    self.A = None
    self.B = None

    # The product of each coefficient and the true frequency distribution rho(f).
    self.A_rho = None
    self.B_rho = None

    self.ref_t0 = 0
    self.update()
    self.ref_t0 = np.arctan2(np.sum(self.B_rho.heights), np.sum(self.A_rho.heights)) / (2 * np.pi * const.info["f"].magic * const.kHz_us)
    self.update()
    print(f"optimized ref_t0: {self.ref_t0:.4f}")

    # The four main terms in the analytical derivation of the cosine transform.
    self.peak = None
    self.distortion = None
    self.background = None
    self.wiggle = None

    # Predicted transform from the above terms, and corrected transform after removing from result.
    self.predicted_transform = None
    self.corrected_transform = None

  # ================================================================================================

  def update(self):
    
    # Take truth distribution, map time values to A and B coefficients, and average over time.
    self.A = self.ref_joint.copy().map(x = lambda tau: calc.A(tau * 1E-3, self.ref_t0, harmonic = self.transform.harmonic)).mean(axis = 0, empty = 0)
    self.B = self.ref_joint.copy().map(x = lambda tau: calc.B(tau * 1E-3, self.ref_t0, harmonic = self.transform.harmonic)).mean(axis = 0, empty = 0)

    self.A_rho = self.A.multiply(self.ref_frequency)
    self.B_rho = self.B.multiply(self.ref_frequency)

    extra = len(self.ref_frequency.heights) // 2
    df = np.mean(self.ref_frequency.width)
    leftPad = np.arange(self.ref_frequency.edges[0] - extra * df, self.ref_frequency.edges[0], df)
    rightPad = np.arange(self.ref_frequency.edges[-1] + df, self.ref_frequency.edges[-1] + (extra + 1) * df, df)
    paddedEdges = np.concatenate((leftPad, self.ref_frequency.edges, rightPad))

    self.A_rho.edges = paddedEdges.copy()
    self.A_rho.heights = np.concatenate((np.zeros(len(leftPad)), self.A_rho.heights, np.zeros(len(rightPad))))
    self.A_rho.update_bins()
    self.A_rho.cov = np.zeros((len(self.A_rho.heights), len(self.A_rho.heights)))
    self.A_rho.update_errors()

    self.B_rho.edges = paddedEdges.copy()
    self.B_rho.heights = np.concatenate((np.zeros(len(leftPad)), self.B_rho.heights, np.zeros(len(rightPad))))
    self.B_rho.update_bins()
    self.B_rho.cov = np.zeros((len(self.B_rho.heights), len(self.B_rho.heights)))
    self.B_rho.update_errors()
  
  def correct(self, peak = True, distortion = True, background = False, wiggle = False):

    # Peak term. The factor of 1/2 comes from transforming rho(w) -> rho(f).
    self.peak = self.A_rho.multiply(0.5)

    # Distortion term.
    self.distortion = self.B_rho.convolve(
      lambda f: calc.c(f, self.transform.harmonic * self.transform.start, self.transform.end, self.transform.t0)
    )

    # Background term.
    self.background = self.A_rho.multiply(-1).convolve(
      lambda f: calc.sinc(2*np.pi*f, self.transform.harmonic * (self.transform.start - self.transform.t0) * const.kHz_us)
    )

    # Wiggle term.
    self.wiggle = self.ref_frequency.copy().clear().set_heights(
      calc.s(self.ref_frequency.centers, self.transform.start, self.transform.end, self.transform.t0)
    )

    self.A_interpolated = self.A.interpolate(self.transform.opt_cosine.centers, spline = False)
    self.B_interpolated = self.B.interpolate(self.transform.opt_cosine.centers, spline = False)
    
    if self.tweak:
      # Where the true frequency distribution is <5% of the maximum value, kill the coefficients to avoid bad behavior.
      ref_frequency_interpolated = self.ref_frequency.interpolate(self.transform.opt_cosine.centers, spline = False)
      mask = ref_frequency_interpolated.heights < 0.05 * np.max(self.ref_frequency.heights)
      self.A_interpolated.heights[mask] = 0
      self.B_interpolated.heights[mask] = 0

    # Calculate the predicted transform and corrected transform.
    self.predicted_transform = self.peak
    self.corrected_transform = self.bg_transform.copy()

    self.distortion_interpolated = self.distortion.interpolate(self.corrected_transform.centers)
    self.background_interpolated = self.background.interpolate(self.corrected_transform.centers)
    self.wiggle_interpolated = self.wiggle.interpolate(self.corrected_transform.centers)

    if distortion:
      self.predicted_transform = self.predicted_transform.add(self.distortion)
      self.corrected_transform = self.corrected_transform.subtract(self.distortion_interpolated)
    if background:
      self.predicted_transform = self.predicted_transform.add(self.background)
      self.corrected_transform = self.corrected_transform.subtract(self.background_interpolated)
    if wiggle:
      self.predicted_transform = self.predicted_transform.add(self.wiggle)
      self.corrected_transform = self.corrected_transform.subtract(self.wiggle_interpolated)

    self.A_interpolated.cov = np.diag(self.A_interpolated.cov)
    
    #bad_indices = np.argwhere((np.abs(self.A_interpolated.heights) < 0.05) | (np.abs(self.corrected_transform.heights) < 0.001 * np.max(self.corrected_transform.heights)))

    self.corrected_before_division = self.corrected_transform.copy()
    if peak:
      self.corrected_transform = self.corrected_transform.divide(self.A_interpolated, zero = 1)
      self.fix_bad_points()

    #self.corrected_transform.heights[bad_indices] = self.bg_transform.heights[bad_indices]
    # if self.tweak:
      
    # linearly interpolate across the point with the worst uncertainty
    #if np.any(np.abs(self.A_interpolated.heights) < 0.05):
    #  bad_index = np.argmax(self.corrected_transform.errors)
    #  self.corrected_transform.heights[bad_index] = np.mean(self.corrected_transform.heights[[bad_index - 1, bad_index + 1]])
    
    # zero points outside the physical range, since misbehaved points out there can mess up plotting
    self.corrected_transform.heights[const.unphysical(self.corrected_transform.centers)] = 0
    self.predicted_transform.heights[const.unphysical(self.predicted_transform.centers)] = 0

  # ================================================================================================
  
  def fix_bad_points(self):

    bad_indices = []
    for i in range(len(self.corrected_transform.heights)):
      # A_bad = (0 < np.abs(self.A_interpolated.heights[i]) < 0.05)
      A_bad = (np.abs(self.A_interpolated.heights[i]) < 0.1)
      # B_bad = (np.abs(self.corrected_before_division.heights[i]) < 0.1 * np.max(self.corrected_before_division.heights))
      #B_bad = False
      if A_bad:# and B_bad:
        if len(bad_indices) > 0 and bad_indices[-1][-1] == (i - 1):
          bad_indices[-1].append(i)
        else:
          bad_indices.append([i])

    bad_indices_flat = [i for region in bad_indices for i in region]

    print("interpolating through bad regions:", [self.corrected_transform.centers[indices] for indices in bad_indices])
    max_interpolation_length = 5
    for region in bad_indices:
      # don't interpolate across too many points. if there are too many in one contiguous region,
      # undo (and therefore omit) the division step of the correction instead.
      if len(region) > max_interpolation_length:
        print("skipping region")
        self.corrected_transform.heights[region] *= self.A_interpolated.heights[region]
        continue
      bad_centers = self.corrected_transform.centers[region]
      start, end = region[0], region[-1] + 1
      neighbor_slice = list(range(start - 3, start)) + list(range(end, end + 3))
      # only use neighbors for interpolation if they're not themselves bad points
      neighbor_slice = [i for i in neighbor_slice if i not in bad_indices_flat] 
      x_neighbors = self.corrected_transform.centers[neighbor_slice]
      y_neighbors = self.corrected_transform.heights[neighbor_slice]
      spline = interp.CubicSpline(x_neighbors, y_neighbors)
      self.corrected_transform.heights[region] = spline(bad_centers)

  # ================================================================================================

  def plot(self, output_path):

    cosine_transform = self.plain_cosine if self.plain_cosine is not None else self.bg_transform

    pdf = style.make_pdf(f"{output_path}/ReferencePlots.pdf")

    # Plot A(f) and B(f).
    style.draw_horizontal()
    self.A.plot(label = "$A(f)$", color = "C0")
    # self.A_interpolated.plot(ls = ":", color = "C0")
    self.B.plot(label = "$B(f)$", color = "C1")
    # self.B_interpolated.plot(ls = ":", color = "C1")
    plt.ylim(-1, 1)
    style.set_physical_limits()
    style.label_and_save("Frequency (kHz)", "Coefficient", pdf)

    style.draw_horizontal()
    self.A.plot(label = "$A(f)$", color = "C0")
    # self.A_interpolated.plot(ls = ":", color = "C0")
    plt.ylim(-1, 1)
    plt.twinx()
    self.corrected_before_division.plot(label = "Before Dividing $A(f)$", color = "C1")
    style.set_physical_limits()
    style.label_and_save("Frequency (kHz)", "", f"{output_path}/before_dividing.pdf")

    # Plot the scaled distributions A(f)p(f) and B(f)p(f).
    style.draw_horizontal()
    self.A_rho.plot(label = r"$A(f)\rho(f)$")
    self.B_rho.plot(label = r"$B(f)\rho(f)$")
    style.set_physical_limits()
    style.label_and_save("Frequency (kHz)", "Scaled Distribution", pdf)

    # Plot the main terms individually.
    style.draw_horizontal()
    self.peak.plot(label = "Peak", color = "C0")
    # self.distortion_interpolated.plot(ls = ":", color = "C1")
    self.background.plot(label = "Background", color = "C2")
    # self.background_interpolated.plot(ls = ":", color = "C2")
    # self.wiggle.multiply(5).plot(errors = True, color = "C3", label = "Wiggle (5x)")
    self.distortion.plot(label = "Distortion", color = "C1")
    style.set_physical_limits()
    style.label_and_save("Frequency (kHz)", "Term", pdf)

    # Plot the predicted transform and actual transform.
    style.draw_horizontal()
    self.ref_frequency.plot(label = "Ref. Distribution", scale = self.predicted_transform, ls = "--")
    self.predicted_transform.plot(label = "Predicted Transform")
    cosine_transform.plot(label = "Actual Transform")
    style.set_physical_limits()
    style.label_and_save("Frequency (kHz)", "Arbitrary Units", pdf)

    # Plot the actual transform and truth.
    style.draw_horizontal()
    self.ref_frequency.plot(label = "Ref. Distribution", scale = self.bg_transform, ls = "--")
    cosine_transform.plot(label = "Cosine Transform")
    style.set_physical_limits()
    style.label_and_save("Frequency (kHz)", "Arbitrary Units", pdf)

    # Plot the actual transform and corrected transform.
    style.draw_horizontal()
    self.ref_frequency.plot(label = "Ref. Distribution", ls = "--")
    cosine_transform.plot(label = "Cosine Transform", scale = self.ref_frequency)
    self.corrected_transform.plot(label = "Corrected Transform", scale = self.ref_frequency)
    style.set_physical_limits()
    style.label_and_save("Frequency (kHz)", "Arbitrary Units", pdf)

    pdf.close()
