from gm2fr.src.BackgroundFit import BackgroundFit
import gm2fr.src.constants as const

# import math
import numpy as np
import matplotlib.pyplot as plt
import gm2fr.src.style as style
style.set_style()
import time

# ==================================================================================================

class Optimizer:

  def __init__(
    self,
    transform,
    model,
    width = 0.001,
    steps = 10,
    seed = None,
    bg_space = None
  ):

    self.transform = transform
    self.model = model

    self.seed = seed if seed is not None else self.get_seed()
    self.steps = int(steps)
    self.width = width
    self.times = None
    self.bg_space = bg_space

    self.fits = None
    self.chi2_ndf = None

    self.t0 = None
    self.err_t0 = None

    self.iteration = 0

    self.p_opt = None
    self.opt_chi2_ndf = None

  # ================================================================================================

  # Estimate the t_0 time from the fast rotation signal.
  def get_seed(self):

    # Get the fast rotation times and heights from the histogram.
    time, signal = self.transform.full_signal.centers, self.transform.full_signal.heights

    # Select one cyclotron period as early as possible after 4 microseconds.
    start = max(4, time[signal > 0][0])
    mask = (time >= start) & (time <= start + const.info["T"].magic * 1E-3)

    # Refine the selection by taking the first local minimum as the start of the turn.
    start = time[mask][np.argmin(signal[mask])]
    end = start + const.info["T"].magic * 1E-3
    mask = (time >= start) & (time <= end)

    # Compute the average of the times within this turn.
    mid = np.average(time[mask], weights = signal[mask])

    # Subtract multiples of the cyclotron period to get near t0 ~ 0.
    periods = int(np.floor(mid / (const.info["T"].magic * 1E-3)))
    seed = mid - periods * const.info["T"].magic * 1E-3

    print(f"\nEstimated t0 seed: {seed*1E3:.2f} ns.")
    return seed

  # ================================================================================================

  def optimize(self, check_minimum = True, max_iterations = 10):

    begin = time.time()
    self.times = np.linspace(self.seed - self.width / 2, self.seed + self.width / 2, self.steps)
    self.fits = [None] * len(self.times)

    for i in range(len(self.times)):
      self.fits[i] = BackgroundFit(self.transform, self.model, t0 = self.times[i], bg_space = self.bg_space).fit()

    # Extract an array of the reduced chi-squareds from the fits.
    self.chi2_ndf = np.array([fit.model.chi2_ndf for fit in self.fits])

    # Fit the chi2/ndf to a parabola and estimate the minimum.
    self.p_opt = np.polyfit(self.times, self.chi2_ndf, 2)
    a, b, c = self.p_opt # ax^2 + bx + c
    self.t0 = -b / (2 * a)
    self.opt_chi2_ndf = np.polyval(self.p_opt, self.t0)

    # Check that we have found the optimal t0 at a well-behaved chi-squared minimum.
    if check_minimum:

      # Check that the minimum is bowl-shaped: 2nd derivative is positive everywhere in the window.
      valid_minimum = np.all(np.diff(self.chi2_ndf, 2) > 0)
      if not valid_minimum:
        # Shift the t0 seed to the left or right, depending on which side had the smallest chi2.
        self.seed += self.width * (1 if self.chi2_ndf[-1] < self.chi2_ndf[0] else -1)
        print("\nBackground fit chi-squareds do not form a smooth bowl.")
        print(f"Trying again with shifted t0 seed: {self.seed * 1000:.4f} ns.")

      # Check that the fitted chi2 minimum is contained within the window, not extrapolated.
      valid_t0 = self.times[0] < self.t0 < self.times[-1]
      if not valid_t0:
        # Update the t0 seed from the fitted parabolic minimum.
        self.seed = self.t0
        print("\nOptimal t0 extrapolated outside the time window.")
        print(f"Trying again with extrapolated t0 seed: {self.seed * 1000:.4f} ns.")

      # Only proceed to re-optimizing if we haven't exceeded the maximum number of iterations.
      if not valid_minimum or not valid_t0:
        if self.iteration + 1 < max_iterations:
          self.iteration += 1
          return self.optimize(check_minimum, max_iterations)
        else:
          print("\nMaximum number of re-optimization attempts has been exceeded.")

    # Get the value of the minimum chi2 from the fit.
    ndf = self.fits[0].model.ndf
    min_chi2 = self.opt_chi2_ndf * ndf

    # Find the t_0 values where chi2 = chi2_min + 1 on either side.
    # This means chi2(t0) = chi2_min + 1, so the roots of chi2(t0) - (chi2_min + 1) = 0.
    c -= (self.opt_chi2_ndf * ndf + 1) / ndf
    leftTime = (-b - np.sqrt(b**2 - 4*a*c)) / (2*a) # quadratic formula
    rightTime = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a) # quadratic formula
    leftMargin = self.t0 - leftTime
    rightMargin = rightTime - self.t0
    self.err_t0 = ((leftMargin + rightMargin) / 2)

    # Print an update, completing this round of optimization.
    print(f"\nCompleted background optimization in {(time.time() - begin):.2} seconds.")
    print(f"{'best chi2/ndf':>16} = {(min_chi2 / ndf):.4f} +/- {self.fits[0].model.err_chi2_ndf:.4f}")
    print(f"{'new t0':>16} = {self.t0*1000:.4f} +/- {(self.err_t0 * 1000):.4f} ns")

    return self.t0

  # ================================================================================================

  def plot_chi2(self, output = None):

    # Plot the chi2/ndf.
    plt.plot(self.times * 1000, self.chi2_ndf, 'o-')

    # Plot the quadratic fit.
    smooth_times = np.linspace(self.times[0], self.times[-1], 50)
    plt.plot(smooth_times * 1000, np.polyval(self.p_opt, smooth_times), '--')

    # Show the one-sigma t0 bounds as a shaded rectangle.
    plt.axvspan(
      (self.t0 - self.err_t0) * 1000,
      (self.t0 + self.err_t0) * 1000,
      alpha = 0.2,
      fc = "k",
      ec = None
    )

    # Plot the optimized t0 as a vertical line.
    plt.axvline(self.t0 * 1000, c = "k", ls = ":")

    # Show the horizontal reference line where chi2/ndf = 1.
    plt.axhline(1, c = "k", ls = ":")

    # Axis labels.
    # style.xlabel("$t_0$ (ns)")
    # style.ylabel(r"Background $\chi^2$/ndf")
    #
    # # Axis range.
    # yMin, yMax = plt.ylim()
    # plt.ylim(max(0, min(1 - 0.1 * (yMax - yMin), yMin)), None)
    #
    # # Save the result to disk, and clear the figure.
    # plt.savefig(output)
    # plt.clf()

    if output is not None:
      style.label_and_save("$t_0$ (ns)", r"Background $\chi^2$/ndf", output)

  # ================================================================================================

  def plot_fits(self, output):

    # Temporarily turn off LaTeX rendering for faster plots.
    latex = plt.rcParams["text.usetex"]
    plt.rcParams["text.usetex"] = False

    # Initialize the multi-page PDF file for scan plots.
    pdf = style.make_pdf(output)

    # Plot each background fit, updating the initialized plot each time.
    for i in range(len(self.fits)):
      self.fits[i].plot()
      pdf.savefig()
      plt.clf()

    # Close the multi-page PDF.
    pdf.close()

    # Resume LaTeX rendering, if it was enabled before.
    if latex:
      plt.rcParams["text.usetex"] = True
