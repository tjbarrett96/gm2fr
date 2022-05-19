import numpy as np
import matplotlib.pyplot as plt
import gm2fr.style as style
style.set_style()
import gm2fr.constants as const
import gm2fr.calculations as calc
import scipy.interpolate as interp
from gm2fr.analysis.BackgroundFit import BackgroundFit
from gm2fr.analysis.BackgroundModels import Template
from gm2fr.analysis.Optimizer import Optimizer

# ==================================================================================================

class Iterator:

# ==================================================================================================

  def __init__(self, transform, bgFit, limit = 200):

    self.transform = transform

    self.iterations = [0]
    self.chi2_ndf = [bgFit.model.chi2_ndf]
    self.err_chi2_ndf = [bgFit.model.err_chi2_ndf]
    self.t0 = [transform.t0]
    self.err_t0 = [transform.err_t0]

    self.limit = limit
    self.result = transform.optCosine.subtract(bgFit.result)
    self.success = False

# ==================================================================================================

  def iterate(self, optimize = True, fineWidth = 0.001, fineSteps = 10):

    tol_chi2_ndf = 0.0001
    tol_t0 = 0.001 * 1E-3

    tempTransform = self.transform.combine_at_t0(self.t0[-1], self.err_t0[-1])

    for i in range(1, self.limit + 1):
      print(f"\nWorking on background iteration {i}.")

      background = self.result.interpolate(0.5)
      background.heights[const.unphysical(background.centers)] = 0
      background.heights[background.heights < 0] = 0
      background.normalize()
      background = background.convolve(
        lambda f: calc.sinc(2*np.pi*f, (self.transform.start - self.t0[-1]) * const.kHz_us)
      ).multiply(-self.transform.scale * background.width)

      template = interp.CubicSpline(background.centers, background.heights)
      newModel = Template(template)

      opt_t0 = self.t0[-1]
      newScan = None
      if optimize:
        newScan = Optimizer(self.transform, newModel, fineWidth, fineSteps, seed = self.t0[-1])
        newScan.optimize()
        opt_t0 = newScan.t0
        tempTransform = self.transform.combine_at_t0(opt_t0, newScan.err_t0)

      self.newBGFit = BackgroundFit(tempTransform, opt_t0, self.transform.start, newModel).fit()
      self.newBGFit.model.print()

      if i == 1:
        # On first iteration, allow chi-squared to increase within 1-sigma.
        self.success = self.newBGFit.model.chi2_ndf < self.chi2_ndf[0] + self.err_chi2_ndf[0]
      else:
        # On subsequent iterations, only proceed if the chi-squared improves.
        self.success = self.newBGFit.model.chi2_ndf < self.chi2_ndf[-1]

      # Terminate if no improvement.
      if not self.success:
        print("\nTerminating iteration: no further improvement.")
        break

      # Update the result.
      self.result = self.transform.optCosine.subtract(self.newBGFit.result)
      self.chi2_ndf.append(self.newBGFit.model.chi2_ndf)
      self.err_chi2_ndf.append(self.newBGFit.model.err_chi2_ndf)
      self.t0.append(opt_t0)
      self.err_t0.append(newScan.err_t0 if newScan is not None else 0)
      self.iterations.append(i)

      # If the change in the chi-squared is within tolerance, finish iterating.
      if abs(self.chi2_ndf[-1] - self.chi2_ndf[-2]) < tol_chi2_ndf:
        break

    self.transform.set_t0(self.t0[-1], self.err_t0[-1])
    return self.result

# ==================================================================================================

  def plot(self, output = None):

    style.errorbar(self.iterations, self.chi2_ndf, self.err_chi2_ndf, c = "C0")
    style.xlabel("Background Iteration")
    style.ylabel(r"$\chi^2$/ndf")

    style.twinx()
    style.errorbar(self.iterations, np.array(self.t0) * 1000, np.array(self.err_t0) * 1000, c = "C1")
    style.ylabel(r"$t_0$ (ns)")

    if output is not None:
      plt.savefig(output)
    plt.clf()
