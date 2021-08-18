import numpy as np
import matplotlib.pyplot as plt

# import time
import os

import gm2fr.analysis.WiggleFit as wg
import gm2fr.analysis.BackgroundFit as bg
import gm2fr.analysis.Transform as tr
import gm2fr.utilities as util
from gm2fr.simulation.histogram import Histogram

import gm2fr.style as style
style.setStyle()

import ROOT as root
import root_numpy as rnp

# ==============================================================================

class FastRotation:

  # Constructor.
  def __init__(self, time, signal, error, fit = None, units = "us", n = 0.108):

    # Fast rotation data.
    self.time = time
    self.signal = signal
    self.error = error

    # Ensure the time units are microseconds.
    if units == "us":
      pass
    elif units == "ns":
      self.time *= 1E-3
    else:
      raise ValueError(f"Time units '{units}' not recognized.")

    # Perform a wiggle fit.
    self.wgFit = None
    if fit is not None:

      self.wgFit = wg.WiggleFit(
        self.time.copy(),
        self.signal.copy(),
        model = fit,
        n = n
      )
      self.wgFit.fit()

      # Divide out the wiggle fit from the fine binning, rescaled for rebinning.
      normalization = self.wgFit.fineResult
      self.signal /= normalization
      self.error /= normalization

  # ============================================================================

  @classmethod
  def produce(cls, input, fit = "nine", n = 0.108, units = "us"):

    # Interpret the input as ROOT, with format (filename, label, pileup).
    if type(input) is tuple:

      rootFile = root.TFile(input[0])
      histogram = rootFile.Get(input[1])

      if input[2] is not None:
        pileup = rootFile.Get(input[2])
        histogram.Add(pileup, -1)

      if histogram.GetEntries() < 5E5:
        return None

      signal, edges = rnp.hist2array(histogram, return_edges = True)
      time = (edges[0][1:] + edges[0][:-1]) / 2

      error = np.sqrt(np.abs(signal))
      error[error == 0] = 1

      rootFile.Close()

    # Interpret the input as the path to a saved Histogram object.
    else:

      histogram = Histogram.load(input, "signal")
      signal, time, error = histogram.heights, histogram.xCenters, histogram.errors
      # self.fastRotation = fr.FastRotation(h.xCenters, h.heights, h.errors, fit, self.units, n)

    if units == "ns":
      time *= 1E-3
    elif units == "us":
      pass
    else:
      raise ValueError(f"Time units '{units}' not recognized.")

    # Create the fast rotation signal using a wiggle fit.
    normalization = 1
    wgFit = None

    if fit is not None:

      # Perform the fit, using the coarsely-binned data.
      wgFit = wg.WiggleFit(
        time,
        signal,
        model = fit,
        n = n
      )
      wgFit.fit()

      # Divide out the wiggle fit from the fine binning, rescaled for rebinning.
      normalization = wgFit.fineResult

      print(f"\nFinished preparing fast rotation signal.")#", in {(time.time() - begin):.2f} seconds.")

    # Construct the fast rotation object
    fr = cls(time, signal / normalization, error / normalization)
    fr.wgFit = wgFit
    return fr

  # ============================================================================

  # Plot the fast rotation signal.
  def plot(self, output, endTimes):

    if output is not None:

      # Plot the signal.
      plt.plot(self.time, self.signal)

      # Label the axes.
      style.xlabel(r"Time ($\mu$s)")
      style.ylabel("Intensity")

      # Save the figure over a range of time axis limits (in us).
      for end in endTimes:

        # Set the time limits.
        plt.xlim(4, end)

        # Update the intensity limits.
        view = self.signal[(self.time >= 4) & (self.time <= end)]
        plt.ylim(np.min(view), np.max(view))

        # Save the figure.
        plt.savefig(f"{output}/signal/FastRotation_{end}us.pdf")

      # Clear the figure.
      plt.clf()

  # ============================================================================

  def save(self, output, end = 300):

    mask = (self.time < end)

    # Save the transform and all axis units in NumPy format.
    np.savez(
      output,
      time = self.time[mask],
      signal = self.signal[mask],
      error = self.error[mask]
    )

    # Create a ROOT histogram for the frequency distribution.
    dt = self.time[1] - self.time[0]
    histogram = root.TH1F(
      "signal",
      ";Time (#mus);Arbitrary Units",
      len(self.time[mask]),
      self.time[mask][0] - dt/2,
      self.time[mask][-1] + dt/2
    )

    # Copy the signal into the histogram.
    rnp.array2hist(self.signal[mask], histogram, errors = self.error[mask])

    # Save the histogram.
    name = output.split(".")[0]
    outFile = root.TFile(f"{name}.root", "RECREATE")
    histogram.Write()
    outFile.Close()
