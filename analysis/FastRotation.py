import numpy as np
import matplotlib.pyplot as plt

import time
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

  # ============================================================================

  # Constructor.
  def __init__(self, time, signal, error, units = "us"):

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

    # Wiggle fit object.
    self.wgFit = None

  # ============================================================================

  # TODO: add wiggle plots
  # TODO: add optional pileup histogram
  @classmethod
  def produce(cls, input, label, pileup = None, fit = "nine", n = 0.108, units = "us"):

    # Create the fast rotation signal using a wiggle fit.
    if fit is not None:

      print("\nPreparing fast rotation signal...")
      begin = time.time()

      # Get the histograms from the file.
      rootFile = root.TFile(input)
      wiggle = rootFile.Get(label)

      # Quit if the histogram is empty, or statistics are too low.
      if wiggle.GetEntries() < 5E5:
        return None

      # Subtract pileup, if provided.
      if pileup is not None:
        pileup = rootFile.Get(pileup)
        wiggle.Add(pileup, -1)

      # Pull the finely-binned wiggle plot data.
      wgSignal, wgEdges = rnp.hist2array(wiggle, return_edges = True)
      wgTime = (wgEdges[0][1:] + wgEdges[0][:-1]) / 2
      wgError = np.sqrt(np.abs(wgSignal))
      wgError[wgError == 0] = 1

      # Close the input file.
      rootFile.Close()

      if units == "ns":
        wgTime *= 1E-3
      elif units == "us":
        pass
      else:
        raise ValueError(f"Time units '{units}' not recognized.")

      # Perform the fit, using the coarsely-binned data.
      wgFit = wg.WiggleFit(
        wgTime,
        wgSignal,
        model = fit,
        n = n
      )
      wgFit.fit()

      # Divide out the wiggle fit from the fine binning, rescaled for rebinning.
      normalization = wgFit.fineResult
      frSignal = wgSignal / normalization
      frError = wgError / normalization

      print(f"\nFinished preparing fast rotation signal, in {(time.time() - begin):.2f} seconds.")

      # Construct the fast rotation object
      fr = cls(wgTime, frSignal, frError)
      fr.wgFit = wgFit
      return fr

    # No wiggle fit; interpret the data directly as a fast rotation signal.
    else:

      # Get the fast rotation histogram from the file.
      rootFile = root.TFile(input)
      fastRotation = rootFile.Get(label)

      # Convert to NumPy format.
      frSignal, edges = rnp.hist2array(fastRotation, return_edges = True)
      frTime = (edges[0][1:] + edges[0][:-1]) / 2

      # TODO: Errors not currently implemented when loading data this way.
      frError = np.ones(len(frSignal))

      # Construct the fast rotation object
      return cls(frTime, frSignal, frError)

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
