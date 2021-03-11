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
      fineWiggle = rootFile.Get(label)

      # Quit if the histogram is empty, or statistics are too low.
      if fineWiggle.GetEntries() < 5E5:
        return None

      # Subtract pileup, if provided.
      if pileup is not None:
        finePileup = rootFile.Get(pileup)
        fineWiggle.Add(finePileup, -1)

      # Pull the coarsely-binned wiggle plot data.
      coarseWiggle = fineWiggle.Rebin(149, "coarse")
      coarseSignal, coarseEdges = rnp.hist2array(coarseWiggle, return_edges = True)
      coarseTime = (coarseEdges[0][1:] + coarseEdges[0][:-1]) / 2
#      coarseError = np.sqrt(coarseSignal)
#      coarseError[coarseError == 0] = 1

      # Pull the finely-binned wiggle plot data.
      fineSignal, fineEdges = rnp.hist2array(fineWiggle, return_edges = True)
      fineTime = (fineEdges[0][1:] + fineEdges[0][:-1]) / 2
      fineError = np.sqrt(np.abs(fineSignal))
      fineError[fineError == 0] = 1

      # Close the input file.
      rootFile.Close()

      if units == "ns":
        coarseTime *= 1E-3
        fineTime *= 1E-3
      elif units == "us":
        pass
      else:
        raise ValueError(f"Time units '{units}' not recognized.")

      # Perform the fit, using the coarsely-binned data.
      wgFit = wg.WiggleFit(
        coarseTime,
        coarseSignal,
        model = fit,
        n = n
      )
      wgFit.fit()

      # Divide out the wiggle fit from the fine binning, rescaled for rebinning.
      normalization = wgFit.function(fineTime, *wgFit.pOpt) / 149
      fineSignal /= normalization
      fineError /= normalization

      print(f"\nFinished preparing fast rotation signal, in {(time.time() - begin):.2f} seconds.")

      # Construct the fast rotation object
      fr = cls(fineTime, fineSignal, fineError)
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
  # TODO: add percent error plot over same windows.
  def plot(self, output = None):

    if output is not None:

      # Plot the signal.
      plt.plot(self.time, self.signal)

      # Label the axes.
      style.xlabel(r"Time ($\mu$s)")
      style.ylabel("Intensity")

      # Save the figure over a range of time axis limits (in us).
      for end in [5, 10, 30, 50, 100, 150, 200, 300]:

        # Set the time limits.
        plt.xlim(4, end)

        # Update the intensity limits.
        view = self.signal[(self.time >= 4) & (self.time <= end)]
        plt.ylim(np.min(view), np.max(view))

        # Save the figure.
        plt.savefig(f"{output}/signal/FastRotation_{end}us.pdf")

      # Clear the figure.
      plt.clf()
