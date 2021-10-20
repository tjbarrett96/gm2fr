import numpy as np
import numpy.lib.recfunctions as rec
import matplotlib.pyplot as plt
import scipy.optimize as opt
from matplotlib.backends.backend_pdf import PdfPages

import gm2fr.utilities as util
import gm2fr.style as style
from gm2fr.Histogram1D import Histogram1D
from gm2fr.Histogram2D import Histogram2D
from gm2fr.analysis.Optimizer import Optimizer
from gm2fr.analysis.Results import Results
import gm2fr.analysis.WiggleFit as wg
from gm2fr.analysis.BackgroundFit import BackgroundFit

import ROOT as root
import root_numpy as rnp

# Filesystem management.
import os
import shutil
import gm2fr.analysis
import time
import itertools
import re

# ==============================================================================

# Organizes the high-level logic of the fast rotation analysis.
class Analyzer:

  # ============================================================================

  # Constructor.
  def __init__(
    self,
    # Filename(s) containing data to analyze, as a string or list of strings.
    files,
    # Label(s) for signal data inside each file. If a list, must match len(files).
    signal = "signal",
    # Label(s) for pileup data inside file. If a list, must match len(signal).
    pileup = None,
    # Output directory name(s), within gm2fr/analysis/results. Must have one output per input.
    tags = None,
    # Name for a top-level directory containing each individual result folder.
    group = None,
    # Time units as a multiple of seconds, e.g. 1e-9 = nanoseconds, 1e-6 = microseconds.
    units = "us"
  ):

    # Initialize a list of input information, one for each output.
    self.input = []

    # Helper function to force an argument into an iterable list.
    def forceList(obj):
      return [obj] if type(obj) is not list else obj

    # Force the filenames, signal labels, etc. into lists.
    files = forceList(files)
    signal = forceList(signal)
    pileup = forceList(pileup)
    tags = forceList(tags)

    # There can be one file and multiple signal names, or multiple files and one
    # signal name, or one-to-one lists of file and signal names. If there are
    # multiple of each and they don't match, raise an error.
    if len(files) > 1 and len(signal) > 1 and len(signal) != len(files):
      raise ValueError("The number of signal labels does not match the number of input files.")

    # There should be a unique pileup correction for each signal.
    if len(signal) > 1 and len(pileup) != len(signal):
      raise ValueError("The number of pileup labels does not match the number of signal labels.")

    # There should be a unique output tag for each input signal.
    if len(tags) != max(len(files), len(signal)):
      raise ValueError("The number of output tags does not match the number of input signals.")

    # Look for integers in each tag to serve as numerical indices for output.
    # e.g. tag "Calo24" -> numerical index 24 saved to results file
    # TODO: merge with non-group indices logic, put in input tuple
    self.groupLabels = np.zeros(len(tags))
    for i, tag in enumerate(tags):
      match = re.search(r"(\d+)$", tag)
      self.groupLabels[i] = match.group(1) if match else np.nan

    # Fill the input list with tuples of (filename, signal, pileup, tag).
    for i in range(len(tags)):
      self.input.append((
        files[i] if len(files) > 1 else files[0],
        signal[i] if len(signal) > 1 else signal[0],
        pileup[i] if len(pileup) > 1 else pileup[0],
        tags[i]
      ))

    # Validate the output group.
    if group is None or type(group) is str:
      self.group = group
    else:
      raise ValueError(f"\nOutput group '{group}' not recognized.")

    # The current output directory path.
    self.output = None

    # The current (structured) NumPy array of results.
    self.results = Results()
    self.groupResults = Results() if self.group is not None else None

    # Get the path to the gm2fr/analysis/results directory.
    self.parent = f"{util.path}/analysis/results"

    # Check that the results directory is valid.
    if not os.path.isdir(self.parent):
      raise RuntimeError("\nCould not find results directory 'gm2fr/analysis/results'.")

    # The current FastRotation object and time units.
    self.fastRotation = None
    self.units = units

    # The current Transform object.
    self.transform = None

  # ============================================================================

  # Setup the output directory 'gm2fr/analysis/results/{group}/{tag}'.
  def setup(self, tag):

    def makeIfAbsent(path):
      if not os.path.isdir(path):
        print(f"\nCreating output directory '{path}'.")
        os.mkdir(path)

    if tag is not None:

      # If the group directory doesn't already exist, create it.
      if self.group is not None:
        makeIfAbsent(f"{self.parent}/{self.group}")

      # Set the path for the current analysis within the results directory.
      if self.group is not None:
        self.output = f"{self.parent}/{self.group}/{tag}"
      else:
        self.output = f"{self.parent}/{tag}"

      # Make the output directories, if they don't already exist.
      makeIfAbsent(self.output)

  # ============================================================================

  # TODO: fit gaussian for one period after start time, extrapolate mean back nearest to zero for t0 seed
  def analyze(
    self,
    # Wiggle fit model. Options: None / "two" / "five" / "nine".
    fit = "nine",
    # Expected quadrupole field index, to aid the nine-parameter fit.
    n = 0.108,
    # Start time (us) for the cosine transform.
    start = 4,
    # End time (us) for the cosine transform.
    end = 200,
    # t0 time (us) for the cosine transform.
    t0 = 0.070,
    # Search for an optimal t0 from the seed, or use the fixed value.
    optimize = True,
    # Background fit model. Options: "constant" / "parabola" / "sinc" / "error".
    model = "sinc",
    # Frequency interval (kHz) for the cosine transform.
    df = 2,
    # +/- range (in us) for the initial coarse t0 scan range.
    coarseRange = 0.020,
    # Step size (in us) for the initial coarse t0 scan range.
    coarseStep = 0.002,
    # +/- range (in us) for the subsequent fine t0 scan ranges.
    fineRange = 0.0005,
    # Step size (in us) for the subsequent fine t0 scan ranges.
    fineStep = 0.00005,
    # Plotting option. 0 = nothing, 1 = main results, 2 = more details (slower).
    plots = 1,
    # Optional "data.npz" file from toy Monte Carlo simulation.
    truth = None,
    rebin = 1
  ):

    begin = time.time()

    # Force scannable parameters into lists.
    if type(start) not in [list, np.ndarray]:
      start = np.array([start])
    if type(end) not in [list, np.ndarray]:
      end = np.array([end])

    f = np.arange(6631, 6780, df)

    # Loop over all specified inputs.
    for groupIndex, (file, signal, pileup, tag) in enumerate(self.input):

      print(f"\nWorking on '{tag}'.")

      # Load the truth-level data for Toy MC, if supplied.
      # TODO: incorporate this into input, not analyze
      truth_results = None
      if truth is not None:

        if truth == "same":
          truth = file

        truth_joint = Histogram2D.load(truth, "joint")
        truth_frequency = Histogram1D.load(truth, "frequencies").normalize()

        ref_predicted = truth_frequency.copy()
        # truth_results = truth_frequency.results()

      try:
        self.fastRotation = Histogram1D.load(file, signal)
      except:
        print(f"\nWarning: could not load fast rotation signal; continuing to next file.")
        continue

      try:
        if pileup is not None:
          self.fastRotation.heights -= Histogram1D.load(file, pileup).heights
      except:
        print(f"\nWarning: could not load pileup histogram; continuing without pileup correction.")

      if self.units == "ns":
        self.fastRotation.map(lambda t: t * 1E-3)

      # Perform the wiggle fit, and remove it from the input signal.
      wgFit = None
      if fit is not None:
        wgFit = wg.WiggleFit(self.fastRotation, model = fit, n = n)
        wgFit.fit()
        self.fastRotation *= 1 / wgFit.fineResult

      # Setup the output directory.
      self.setup(tag)

      self.fastRotation.save(f"{self.output}/signal.npz")
      self.fastRotation.save(f"{self.output}/signal.root", "signal")

      # Plot the fast rotation signal, and wiggle fit (if present).
      if plots >= 1:

        self.fastRotation.plot(errors = False)
        style.xlabel(r"Time ($\mu$s)")
        style.ylabel("Arbitrary Units")
        plt.xlim(0, 5)

        pdf = PdfPages(f"{self.output}/FastRotation.pdf")
        pdf.savefig()
        endTimes = [5, 10, 30, 50, 100, 150, 200, 300]
        for endTime in endTimes:
          mask = (self.fastRotation.centers >= 4) & (self.fastRotation.centers <= endTime)
          plt.xlim(4, endTime)
          plt.ylim(0, 1.05 * np.max(self.fastRotation.heights[mask]))
          pdf.savefig()
        pdf.close()
        plt.clf()

        if wgFit is not None:
          wgFit.plot(f"{self.output}/WiggleFit.pdf")
          wgFit.plotFine(self.output, endTimes)
          util.plotFFT(
            wgFit.fineSignal.centers,
            wgFit.fineSignal.heights,
            f"{self.output}/RawSignalFFT.pdf"
          )

      # Zip together each parameter in the scans.
      iterations = list(itertools.product(start, end))

      # Turn off plotting if we're doing a scan.
      if len(iterations) > 1:
        plots = 0

      for i, (iStart, iEnd) in enumerate(iterations):

        if len(iterations) > 1:
          print("\nWorking on configuration:")
          print(f"start = {iStart}, end = {iEnd}")

        frMask = self.fastRotation.copy().mask((iStart, iEnd))

        fineScan = None
        coarseScan = None

        opt_t0 = t0
        if optimize and model is not None:

          coarseScan = Optimizer(frMask, f, model, 0.002, 0.020)
          coarseScan.optimize()

          fineScan = Optimizer(
            frMask,
            f,
            model,
            fineStep,
            2*fineRange,
            seed = coarseScan.t0
          )
          fineScan.optimize()
          opt_t0 = fineScan.t0

          if plots > 0:
            fineScan.plotChi2(f"{self.output}/BackgroundChi2.pdf")

        self.transform = Histogram1D.transform(frMask, f, opt_t0)

        self.bgFit = None
        if model is not None:
          self.bgFit = BackgroundFit(self.transform, opt_t0, iStart, model).fit()
          self.transform = self.bgFit.subtract()

        if truth is not None:

          # Take truth distribution, map time values to A and B coefficients, and average over time.
          A = truth_joint.copy().map(x = lambda tau: util.A(tau*1E-3, opt_t0)).mean(axis = 0, empty = 0)
          B = truth_joint.copy().map(x = lambda tau: util.B(tau*1E-3, opt_t0)).mean(axis = 0, empty = 0)

          # Plot A(f) and B(f).
          style.yZero()
          A.plot(errors = True, label = "$A(f)$")
          B.plot(errors = True, label = "$B(f)$")
          plt.ylim(-1, 1)
          plt.xlim(util.min["f"], util.max["f"])
          style.xlabel("Frequency (kHz)")
          style.ylabel("Coefficient")
          plt.legend()
          plt.savefig(f"{self.output}/coefficients.pdf")
          plt.clf()

          A_rho = A * truth_frequency
          B_rho = B * truth_frequency

          # Plot the scaled distributions A(f)p(f) and B(f)p(f).
          style.yZero()
          A_rho.plot(errors = True, label = r"$A(f)\rho(f)$")
          B_rho.plot(errors = True, label = r"$B(f)\rho(f)$")
          plt.xlim(util.min["f"], util.max["f"])
          style.xlabel("Frequency (kHz)")
          style.ylabel("Scaled Distribution")
          plt.legend()
          plt.savefig(f"{self.output}/scaled.pdf")
          plt.clf()

          # Calculate the four main terms with appropriate scale factors.
          scale = 1 / (self.fastRotation.width * util.kHz_us)
          peak = (A * truth_frequency) * (scale * 0.5)

          def convolve(histogram, function):
            result = histogram.copy().clear()
            fDifferences = function(np.subtract.outer(histogram.centers, histogram.centers))
            result.heights = np.einsum("i, ik -> k", histogram.heights, fDifferences)
            if histogram.cov.ndim == 2:
              result.cov = np.einsum("ik, jl, ij -> kl", fDifferences, fDifferences, histogram.cov)
            else:
              result.cov = np.einsum("ik, il, i -> kl", fDifferences, fDifferences, histogram.cov)
            result.updateErrors()
            return result

          distortion = convolve(
            B_rho,
            lambda x: util.cosine(x, iStart, iEnd, opt_t0)
          ) * (-scale * truth_frequency.width)

          background = convolve(
            A_rho,
            lambda x: util.sinc(2*np.pi*x, (iStart - opt_t0) * util.kHz_us)
          ) * (-scale * truth_frequency.width)

          # wiggle = scale * util.sine(ref.frequency, iStart, iEnd, self.transform.t0)
          wiggle = truth_frequency.copy().clear().setHeights(
            scale * util.sine(truth_frequency.centers, iStart, iEnd, opt_t0)
          )

          # Plot the four main terms individually.
          style.yZero()
          peak.plot(errors = True, label = "Peak")
          distortion.plot(errors = True, label = "Distortion")
          background.plot(errors = True, label = "Background")
          (wiggle*5).plot(errors = True, label = "Wiggle (5x)")
          plt.xlim(util.min["f"], util.max["f"])
          style.xlabel("Frequency (kHz)")
          style.ylabel("Term")
          plt.legend()
          plt.savefig(f"{self.output}/terms.pdf")
          plt.clf()

          # Calculate the predicted transform, minus the background/wiggle.
          ref_predicted = peak + distortion

          # If no background subtraction, include it in the prediction.
          if self.bgFit is None:
            ref_predicted += background*(-1) + wiggle

          truth_frequency.plot(errors = False, label = "True Distribution", scale = np.max(ref_predicted.heights) / np.max(truth_frequency.heights), ls = ":")
          ref_predicted.plot(label = "Predicted Transform")
          self.transform.plot(label = "Actual Transform")
          plt.savefig(f"{self.output}/predicted_result.pdf")
          plt.clf()

          truth_frequency.plot(errors = False, label = "True Distribution", scale = np.max(self.transform.heights) / np.max(truth_frequency.heights), ls = ":")
          self.transform.plot(label = "Cosine Transform")
          plt.savefig(f"{self.output}/truth_raw.pdf")
          plt.clf()

          # Subtract the distortion term, interpolated to match the transform.
          transform_corr = self.transform.copy() + distortion.interpolate(self.transform.centers) * (-1)

          # Subtract the background term (if not already fit), interpolated to match the transform.
          if self.bgFit is None:
            transform_corr += background.interpolate(self.transform.centers) * (-1)

          # Divide by A(f), interpolated to match the transform, replacing zeros with 1.
          transform_corr.divide(A.interpolate(self.transform.centers), zeros = 1)

          truth_frequency.plot(label = "True Distribution", ls = ":", scale = scale * 0.5)
          transform_corr.plot(label = "Corrected Transform", ls = "--")
          plt.savefig(f"{self.output}/truth_corrected.pdf")
          plt.clf()

        # Determine which units to plot a distribution for.
        axesToPlot = []
        if plots > 0:
          axesToPlot = ["f", "x"]
          if plots > 1:
            axesToPlot = util.frequencyTo.keys()

        pdf = PdfPages(f"{self.output}/AllDistributions.pdf")

        # Make the final distribution plots for each unit.
        for axis in axesToPlot:
          # Plot the truth-level distribution for comparison, if present.
          if truth is not None:
            ref_predicted.plot(label = "Predicted")
          self.transform.copy().map(util.frequencyTo[axis]).plot(
            label = None if truth is None else "Result"
          )
          pdf.savefig()
          plt.clf()

        pdf.close()

        if plots > 0:

          if truth is not None:
            truth_frequency.plot(errors = False, label = "Truth")
          Histogram1D.transform(self.fastRotation, f, opt_t0, type = "cosine").plot()
          Histogram1D.transform(self.fastRotation, f, opt_t0, type = "sine").plot()
          Histogram1D.transform(self.fastRotation, f, opt_t0, type = "magnitude").plot()
          plt.savefig(f"{self.output}/magnitude.pdf")
          plt.clf()
          # self.transform.plotMagnitude(self.output, scale = np.max(ref.signal) if truth is not None else 1)
          util.plotFFT(
            frMask.centers,
            frMask.heights,
            f"{self.output}/FastRotationFFT.pdf"
          )

          # Plot the final background fit.
          if self.bgFit is not None:

            self.bgFit.plot(f"{self.output}/BackgroundFit.pdf")

            # Plot the correlation matrix among frequency bins in the background fit.
            # self.bgFit.plotCorrelation(
            #   f"{self.output}/BackgroundCorr.pdf"
            # )

            # self.transform.bgFit.save(f"{self.output}/background.npz")

          self.transform.save(f"{self.output}/transform.npz")

        # Compile the results list of (name, value) pairs from each object.
        results = Results({"start": iStart, "end": iEnd, "df": df, "t0": opt_t0, "err_t0": fineScan.err_t0 if fineScan is not None else 0})
        for unit in util.frequencyTo.keys():
          hist = self.transform.copy().map(util.frequencyTo[unit])
          mean, mean_err = hist.mean(error = True)
          std, std_err = hist.std(error = True)
          results.merge(Results({
            unit: mean,
            f"err_{unit}": mean_err,
            f"sig_{unit}": std,
            f"err_sig_{unit}": std_err
          }))
        if self.bgFit is not None:
          results.merge(self.bgFit.results())
        if wgFit is not None:
          results.merge(wgFit.results())

        # Add t_0 errors in quadrature with statistical errors.
        # TODO: info box on plots needs to incorporate this
        if fineScan is not None:
          errors = fineScan.errors(self.transform)
          for (axis, data) in errors.table.iteritems():
            results.table[axis] = np.sqrt(results.table[axis]**2 + data**2)

        # Include the differences from the reference distribution, if provided.
        # if truth_results is not None:
        #   diff_results = truth_results.copy()
        #   columnsToDrop = [x for x in diff_results.table.columns if "err_" in x]
        #   diff_results.table.drop(columns = columnsToDrop, inplace = True)
        #   # Set the ref_results column data to the difference from the results.
        #   for (name, data) in diff_results.table.iteritems():
        #     diff_results.table[name] = results.table[name] - diff_results.table[name]
        #   # Change the column names with "diff" prefix.
        #   diff_results.table.columns = [f"diff_{name}" for name in diff_results.table.columns]
        #   results.merge(diff_results)

        self.results.append(results)
        if self.groupResults is not None:
          self.groupResults.append(
            results,
            self.groupLabels[groupIndex] if not np.isnan(self.groupLabels[groupIndex]) else groupIndex
          )

      # Save the results to disk.
      if self.output is not None:
        self.results.save(self.output)

    if self.group is not None:
      self.groupResults.save(f"{self.parent}/{self.group}")

    print(f"\nCompleted in {time.time() - begin:.2f} seconds.")
