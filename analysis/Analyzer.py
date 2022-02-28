import numpy as np
import numpy.lib.recfunctions as rec
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.interpolate as interp
from matplotlib.backends.backend_pdf import PdfPages

# import gm2fr.utilities as util
import gm2fr.io as io
import gm2fr.constants as const
import gm2fr.calculations as calc
import gm2fr.style as style
from gm2fr.analysis.Transform import Transform
from gm2fr.Histogram1D import Histogram1D
from gm2fr.Histogram2D import Histogram2D
from gm2fr.analysis.Optimizer import Optimizer
from gm2fr.analysis.Results import Results
import gm2fr.analysis.WiggleFit as wg
from gm2fr.analysis.BackgroundFit import BackgroundFit
from gm2fr.analysis.BackgroundModels import Template
from gm2fr.analysis.Iterator import Iterator

import ROOT as root
import root_numpy as rnp

# Filesystem management.
import os
import gm2fr.analysis
import time
import itertools
import re

# ==================================================================================================

# Organizes the high-level logic of the fast rotation analysis.
class Analyzer:

  # ================================================================================================

  # Constructor.
  def __init__(
    self,
    files, # Filename(s) containing data to analyze, as a string or list of strings.
    signal = "signal", # Label(s) for signal data inside each file. If a list, must match len(files).
    pileup = None, # Label(s) for pileup data inside file. If a list, must match len(signal).
    tags = None, # Output directory name(s), within gm2fr/analysis/results. Must have one output per input.
    truth = None, # Truth files.
    group = None, # Name for a top-level directory containing each individual result folder.
    units = "us" # Time units as a multiple of seconds, e.g. 1e-9 = nanoseconds, 1e-6 = microseconds.
  ):

    # Force the filenames, signal labels, etc. into lists.
    files = io.forceList(files)
    signal = io.forceList(signal)
    pileup = io.forceList(pileup)
    tags = io.forceList(tags)
    truth = io.forceList(truth)

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

    # There should be one common truth file, or one for each signal.
    if len(truth) > 1 and len(truth) != max(len(files), len(signal)):
      raise ValueError("The number of truth files does not match the number of input signals.")

    # Look for integers in each tag to serve as numerical indices for output.
    # e.g. tag "Calo24" -> numerical index 24 saved to results file
    # TODO: merge with non-group indices logic, put in input tuple
    self.groupLabels = [io.findIndex(tag) for tag in tags]

    # Fill an input list with tuples of (filename, signal, pileup, truth, tag).
    self.input = []
    for i in range(len(tags)):
      self.input.append((
        files[i] if len(files) > 1 else files[0],
        signal[i] if len(signal) > 1 else signal[0],
        pileup[i] if len(pileup) > 1 else pileup[0],
        truth[i] if len(truth) > 1 else truth[0],
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
    self.results = None
    self.groupResults = Results() if self.group is not None else None

    # Get the path to the gm2fr/analysis/results directory.
    self.parent = f"{io.path}/analysis/results"

    # Check that the results directory is valid.
    if not os.path.isdir(self.parent):
      raise RuntimeError("\nCould not find results directory 'gm2fr/analysis/results'.")

    # The current FastRotation object and time units.
    self.fastRotation = None
    self.units = units

    # The current Transform object.
    self.transform = None

  # ================================================================================================

  # Setup the output directory 'gm2fr/analysis/results/{group}/{tag}'.
  def setup(self, tag):
    if tag is not None:
      if self.group is not None:
        io.makeIfAbsent(f"{self.parent}/{self.group}")
        self.output = f"{self.parent}/{self.group}/{tag}"
      else:
        self.output = f"{self.parent}/{tag}"
      io.makeIfAbsent(self.output)

  # ================================================================================================

  def analyze(
    self,
    fit = "nine", # Wiggle fit model: None / "two" / "five" / "nine" / "ratio".
    n = 0.108, # Expected quadrupole field index, for C_E and the nine-parameter fit's f_cbo seed.
    start = 4, # Start time (us) for the cosine transform.
    end = 200, # End time (in us) for the cosine transform.
    t0 = 0.070, # t0 time (in us) for the cosine transform.
    optimize = True, # Search for an optimal t0 from the seed, or use the fixed value.
    iterate = True,
    model = "sinc", # Background fit model: "constant" / "parabola" / "sinc" / "error".
    df = 2, # Frequency interval (in kHz) for the cosine transform.
    coarseWidth = 0.020, # full range (in us) for the initial coarse t0 scan range.
    coarseSteps = 5, # Number of steps for the initial coarse t0 scan range.
    fineWidth = 0.001, # full range (in us) for the subsequent fine t0 scan ranges.
    fineSteps = 10, # Number of steps for the subsequent fine t0 scan ranges.
    plots = 1, # Plotting option. 0 = nothing, 1 = main results, 2 = more details (slower).
    rebin = 1 # Integer rebinning group size for the fast rotation signal.
  ):

    begin = time.time()

    # Force scannable parameters into lists.
    if type(start) not in [list, np.ndarray]:
      start = np.array([start])
    if type(end) not in [list, np.ndarray]:
      end = np.array([end])

    # Loop over all specified inputs.
    for groupIndex, (file, signal, pileup, truth, tag) in enumerate(self.input):

      print(f"\nWorking on '{tag}'.")

      self.results = Results()
      rawSignal = None
      wgFit = None

      # Try to load the fast rotation signal.
      try:

        rawSignal = Histogram1D.load(file, signal)
        if pileup is not None:
          try:
            rawSignal.heights -= Histogram1D.load(file, pileup).heights
          except:
            print(f"\nWarning: could not load pileup histogram; continuing without pileup correction.")
        if self.units == "ns":
          rawSignal.map(lambda t: t * 1E-3)

        # Create the fast rotation signal using the ratio method.
        if fit == "ratio":

          numerator = Histogram1D.load(file, f"{signal}_Num")
          denominator = Histogram1D.load(file, f"{signal}_Den")
          if pileup is not None:
            try:
              numerator.heights -= Histogram1D.load(file, f"{pileup}_Num").heights
              denominator.heights -= Histogram1D.load(file, f"{pileup}_Den").heights
            except:
              print(f"\nWarning: could not load pileup histogram; continuing without pileup correction.")
          self.fastRotation = numerator.divide(denominator, zero = 1)
          if self.units == "ns":
            self.fastRotation.map(lambda t: t * 1E-3)

        # Load the wiggle data, to be fit below.
        elif fit is not None:

          wgFit = wg.WiggleFit(rawSignal, model = fit, n = n)
          wgFit.fit()
          self.fastRotation = rawSignal.divide(wgFit.fineResult)

        else:
          self.fastRotation = rawSignal

      except FileNotFoundError:
        print(f"\nWarning: could not load fast rotation signal; continuing to next file.")
        continue

      # Setup the output directory.
      self.setup(tag)

      # Zip together each parameter in the scans.
      iterations = list(itertools.product(start, end))

      # Turn off plotting if we're doing a scan.
      if len(iterations) > 1:
        plots = 0

      for i, (iStart, iEnd) in enumerate(iterations):

        if len(iterations) > 1:
          print("\nWorking on configuration:")
          print(f"start = {iStart:.2f}, end = {iEnd:.2f}")

        frMask = self.fastRotation.copy().mask((iStart, iEnd))

        transform = Transform(self.fastRotation, iStart, iEnd, df)

        fineScan = None
        coarseScan = None

        opt_t0 = t0
        if optimize and model is not None:

          coarseScan = Optimizer(transform, model, coarseWidth, coarseSteps)
          coarseScan.optimize()

          fineScan = Optimizer(transform, model, fineWidth, fineSteps, seed = coarseScan.t0)
          fineScan.optimize()
          opt_t0 = fineScan.t0

          if plots > 0:
            fineScan.plotChi2(f"{self.output}/BackgroundChi2.pdf")

        transform.setT0(opt_t0, fineScan.err_t0 if fineScan is not None else 0)
        corr_transform = transform.optCosine.copy()

        self.bgFit = None
        if model is not None:
          self.bgFit = BackgroundFit(transform.optCosine, opt_t0, iStart, model).fit()
          corr_transform = transform.optCosine.subtract(self.bgFit.result)

          iterator = None
          if iterate:
            iterator = Iterator(transform, self.bgFit)
            corr_transform = iterator.iterate(optimize)

          if plots > 0 and iterator is not None:
            iterator.plot(f"{self.output}/Iterations.pdf")
            pass

        truth_results = None
        if truth is not None:

          if truth == "same":
            truth = file
          truth_joint = Histogram2D.load(truth, "joint")
          truth_frequency = Histogram1D.load(truth, "frequencies").normalize()
          ref_predicted = truth_frequency.copy()
          # truth_results = truth_frequency.results()

          # Take truth distribution, map time values to A and B coefficients, and average over time.
          A = truth_joint.copy().map(x = lambda tau: calc.A(tau*1E-3, opt_t0)).mean(axis = 0, empty = 0)
          B = truth_joint.copy().map(x = lambda tau: calc.B(tau*1E-3, opt_t0)).mean(axis = 0, empty = 0)

          # Plot A(f) and B(f).
          style.yZero()
          A.plot(errors = True, label = "$A(f)$")
          B.plot(errors = True, label = "$B(f)$")
          plt.ylim(-1, 1)
          plt.xlim(const.info["f"].min, const.info["f"].max)
          style.labelAndSave("Frequency (kHz)", "Coefficient", f"{self.output}/coefficients.pdf")

          A_rho = A.multiply(truth_frequency)
          B_rho = B.multiply(truth_frequency)

          # Plot the scaled distributions A(f)p(f) and B(f)p(f).
          style.yZero()
          A_rho.plot(errors = True, label = r"$A(f)\rho(f)$")
          B_rho.plot(errors = True, label = r"$B(f)\rho(f)$")
          plt.xlim(const.info["f"].min, const.info["f"].max)
          style.labelAndSave("Frequency (kHz)", "Scaled Distribution", f"{self.output}/scaled.pdf")

          # Calculate the four main terms with appropriate scale factors.
          # Note: the factor of 1/2 in the peak scale comes from transforming rho(w) -> rho(f).
          scale = 1 / (self.fastRotation.width * const.kHz_us)
          peak = A_rho.multiply(scale / 2)
          distortion = B_rho.convolve(lambda x: calc.c(x, iStart, iEnd, opt_t0)).multiply(-scale * truth_frequency.width)
          background = A_rho.convolve(lambda x: calc.sinc(2*np.pi*x, (iStart - opt_t0) * const.kHz_us)).multiply(-scale * truth_frequency.width)
          wiggle = truth_frequency.copy().clear().setHeights(scale * calc.s(truth_frequency.centers, iStart, iEnd, opt_t0))

          # Plot the four main terms individually.
          style.yZero()
          peak.plot(errors = True, label = "Peak")
          distortion.plot(errors = True, label = "Distortion")
          background.plot(errors = True, label = "Background")
          wiggle.multiply(5).plot(errors = True, label = "Wiggle (5x)")
          plt.xlim(const.info["f"].min, const.info["f"].max)
          style.labelAndSave("Frequency (kHz)", "Term", f"{self.output}/terms.pdf")

          # Calculate the predicted transform, minus the background/wiggle.
          ref_predicted = peak.add(distortion)

          # If no background subtraction, include it in the prediction.
          if self.bgFit is None:
            ref_predicted += background.multiply(-1).add(wiggle)

          truth_frequency.plot(errors = False, label = "True Distribution", scale = np.max(ref_predicted.heights) / np.max(truth_frequency.heights), ls = ":")
          ref_predicted.plot(label = "Predicted Transform")
          corr_transform.plot(label = "Actual Transform")
          style.labelAndSave("Frequency (kHz)", "Arbitrary Units", f"{self.output}/predicted_result.pdf")

          truth_frequency.plot(errors = False, label = "True Distribution", scale = np.max(corr_transform.heights) / np.max(truth_frequency.heights), ls = ":")
          corr_transform.plot(label = "Cosine Transform")
          style.labelAndSave("Frequency (kHz)", "Arbitrary Units", f"{self.output}/truth_raw.pdf")

          # Subtract the distortion term, interpolated to match the transform.
          transform_corr = corr_transform.subtract(distortion.interpolate(corr_transform.centers))

          # Subtract the background term (if not already fit), interpolated to match the transform.
          if self.bgFit is None:
            transform_corr = transform_corr.subtract(background.interpolate(corr_transform.centers))

          # Divide by A(f), interpolated to match the transform, replacing zeros with 1.
          transform_corr = transform_corr.divide(A.interpolate(corr_transform.centers, spline = False), zero = 1)

          truth_frequency.plot(label = "True Distribution", ls = ":", scale = scale * 0.5)
          transform_corr.plot(label = "Corrected Transform", ls = "--")
          style.labelAndSave("Frequency (kHz)", "Arbitrary Units", f"{self.output}/truth_corrected.pdf")

      # Save the results to disk.
      if self.output is not None:

        # Plot the fast rotation signal, and wiggle fit (if present).
        if plots >= 1:

          frPlotBegin = time.time()

          self.fastRotation.save(f"{self.output}/signal.npz")
          self.fastRotation.save(f"{self.output}/signal.root", "signal")

          self.fastRotation.plot(errors = False)
          style.xlabel(r"Time ($\mu$s)")
          style.ylabel("Arbitrary Units")

          mask = (self.fastRotation.centers >= 0) & (self.fastRotation.centers <= 5)
          plt.ylim(0, 1.05 * np.max(self.fastRotation.heights[mask]))
          plt.xlim(0, 5)

          pdf = PdfPages(f"{self.output}/FastRotation.pdf")
          pdf.savefig()
          # endTimes = [5, 10, 30, 50, 100, 150, 200, 300]
          endTimes = [5, 100, 300]
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
            calc.plotFFT(rawSignal.centers, rawSignal.heights, f"{self.output}/RawSignalFFT.pdf")

          print(f"\nFinished plotting fast rotation signal in {time.time() - frPlotBegin:.2f} seconds.")

        plotBegin = time.time()

        # Determine which units to plot a distribution for.
        axesToPlot = []
        if plots > 0:
          axesToPlot = ["f", "x", "dp_p0"]
          if plots > 1:
            axesToPlot = list(const.info.keys())
            axesToPlot.remove("c_e")
            axesToPlot.remove("beta")

        pdf = PdfPages(f"{self.output}/AllDistributions.pdf")
        rootFile = root.TFile(f"{self.output}/transform.root", "RECREATE")

        # Compile the results list of (name, value) pairs from each object.
        results = Results({"start": iStart, "end": iEnd, "df": df, "t0": opt_t0, "err_t0": fineScan.err_t0 if fineScan is not None else 0})
        for unit in const.info.keys():

          hist = corr_transform.copy().map(const.info[unit].fromF)
          mean, mean_err = hist.mean(error = True)
          std, std_err = hist.std(error = True)
          results.merge(Results(
            {unit: mean, f"err_{unit}": mean_err, f"sig_{unit}": std, f"err_sig_{unit}": std_err}
          ))

          if unit in axesToPlot:
            # Plot the truth-level distribution for comparison, if present.
            if truth is not None:
              ref_predicted.plot(label = "Predicted")
            hist.toRoot(f"transform_{unit}", f";{const.info[unit].formatLabel()};").Write()
            hist.plot(label = None if truth is None else "Result")
            plt.axvline(const.info[unit].magic, ls = ":", c = "k", label = "Magic")
            style.yZero()
            style.ylabel("Arbitrary Units")
            style.xlabel(const.info[unit].formatLabel())
            style.databox(
              style.Entry(mean, rf"\langle {const.info[unit].symbol} \rangle", mean_err, const.info[unit].units),
              style.Entry(std, rf"\sigma_{{{const.info[unit].symbol}}}", std_err, const.info[unit].units)
            )
            plt.xlim(const.info[unit].min, const.info[unit].max)
            plt.legend()
            pdf.savefig()
            plt.clf()

        pdf.close()
        rootFile.Close()

        if self.bgFit is not None:
          results.merge(self.bgFit.results())
        if wgFit is not None:
          results.merge(wgFit.results())

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

        self.results.append(results, self.groupLabels[groupIndex] if self.groupLabels[groupIndex] is not None else groupIndex)
        if self.groupResults is not None:
          self.groupResults.append(
            results,
            self.groupLabels[groupIndex] if self.groupLabels[groupIndex] is not None else groupIndex
          )

        self.results.save(self.output)

        if plots > 0:

          if truth is not None:
            truth_frequency.plot(errors = False, label = "Truth")
          transform.optCosine.plot(label = "Cosine Transform")
          transform.optSine.plot(label = "Sine Transform")
          transform.magnitude.plot(label = "Fourier Magnitude")
          style.yZero()
          plt.axvline(const.info["f"].magic, ls = ":", c = "k", label = "Magic")
          style.labelAndSave("Frequency (kHz)", "Arbitrary Units", f"{self.output}/magnitude.pdf")

          calc.plotFFT(frMask.centers, frMask.heights, f"{self.output}/FastRotationFFT.pdf")

          # Plot the final background fit.
          if self.bgFit is not None:
            self.bgFit.plot(f"{self.output}/BackgroundFit.pdf")

          # if self.newBGFit is not None:
          #   self.newBGFit.plot(f"{self.output}/TemplateFit.pdf")

            # corr_transform.bgFit.save(f"{self.output}/background.npz")

          corr_transform.save(f"{self.output}/transform.npz")
          # corr_transform.save(f"{self.output}/transform.root", "transform")

          print(f"\nFinished plotting and saving results in {time.time() - plotBegin:.2f} seconds.")

    if self.group is not None:
      self.groupResults.save(f"{self.parent}/{self.group}")

    print(f"\nCompleted in {time.time() - begin:.2f} seconds.")
