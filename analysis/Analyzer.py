import numpy as np
import numpy.lib.recfunctions as rec
import matplotlib.pyplot as plt

import gm2fr.utilities as util
import gm2fr.style as style
import gm2fr.analysis.FastRotation as fr
import gm2fr.analysis.Transform as tr
from gm2fr.simulation.histogram import Histogram
from gm2fr.analysis.Optimizer import Optimizer
from gm2fr.analysis.Results import Results

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

# Organizes input/output for the fast rotation analysis.
class Analyzer:

  # ============================================================================

  # Constructor.
  def __init__(
    self,
    # Format: (filename, histogram) for ROOT file; string for NumPy filename.
    input,
    # Desired output directory name, within gm2fr/analysis/results.
    tags = None,
    # Name for a top-level directory containing each individual result folder.
    group = None,
    # Time units. Options: "ns" / "us".
    units = "us"
  ):

    # Initialize the validated input list.
    self.input = []

    # Force the input into a list.
    if type(input) is not list:
      input = [input]

    # Validate the input, copying one-by-one to the validated list.
    for i in range(len(input)):

      if type(input[i]) is tuple and len(input[i]) in [2, 3]:

        if len(input[i]) == 2:
          filename, histogram = input[i]
          pileup = None
        else:
          filename, histogram, pileup = input[i]

        self.input.append((filename, histogram, pileup))

      elif type(input[i]) is str:
        self.input.append(input[i])

      else:
        raise ValueError(f"\nInput format '{input[i]}' not recognized.")

    # Force the output tags into a list.
    if type(tags) is not list:
      tags = [tags]

    # Validate the output tags.
    if len(tags) == len(self.input):

      self.tags = tags

      # Look for integers in each tag to serve as numerical indices for output.
      # e.g. tag "Calo24" -> index 24
      self.groupLabels = np.zeros(len(self.tags))
      for i, tag in enumerate(self.tags):
        match = re.search(r"(\d+)$", tag)
        self.groupLabels[i] = match.group(1) if match else np.nan

    else:
      raise ValueError(f"\nOutput tags do not match input format.")

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
    self.parent = os.path.dirname(gm2fr.analysis.__file__) + "/results"

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

    # If the group directory doesn't already exist, create it.
    if self.group is not None and not os.path.isdir(f"{self.parent}/{self.group}"):
      os.mkdir(f"{self.parent}/{self.group}")

    if tag is not None:

      # Set the path for the current analysis within the results directory.
      if self.group is not None:
        self.output = f"{self.parent}/{self.group}/{tag}"
      else:
        self.output = f"{self.parent}/{tag}"

      # If the output directory already exists, clear its contents.
      # if os.path.isdir(self.output):
      #
      #   # Make sure the contents are consistent with an analysis directory.
      #   subdirectories = [f.name for f in os.scandir(self.output) if f.is_dir()]
      #   for subdir in subdirectories:
      #     if subdir not in ["background", "signal"]:
      #       raise RuntimeError((
      #         "\nExisting output directory has unexpected structure."
      #         "\nFor safety, will not delete/overwrite."
      #       ))
      #
      #   # If the contents are normal, clear everything inside.
      #   shutil.rmtree(self.output)

      def makeIfAbsent(path):
        if not os.path.isdir(path):
          print(f"\nCreating output directory '{path}'.")
          os.mkdir(path)

      # Make the output directories, if they don't already exist.
      makeIfAbsent(self.output)
      makeIfAbsent(f"{self.output}/background")
      makeIfAbsent(f"{self.output}/signal")

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
    # Background fit model. Options: "parabola" / "sinc" / "error".
    model = "parabola",
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
    truth = None
  ):

    begin = time.time()

    # Force scannable parameters into lists.
    if type(start) not in [list, np.ndarray]:
      start = np.array([start])
    if type(end) not in [list, np.ndarray]:
      end = np.array([end])

    # Loop over all specified inputs.
    groupIndex = 0
    for input, tag in zip(self.input, self.tags):

      # Load the truth-level data for Toy MC, if supplied.
      truth_results = None
      if truth is not None:
        if type(truth) is str:
          if truth == "same":
            truth = input

          truth_joint = Histogram.load(truth, "joint")
          truth_frequency = Histogram.load(truth, "frequencies")

          ref = tr.Transform(None, n = n)
          ref.signal = truth_frequency.heights
          ref.frequency = truth_frequency.xCenters
          ref.df = ref.frequency[1] - ref.frequency[0]
          ref.cov = np.diag(truth_frequency.errors**2)
          ref.setup()

          truth_results = ref.results(parameters = False)

        else:
          raise ValueError("Could not load reference distribution.")

      # Produce the fast rotation signal.
      self.fastRotation = fr.FastRotation.produce(input, fit, n, self.units)

      # If the fast rotation signal couldn't be produced, skip this input.
      if self.fastRotation is None:
        continue

      # Setup the output directory.
      self.setup(tag)

      self.fastRotation.save(f"{self.output}/signal/signal.npz")

      # Plot the fast rotation signal, and wiggle fit (if present).
      if plots >= 1:

        endTimes = [5, 10, 30, 50, 100, 150, 200, 300]
        self.fastRotation.plot(self.output, endTimes)

        if self.fastRotation.wgFit is not None:
          self.fastRotation.wgFit.plot(f"{self.output}/signal/WiggleFit.pdf")
          self.fastRotation.wgFit.plotFine(self.output, endTimes)
          util.plotFFT(
            self.fastRotation.wgFit.fineTime,
            self.fastRotation.wgFit.fineSignal,
            f"{self.output}/signal/RawFFT.pdf"
          )

      # Zip together each parameter in the scans.
      iterations = list(itertools.product(start, end))

      # Turn off plotting if we're doing a scan.
      if len(iterations) > 1:
        plots = 0

      i = 0
      for iStart, iEnd in iterations:

        if len(iterations) > 1:
          print("\nWorking on configuration:")
          print(f"start = {iStart}, end = {iEnd}")

        # Evaluate the frequency distribution.
        self.transform = tr.Transform(
          self.fastRotation,
          iStart,
          iEnd,
          df,
          model,
          t0 if type(t0) != list else t0[groupIndex],
          n
        )

        fineScan = None
        coarseScan = None

        if optimize:

          coarseScan = Optimizer(
            self.transform,
            t0 if type(t0) != list else t0[groupIndex],
            coarseStep,
            2*coarseRange
          )
          coarseScan.optimize()

          fineScan = Optimizer(
            self.transform,
            coarseScan.t0,
            fineStep,
            2*fineRange
          )
          fineScan.optimize()
          self.transform.t0 = fineScan.t0

          if plots > 0:
            coarseScan.plotChi2(f"{self.output}/background/coarse_scan.pdf")
            fineScan.plotChi2(f"{self.output}/background/fine_scan.pdf")
            fineScan.leftFit.plot(f"{self.output}/background/LeftFit.pdf")
            fineScan.rightFit.plot(f"{self.output}/background/RightFit.pdf")
            if plots > 1:
              coarseScan.plotFits(f"{self.output}/background/AllFits_coarse.pdf")

        self.transform.process()

        if truth is not None:

          # Calculate and plot A(f), B(f).
          A = util.A(truth_joint.yCenters, truth_joint.xCenters*1E-3, truth_joint.heights.T, self.transform.t0)
          B = util.B(truth_joint.yCenters, truth_joint.xCenters*1E-3, truth_joint.heights.T, self.transform.t0)
          plt.plot(truth_joint.yCenters, A, label = r"$A(f)$")
          plt.plot(truth_joint.yCenters, B, label = r"$B(f)$")
          style.xlabel("Frequency (kHz)")
          style.ylabel("Coefficient")
          plt.legend()
          plt.savefig(f"{self.output}/coefficients.pdf")
          plt.clf()

          # Normalize the truth-level distribution to the same area as result.
          area = np.sum(self.transform.signal * self.transform.df)
          ref_area = np.sum(ref.signal * ref.df)
          ref_scale = area / ref_area
          ref.signal *= ref_scale
          ref.cov *= ref_scale**2

        # Determine which units to plot a distribution for.
        axesToPlot = []
        if plots > 0:
          axesToPlot = ["f", "x"]
          if plots > 1:
            axesToPlot = self.transform.axes.keys()

        # Make the final distribution plots for each unit.
        for axis in axesToPlot:
          # Plot the truth-level distribution for comparison, if present.
          if truth is not None:
            ref.plot(None, axis, databox = False, magic = False, label = "Truth")
          self.transform.plot(
            f"{self.output}/{util.labels[axis]['file']}.pdf",
            axis,
            label = None if truth is None else "Result"
          )

        if plots > 0:

          if truth is not None:
            ref.plot(None, "f", databox = False, magic = False, label = "Truth")
          self.transform.plotMagnitude(self.output)
          util.plotFFT(
            self.transform.frTime,
            self.transform.frSignal,
            f"{self.output}/signal/FastRotationFFT.pdf"
          )

          # Plot the final background fit.
          self.transform.bgFit.plot(f"{self.output}/BackgroundFit.pdf")

          # Plot the correlation matrix among frequency bins in the background fit.
          self.transform.bgFit.plotCorrelation(
            f"{self.output}/background/correlation.pdf"
          )

          self.transform.save(f"{self.output}/transform.npz")
          self.transform.bgFit.save(f"{self.output}/background.npz")

        # Compile the results list of (name, value) pairs from each object.
        results = self.transform.results()
        if self.transform.bgFit is not None:
          results.merge(self.transform.bgFit.results())
        if self.fastRotation.wgFit is not None:
          results.merge(self.fastRotation.wgFit.results())

        # Add t_0 errors in quadrature with statistical errors.
        # TODO: info box on plots needs to incorporate this
        if fineScan is not None:
          errors = fineScan.errors(self.transform)
          for (axis, data) in errors.table.iteritems():
            results.table[axis] = np.sqrt(results.table[axis]**2 + data**2)

        # Include the differences from the reference distribution, if provided.
        if truth_results is not None:
          diff_results = truth_results.copy()
          # Set the ref_results column data to the difference from the results.
          for (name, data) in diff_results.table.iteritems():
            diff_results.table[name] = results.table[name] - diff_results.table[name]
          # Change the column names with "diff" prefix.
          diff_results.table.columns = [f"diff_{name}" for name in diff_results.table.columns]
          results.merge(diff_results)

        self.results.append(results)
        if self.groupResults is not None:
          self.groupResults.append(
            results,
            self.groupLabels[groupIndex] if not np.isnan(self.groupLabels[groupIndex]) else groupIndex
          )

        i += 1

      # Save the results to disk.
      if self.output is not None:
        self.results.save(self.output)

      groupIndex += 1

    if self.group is not None:
      self.groupResults.save(f"{self.parent}/{self.group}")

    print(f"\nCompleted in {time.time() - begin:.2f} seconds.")
