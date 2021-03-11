import numpy as np
import numpy.lib.recfunctions as rec

import gm2fr.utilities as util
import gm2fr.analysis.FastRotation as fr
import gm2fr.analysis.Transform as tr
from gm2fr.simulation.histogram import Histogram

# Filesystem management.
import os
import shutil
import gm2fr.analysis
import time
import itertools
import re

# ==============================================================================

# Organizes input/output for the fast rotation analysis.
# TODO: let this class handle saving plots (pass them back here to save), so it can do PDFPages for scan results
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
    self.results = None
    self.groupResults = None

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
      if os.path.isdir(self.output):

        # Make sure the contents are consistent with an analysis directory.
        subdirectories = [f.name for f in os.scandir(self.output) if f.is_dir()]
        for subdir in subdirectories:
          if subdir not in ["background", "signal"]:
            raise RuntimeError((
              "\nExisting output directory has unexpected structure."
              "\nFor safety, will not delete/overwrite."
            ))

        # If the contents are normal, clear everything inside.
        shutil.rmtree(self.output)

      # Make the output directory.
      print(f"\nCreating output directory '{tag}'.")
      os.mkdir(self.output)

      # Make the background and signal subdirectories.
      os.mkdir(f"{self.output}/background")
      os.mkdir(f"{self.output}/signal")

  # ============================================================================

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
    coarseStep = 0.0005,
    # +/- range (in us) for the subsequent fine t0 scan ranges.
    fineRange = 0.0005,
    # Step size (in us) for the subsequent fine t0 scan ranges.
    fineStep = 0.000025,
    # Plotting option. 0 = nothing, 1 = main results, 2 = more details (slower).
    plots = 1
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

      # Produce the fast rotation signal.
      if type(input) is tuple:
        self.fastRotation = fr.FastRotation.produce(input[0], input[1], input[2], fit, n, self.units)
      else:
        h = Histogram.load(input)
        self.fastRotation = fr.FastRotation(h.xCenters, h.heights, h.errors, self.units)

      # If the fast rotation signal couldn't be produced, skip this input.
      if self.fastRotation is None:
        continue

      # Setup the output directory.
      self.setup(tag)

      # Plot the fast rotation signal.
      if plots >= 1:
        self.fastRotation.plot(self.output)

      # Zip together each parameter in the scans.
      iterations = list(itertools.product(start, end))

      # Turn off plotting if we're doing a scan.
      if len(iterations) > 1:
        plots = 0

      i = 0
      for iStart, iEnd in iterations:

        # Evaluate the frequency distribution.
        self.transform = tr.Transform(
          self.fastRotation,
          iStart,
          iEnd,
          df,
          model,
          coarseRange,
          coarseStep,
          fineRange,
          fineStep,
          self.output,
          optimize,
          t0 if type(t0) != list else t0[groupIndex],
          plots,
          n
        )

        self.transform.process()

        axesToPlot = []
        if plots > 0:
          axesToPlot = ["f", "x"]
          if plots > 1:
            axesToPlot = self.transform.axes.keys()
        for axis in axesToPlot:
          self.transform.plot(
            f"{self.output}/{util.labels[axis]['file']}.pdf",
            axis
          )

        if plots > 0:

          if optimize:

            # Plot the coarse background optimization scan.
            self.transform.plotOptimization(
              outDir = f"{self.output}/background",
              mode = "coarse",
              all = True if plots > 1 else False
            )

            # Plot the fine background optimization scan.
            self.transform.plotOptimization(
              outDir = f"{self.output}/background",
              mode = "fine"
            )

            # Plot the background fit with a one-sigma t0 perturbation to the left.
            self.transform.leftFit.plot(
              f"{self.output}/background/LeftFit.pdf"
            )

            # Plot the background fit with a one-sigma t0 perturbation to the right.
            self.transform.rightFit.plot(
              f"{self.output}/background/RightFit.pdf"
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
        resultsList = self.transform.results
        if self.transform.bgFit is not None:
          resultsList += self.transform.bgFit.results
        if self.fastRotation.wgFit is not None:
          resultsList += self.fastRotation.wgFit.results

        # Initialize the results arrays, if not already done.
        if self.results is None:

          header = [name for name, value in resultsList]
          self.results = np.zeros(
            len(iterations),
            dtype = [(name, np.float32) for name in header]
          )

          if self.group is not None:
            self.groupResults = np.zeros(
              len(self.input) * len(iterations),
              dtype = [("index", np.float32)] + [(name, np.float32) for name in header]
            )

        # Fill the results array.
        for name, value in resultsList:

          self.results[name][i] = value

          if self.groupResults is not None:
            self.groupResults["index"][groupIndex * len(iterations) + i] = self.groupLabels[groupIndex] if not np.isnan(self.groupLabels[groupIndex]) else groupIndex
            self.groupResults[name][groupIndex * len(iterations) + i] = value

        i += 1

      # Save the results to disk.
      if self.output is not None:

        # Save the results array in NumPy format.
        np.save(f"{self.output}/results.npy", self.results)

        # Save the results array as a CSV.
        np.savetxt(
          f"{self.output}/results.txt",
          self.results,
          fmt = "%16.5f",
          header = "  ".join(f"{name:>16}" for name in self.results.dtype.names),
          delimiter = "  ",
          comments = ""
        )

      groupIndex += 1

    if self.group is not None:

      # Save the results array in NumPy format.
      np.save(f"{self.parent}/{self.group}/results.npy", self.groupResults)

      # Save the results array as a CSV.
      np.savetxt(
        f"{self.parent}/{self.group}/results.txt",
        self.groupResults,
        fmt = "%16.5f",
        header = "  ".join(f"{name:>16}" for name in self.groupResults.dtype.names),
        delimiter = "  ",
        comments = ""
      )

    print(f"\nCompleted in {time.time() - begin:.2f} seconds.")
