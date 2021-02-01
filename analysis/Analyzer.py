import numpy as np

import gm2fr.utilities as util
import gm2fr.analysis.FastRotation as fr
import gm2fr.analysis.Transform as tr
from gm2fr.simulation.histogram import Histogram

# Filesystem management.
import os
import shutil
import gm2fr.analysis

# ==============================================================================

# Organizes input/output for the fast rotation analysis.
# TODO: enable lists of inputs and tags
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

      if type(input[i]) is tuple and len(input[i]) == 2 and type(input[i][0]) is str and type(input[i][1]) in [list, str]:
        filename = input[i][0]
        histograms = input[i][1] if type(input[i][1]) is list else [input[i][1]]
        for histogram in histograms:
          self.input.append((filename, histogram))

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
    else:
      raise ValueError(f"\nOutput tags do not match input format.")

    # The current output directory path.
    self.output = None

    # The current (structured) NumPy array of results.
    self.results = None

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

  # Setup the output directory 'gm2fr/analysis/results/{tag}'.
  def setup(self, tag):

    if tag is not None:

      # Set the path for the current analysis within the results directory.
      self.output = f"{self.parent}/{tag}"

      # If it already exists, clear its contents.
      if os.path.isdir(self.output):
        shutil.rmtree(self.output)

      # Make the output directory.
      print(f"\nCreating output directory 'gm2fr/analysis/results/{tag}'.")
      os.mkdir(self.output)

      # Make the background and signal directories.
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
    # Background fit bounds (kHz), fixed if supplied. Format: (lower, upper).
    bounds = None,
    # Frequency interval (kHz) for the cosine transform.
    df = 2,
    # Cutoff (in units of fit pulls) for inclusion in the background definition.
    cutoff = 3
  ):

    # Loop over all specified inputs.
    for input, tag in zip(self.input, self.tags):

      # Setup the output directory.
      self.setup(tag)

      # Produce the fast rotation signal.
      if type(input) is tuple:
        self.fastRotation = fr.FastRotation.produce(input[0], input[1], fit, n)
      else:
        h = Histogram.load(input)
        self.fastRotation = fr.FastRotation(h.xCenters, h.heights, h.errors, self.units)

      # Plot the fast rotation signal.
      self.fastRotation.plot(self.output)

      # Evaluate the frequency distribution.
      self.transform = tr.Transform(self.fastRotation, start, end, df, cutoff, model)
      self.transform.process(t0, optimize, self.output, bounds)

      # Define the quantities to go in the results array.
      columns = [
        "t0",
        "start",
        "end",
        "<f>",
        "sigma_f"
      ]

      # Initialize the results array with zeros.
      self.results = np.zeros(1, dtype = [(col, np.float32) for col in columns])

      # Fill the results array.
      self.results["t0"][0] = self.transform.t0
      self.results["start"][0] = self.transform.start
      self.results["end"][0] = self.transform.end
      self.results["<f>"][0] = self.transform.getMean("frequency")
      self.results["sigma_f"][0] = self.transform.getWidth("frequency")

      # Save the results array.
      if self.output is not None:
        np.save(f"{self.output}/results.npy", self.results)
        np.savetxt(f"{self.output}/results.txt", self.results, fmt = "%.4f", header = ",".join(columns), delimiter = ",")
