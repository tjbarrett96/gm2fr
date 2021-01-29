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
class Analyzer:

  # ============================================================================
  
  # Constructor.
  def __init__(
    self,
    # Format: (filename, histogram) for ROOT file; string for NumPy filename.
    input,
    # Desired output directory name, within gm2fr/analysis/results.
    tag = None,
    # Time units. Options: "ns" / "us".
    units = "us"
  ):
  
    # TODO: extend these to process lists of input automatically.
  
    # The input file(s).
    self.input = input
    
    # The output directory name(s), within gm2fr/analysis/results.
    self.tag = tag
    
    # The full output directory path(s).
    self.output = None
    
    # The (structured) NumPy array of results.
    self.results = None
    
    # Get the path to the gm2fr/analysis/results directory.
    self.parent = os.path.dirname(gm2fr.analysis.__file__) + "/results"
    
    # Check that the results directory is valid.
    if not os.path.isdir(self.parent):
      raise RuntimeError("\nCould not find results directory 'gm2fr/analysis/results'.")
      
    # FastRotation object.
    self.fastRotation = None
    self.units = units
    
    # Transform object.
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
    
    # Setup the output directory.
    self.setup(self.tag)
    
    # Produce the fast rotation signal.
    if type(self.input) is tuple and len(self.input) == 2:
      self.fastRotation = fr.FastRotation.produce(self.input[0], self.input[1], fit, n)
    elif type(self.input) is str:
      h = Histogram.load(self.input)
      self.fastRotation = fr.FastRotation(h.xCenters, h.heights, h.errors, self.units)
    else:
      raise RuntimeError(f"\nInput format '{self.input}' not recognized.")
      
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
