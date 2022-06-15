from gm2fr.src.mixture import GaussianMixture
from gm2fr.src.simulator import Simulator

import sys
import gm2fr.src.io as io
import gm2fr.src.constants as const

from gm2fr.src.Histogram1D import Histogram1D

import numpy as np
import ROOT as root
import root_numpy as rnp
import matplotlib.pyplot as plt

# ==================================================================================================

def emulate_dataset(dataset, muons):

  input_dir = f"{dataset}/Nominal"

  fr_signal = Histogram1D.load(f"{io.gm2fr_path}/results/{input_dir}/signal.npz")
  frequencies = Histogram1D.load(f"{io.gm2fr_path}/results/{input_dir}/transform.npz", "transform_f")
  results = np.load(f"{io.gm2fr_path}/results/{input_dir}/results.npy")

  # No negative bin contents allowed for random sampling.
  frequencies.heights = np.where(frequencies.heights > 0, frequencies.heights, 0)

  # Select 1.5 cyclotron periods after 4 microseconds, to find the left edge of a single-turn window.
  fr_mask_left = (fr_signal.centers >= 4) & (fr_signal.centers <= 4 + 1.5 * const.info["T"].magic * 1E-3)

  # Select the time of the smallest sample, and treat that as the start of a single-turn window.
  fr_start = fr_signal.centers[fr_mask_left][np.argmin(fr_signal.heights[fr_mask_left])]

  # Refine the mask to be 0.5 -- 1.5 cyclotron periods after this new start time, to find the right edge of the single-turn window.
  fr_mask_right = (fr_signal.centers >= fr_start + 0.5 * const.info["T"].magic * 1E-3) & (fr_signal.centers <= fr_start + 1.5 * const.info["T"].magic * 1E-3)

  # Select the time of the smallest sample in the new window, and treat that as the end of the single-turn window.
  fr_end = fr_signal.centers[fr_mask_right][np.argmin(fr_signal.heights[fr_mask_right])]

  # Mask the fast rotation signal within this single-turn window.
  times = fr_signal.copy().mask((fr_start, fr_end))

  # Translate down so the edges of the distribution are at zero, and nullify any negative content.
  min_height = np.min(times.heights)
  times.heights -= min_height
  times.heights[times.heights < 0] = 0

  # Center the times on zero.
  mean_time = times.mean()
  times.centers -= mean_time

  # Scale the width down by the approximate amount of spread since t = 0, assuming no p-t correlation.
  time_width = times.std()
  num_turns = mean_time // (results["T"] * 1E-3)
  shrunken_width = np.sqrt(time_width**2 - (results["sig_T"] * 1E-3 * num_turns)**2)
  times.map(lambda t: t * (shrunken_width / time_width))

  simulation = Simulator(
    f"{dataset}_sim",
    overwrite = True,
    kinematics_dist = frequencies,
    kinematics_type = "f",
    time_dist = times,
    time_units = 1E-6
  )

  simulation.simulate(muons = muons, end = 250, decay = "uniform", detector = (results["t0"] * 1E3) / results["T"])
  simulation.save()
  simulation.plot()

# ==================================================================================================

if __name__ == "__main__":

  # Check for the dataset argument.
  if len(sys.argv) not in (2, 3):
    print("Arguments not recognized. Usage:")
    print("python3 emulate_dataset.py <dataset> [muons]")
    exit()

  # Parse the dataset argument.
  dataset = sys.argv[1]
  muons = int(float(sys.argv[2])) if len(sys.argv) == 3 else 1E10
  emulate_dataset(dataset, muons)
