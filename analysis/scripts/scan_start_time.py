import argparse
import gm2fr.constants as const
import gm2fr.io as io
from gm2fr.simulation.scripts.emulate_dataset import emulate_dataset
from gm2fr.analysis.Analyzer import Analyzer
import os
import numpy as np

# ==================================================================================================

def scan_start_time(dataset, start_times, sim = False):

  if not sim:

    analyzer = Analyzer(
      filename = f"{io.data_path}/FastRotation_{dataset}.root",
      signal_label = f"FastRotation/AllCalos/hHitTime",
      pileup_label = f"FastRotation/AllCalos/hPileupTime",
      output_label = f"{dataset}/StartTimeScan",
      fr_method = "nine",
      n = 0.108 if dataset not in ("1B", "1C") else 0.120,
      time_units = 1E-9
    )

    nominal_results = np.load(f"{io.results_path}/{dataset}/Nominal/results.npy")

  else:

    analyzer = Analyzer(
      filename = f"{io.sim_path}/{dataset}_sim/simulation.root",
      signal_label = "signal",
      output_label = f"{dataset}/StartTimeScan",
      output_prefix = "sim_",
      n = 0.108 if dataset not in ("1B", "1C") else 0.120,
      time_units = 1E-6
    )

    nominal_results = np.load(f"{io.results_path}/{dataset}/Simulation/results.npy")

  analyzer.scan_parameters(
    start = start_times,
    bg_model = ["error"],
    t0 = [nominal_results["t0"]]
  )

# ==================================================================================================

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset")
  parser.add_argument("--sim", action = "store_true")
  parser.add_argument("--start", type = float, default = 4)
  parser.add_argument("--end", type = float, default = 30)
  parser.add_argument("--step", type = float, default = const.info["T"].magic * 1E-3)
  args = parser.parse_args()

  scan_start_time(args.dataset, np.arange(args.start, args.end, args.step), args.sim)
