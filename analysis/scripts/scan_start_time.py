import argparse
import gm2fr.constants as const
import gm2fr.io as io
from gm2fr.simulation.scripts.emulate_dataset import emulate_dataset
from gm2fr.analysis.Analyzer import Analyzer
import os
import numpy as np

# ==================================================================================================

# TODO: make simulation flag which switches to analyzing emulated dataset. only do one at a time.
def scan_start_time(dataset, start_times, emulate):

  data_analyzer = Analyzer(
    filename = f"{io.data_path}/FastRotation_{dataset}.root",
    signal_label = f"FastRotation/AllCalos/hHitTime",
    pileup_label = f"FastRotation/AllCalos/hPileupTime",
    output_label = f"{dataset}/StartTimeScan",
    fr_method = "nine",
    n = 0.108 if dataset not in ("1B", "1C") else 0.120,
    time_units = 1E-9
  )

  nominal_results = np.load(f"{io.results_path}/{dataset}/Nominal/results.npy")

  data_analyzer.scan_parameters(start = start_times, bg_model = ["error"], t0 = [nominal_results["t0"]])

  if emulate:

    sim_path = f"{io.gm2fr_path}/simulation/data/{dataset}_sim/simulation.root"
    if not os.path.isfile(sim_path):
      emulate_dataset(dataset)

    sim_analyzer = Analyzer(
      filename = sim_path,
      signal_label = "signal",
      output_label = f"{dataset}/StartTimeScan",
      output_prefix = "sim_",
      n = 0.108 if dataset not in ("1B", "1C") else 0.120
    )

    sim_analyzer.scan_parameters(start = start_times, bg_model = ["error"], t0 = [nominal_results["t0"]])

# ==================================================================================================

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset")
  parser.add_argument("--emulate", action = "store_true")
  parser.add_argument("--start", type = float, default = 4)
  parser.add_argument("--end", type = float, default = 30)
  parser.add_argument("--step", type = float, default = const.info["T"].magic * 1E-3)
  args = parser.parse_args()

  scan_start_time(args.dataset, np.arange(args.start, args.end, args.step), args.emulate)
