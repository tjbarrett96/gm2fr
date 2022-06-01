import argparse
import re
import gm2fr.constants as const
import gm2fr.io as io
from gm2fr.analysis.Analyzer import Analyzer
import os
import numpy as np

# ==================================================================================================

def scan_parameters(dataset, output_name, parameter_ranges, sim = False):

  if not sim:

    analyzer = Analyzer(
      filename = f"{io.data_path}/FastRotation_{dataset}.root",
      signal_label = f"FastRotation/AllCalos/hHitTime",
      pileup_label = f"FastRotation/AllCalos/hPileupTime",
      output_label = f"{dataset}/{output_name}",
      fr_method = "nine",
      n = 0.108 if dataset not in ("1B", "1C") else 0.120,
      time_units = 1E-9
    )

    nominal_results = np.load(f"{io.results_path}/{dataset}/Nominal/results.npy")

  else:

    analyzer = Analyzer(
      filename = f"{io.sim_path}/{dataset}_sim/simulation.root",
      signal_label = "signal",
      output_label = f"{dataset}/{output_name}",
      output_prefix = "sim_",
      n = 0.108 if dataset not in ("1B", "1C") else 0.120,
      time_units = 1E-6,
      ref_filename = "same"
    )

    nominal_results = np.load(f"{io.results_path}/{dataset}/Simulation/results.npy")

  for parameter, values in parameter_ranges.items():
    if len(values) == 1 and isinstance(values[0], str) and values[0].lower() == "nominal":
      parameter_ranges[parameter] = [nominal_results[parameter][0]]

  analyzer.scan_parameters(**parameter_ranges)

# ==================================================================================================

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", "-d", required = True)
  parser.add_argument("--label", "-l", required = True)
  parser.add_argument("--sim", "-s", action = "store_true")
  parser.add_argument("--parameters", "-p", required = True, nargs = "+")
  args = parser.parse_args()

  parameter_ranges = dict()
  for parameter in args.parameters:
    if re.match(r"\w+:\([\d\.]+,[\d\.]+,[\d\.]+\)$", parameter):
      name, value_range = parameter.split(":")
      start, end, step = [float(item) for item in value_range[1:-1].split(",")]
      parameter_ranges[name] = np.arange(start, end, step)
    elif re.match(r"\w+:\[(?:\w+,?)+\]$", parameter):
      name, value_list = parameter.split(":")
      parameter_ranges[name] = [item for item in value_list.split(",") if item != ""]
      if io.check_all(parameter_ranges.values(), lambda x: re.match(r"[\d\.]+"), x):
        parameter_ranges[name] = [float(item) for item in parameter_ranges[name]]
    elif re.match(r"\w+:[\w\.]+", parameter):
      name, value = parameter.split(":")
      parameter_ranges[name] = [value]
    else:
      print(f"Parameter specification '{parameter}' not understood. Valid formats:")
      print("  name:(start,end,step)")
      print("  name:[value1,value2,...]")
      print("  name:value")
      exit()

  scan_parameters(args.dataset, args.label, parameter_ranges, args.sim)
