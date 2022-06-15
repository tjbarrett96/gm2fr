import argparse
import re
import gm2fr.constants as const
import gm2fr.io as io
from gm2fr.analysis.Analyzer import Analyzer
import os, inspect
import numpy as np

# ==================================================================================================

def scan_parameters(dataset, output_name, constructor_arg_dict = None, analyze_arg_dict = None, sim = False):

  if constructor_arg_dict is None:
    constructor_arg_dict = dict()
  if analyze_arg_dict is None:
    analyze_arg_dict = dict()

  if not sim:

    if "fr_method" not in analyze_arg_dict:
      analyze_arg_dict["fr_method"] = ["nine"]

    analyzer = Analyzer(
      filename = f"{io.data_path}/FastRotation_{dataset}.root",
      signal_label = f"FastRotation/AllCalos/hHitTime",
      pileup_label = f"FastRotation/AllCalos/hPileupTime",
      output_label = f"{dataset}/{output_name}",
      n = 0.108 if dataset not in ("1B", "1C") else 0.120,
      time_units = 1E-9,
      **constructor_arg_dict
    )

    nominal_results = np.load(f"{io.results_path}/{dataset}/Nominal/results.npy", allow_pickle = True)

  else:

    analyzer = Analyzer(
      filename = f"{io.sim_path}/{dataset}_sim/simulation.root",
      signal_label = "signal",
      output_label = f"{dataset}/{output_name}",
      output_prefix = "sim_",
      n = 0.108 if dataset not in ("1B", "1C") else 0.120,
      time_units = 1E-6,
      **constructor_arg_dict
    )

    nominal_results = np.load(f"{io.results_path}/{dataset}/Simulation/results.npy", allow_pickle = True)

  for parameter, values in analyze_arg_dict.items():
    if len(values) == 1 and isinstance(values[0], str) and values[0].lower() == "nominal":
      analyze_arg_dict[parameter] = [nominal_results[parameter][0]]

  analyzer.scan_parameters(**analyze_arg_dict)

# ==================================================================================================

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", "-d", required = True)
  parser.add_argument("--label", "-l", required = True)
  parser.add_argument("--sim", "-s", action = "store_true")
  parser.add_argument("--parameters", "-p", required = True, nargs = "+")
  args = parser.parse_args()

  allowed_constructor_args = inspect.getfullargspec(Analyzer.__init__).args
  allowed_analyze_args = inspect.getfullargspec(Analyzer.analyze).args

  constructor_arg_dict = dict()
  analyze_arg_dict = dict()

  for arg in args.parameters:

    name, value = io.parse_parameter(arg)
    value = io.force_list(value)

    if name in allowed_constructor_args:
      constructor_arg_dict[name] = value
    elif name in allowed_analyze_args:
      analyze_arg_dict[name] = value
    else:
      raise ValueError(f"Parameter '{name}' not recognized.")

  scan_parameters(args.dataset, args.label, constructor_arg_dict, analyze_arg_dict, args.sim)
